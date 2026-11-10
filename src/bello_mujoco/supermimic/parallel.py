"""Single-host data parallelism with local physics and global learning batches."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bello_mujoco.supermimic.collection import (
    CollectorState,
    RunnerState,
    collect_step,
    initialize_collection,
)
from bello_mujoco.supermimic.replay import ReplayBuffer, ReplayStart
from bello_mujoco.supermimic.sampling import SamplingState

AXIS = "environment"
MODEL_WORLD_FIELDS = frozenset(
    (
        "jnt_pos",
        "body_inertia",
        "body_mass",
        "body_ipos",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
    )
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsBatch:
    mjx_model: object
    joint_angle_offset: jax.Array
    torso_com_offset: jax.Array
    physical_randomization: jax.Array
    gravity_fraction: jax.Array

    @classmethod
    def from_environment(cls, env):
        model = env.model
        return cls(*(getattr(model, name) for name in cls.__dataclass_fields__))

    def apply(self, env):
        return replace(
            env,
            model=replace(
                env.model,
                **{name: getattr(self, name) for name in self.__dataclass_fields__},
            ),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EnvironmentArrays:
    motion: object
    kinematics: object
    codec: object
    mapping: object
    mdp: object
    sampler: object
    reset_probabilities: jax.Array

    @classmethod
    def from_environment(cls, env):
        return cls(*(getattr(env, name) for name in cls.__dataclass_fields__))

    def apply(self, env):
        return replace(
            env, **{name: getattr(self, name) for name in self.__dataclass_fields__}
        )


def runner_specs():
    return RunnerState(
        P(),
        CollectorState(P(AXIS), P(AXIS), P(AXIS), P(AXIS), P(AXIS), P(AXIS), P()),
        ReplayBuffer(P(None, AXIS), P(None, AXIS), P(), P()),
        SamplingState(P(), P(), P()),
        P(),
        P(),
        P(),
    )


def collection_layout():
    specs = runner_specs()
    return replace(specs, collector=replace(specs.collector, environment=P()))


def map_partitions(tree, specs, function):
    return jax.tree.map(
        lambda spec, subtree: (
            subtree
            if AXIS not in spec
            else jax.tree.map(lambda value: function(value, spec.index(AXIS)), subtree)
        ),
        specs,
        tree,
        is_leaf=lambda value: isinstance(value, P),
    )


def pmap_axes(specs):
    return jax.tree.map(
        lambda spec: 0 if AXIS in spec else None,
        specs,
        is_leaf=lambda value: isinstance(value, P),
    )


def reduce_collection_metrics(metrics, local_envs):
    rank = jax.lax.axis_index(AXIS)
    selected = jax.lax.pmin(
        jnp.where(metrics["collection/rejected"] > 0, rank, 2**30), AXIS
    )
    selected = jnp.where(selected == 2**30, 0, selected)
    result = {}
    for name, value in metrics.items():
        if name.startswith("diagnostic/") and not name.startswith(
            "diagnostic/invalid/"
        ):
            if name == "diagnostic/environment":
                value = jnp.where(value >= 0, value + rank * local_envs, -1)
            result[name] = jax.lax.psum(jnp.where(rank == selected, value, 0), AXIS)
        elif name in (
            "collection/completed",
            "collection/failures",
            "collection/rejected",
        ) or name.startswith("diagnostic/invalid/"):
            result[name] = jax.lax.psum(value, AXIS)
        elif name.startswith(("physics/", "invalid/")):
            result[name] = jax.lax.pmax(value, AXIS)
        else:
            result[name] = jax.lax.pmean(value, AXIS)
    return result


@dataclass(frozen=True)
class DataParallel:
    mesh: Mesh

    @classmethod
    def create(cls, count, *, devices=None):
        available = jax.local_devices() if devices is None else tuple(devices)
        if jax.process_count() != 1:
            raise ValueError("SuperMimic data parallelism requires one host process")
        if count <= 0 or len(available) < count:
            raise ValueError(
                f"Requested {count} devices; only {len(available)} are available"
            )
        return cls(Mesh(np.asarray(available[:count]), (AXIS,)))

    @property
    def count(self):
        return int(self.mesh.size)

    @property
    def replicated(self):
        return NamedSharding(self.mesh, P())

    def put(self, tree, specs):
        return jax.device_put(tree, self.shardings(specs))

    def shardings(self, specs):
        return jax.tree.map(lambda spec: NamedSharding(self.mesh, spec), specs)

    def environment(self, env):
        return jax.device_put(
            EnvironmentArrays.from_environment(env), self.replicated
        ).apply(env)

    def physics(self, env):
        batch = PhysicsBatch.from_environment(env)
        randomized = env.config.model_randomization.enabled
        model_specs = jax.tree_util.tree_map_with_path(
            lambda path, _: (
                P(AXIS)
                if randomized
                and path
                and getattr(path[0], "name", None) in MODEL_WORLD_FIELDS
                else P()
            ),
            batch.mjx_model,
        )
        if randomized:
            for name in MODEL_WORLD_FIELDS:
                if getattr(model_specs, name) != P(AXIS):
                    raise ValueError(f"MJX world field was not partitioned: {name}")
        world = P(AXIS) if randomized else P()
        specs = PhysicsBatch(model_specs, world, world, world, world)
        return self.put(batch, specs), specs

    def local_config(self, config):
        for name in ("num_envs", "model_batch_size", "imagined_batch_size"):
            if getattr(config, name) % self.count:
                raise ValueError(f"{name} must divide evenly across num_devices")
        return replace(
            config,
            num_envs=config.num_envs // self.count,
            max_replay_bytes=config.max_replay_bytes // self.count,
        )

    def pack(self, tree, specs):
        def split(value, axis):
            size = value.shape[axis]
            if size % self.count:
                raise ValueError(
                    "A physics batch dimension does not divide across devices"
                )
            shape = (
                value.shape[:axis]
                + (self.count, size // self.count)
                + value.shape[axis + 1 :]
            )
            return jnp.moveaxis(value.reshape(shape), axis, 0)

        return map_partitions(tree, specs, split)

    def unpack(self, tree, specs):
        def merge(value, axis):
            value = jnp.moveaxis(value, 0, axis)
            shape = (
                value.shape[:axis]
                + (value.shape[axis] * value.shape[axis + 1],)
                + value.shape[axis + 2 :]
            )
            return value.reshape(shape)

        return map_partitions(tree, specs, merge)

    def collection(self, env, config, physics, physics_specs):
        local = self.local_config(config)
        capacities = {}
        for name in ("warp_naconmax", "warp_naccdmax"):
            value = getattr(env.model, name)
            if value is not None and value % self.count:
                raise ValueError(f"{name} must divide evenly across num_devices")
            capacities[name] = None if value is None else value // self.count
        env = replace(env, model=replace(env.model, **capacities))
        specs = runner_specs()
        arrays = jax.device_put(
            EnvironmentArrays.from_environment(env), self.replicated
        )

        def initialize(learner, key, bank, physical, environment):
            current = physical.apply(
                replace(environment.apply(env), reference_bank=bank)
            )
            return initialize_collection(current, local, learner, key, axis_name=AXIS)

        def collect(runner, bank, physical, environment):
            current = physical.apply(
                replace(environment.apply(env), reference_bank=bank)
            )
            runner, metrics = collect_step(runner, current, local, axis_name=AXIS)
            return runner, reduce_collection_metrics(metrics, local.num_envs)

        devices = tuple(self.mesh.devices.flat)
        initialize = jax.pmap(
            initialize,
            axis_name=AXIS,
            devices=devices,
            axis_size=self.count,
            in_axes=(None, None, None, pmap_axes(physics_specs), None),
            out_axes=pmap_axes(specs),
        )
        collect = jax.pmap(
            collect,
            axis_name=AXIS,
            devices=devices,
            axis_size=self.count,
            in_axes=(pmap_axes(specs), None, pmap_axes(physics_specs), None),
            out_axes=(pmap_axes(specs), None),
        )
        packed_physics = self.pack(physics, physics_specs)
        layout = collection_layout()

        @jax.jit
        def initialize_global(learner, key, bank, physical, environment):
            result = initialize(
                learner, key, bank, self.pack(physical, physics_specs), environment
            )
            return self.unpack(result, layout)

        @jax.jit
        def collect_global(runner, bank, physical, environment):
            result, metrics = collect(
                self.pack(runner, layout), bank, physical, environment
            )
            return self.unpack(result, layout), metrics

        return (
            lambda learner, key, bank, physical: initialize_global(
                learner, key, bank, physical, arrays
            ),
            lambda runner, bank: collect_global(runner, bank, packed_physics, arrays),
        )

    def sample_windows(self, replay, key, batch_size, length):
        if batch_size % self.count:
            raise ValueError("Model batch must divide evenly across devices")

        def sample(replay, key):
            key = jax.random.fold_in(key, jax.lax.axis_index(AXIS))
            return replay.sample_windows(key, batch_size // self.count, length)

        return jax.shard_map(
            sample,
            mesh=self.mesh,
            in_specs=(ReplayBuffer(P(None, AXIS), P(None, AXIS), P(), P()), P()),
            out_specs=P(AXIS),
        )(replay, key)

    def sample_starts(self, replay, key, batch_size, **kwargs):
        if batch_size % self.count:
            raise ValueError("Imagined batch must divide evenly across devices")

        def sample(replay, key):
            key = jax.random.fold_in(key, jax.lax.axis_index(AXIS))
            starts = replay.sample_starts(key, batch_size // self.count, **kwargs)
            valid = jax.lax.pmin(starts.valid.astype(jnp.int32), AXIS).astype(bool)
            return replace(starts, valid=valid)

        return jax.shard_map(
            sample,
            mesh=self.mesh,
            in_specs=(ReplayBuffer(P(None, AXIS), P(None, AXIS), P(), P()), P()),
            out_specs=ReplayStart(P(AXIS), P(AXIS), P(AXIS), P()),
        )(replay, key)
