"""Sole reference geometry and shared NumPy/JAX terrain support calculations."""

from dataclasses import dataclass

import mujoco
import numpy as np


def sole_support_positions(
    *,
    geom_position,
    geom_rotation,
    geom_size,
    geom_type,
    geom_body_ids,
    foot_body_ids,
    terrain_up,
    terrain_height,
    array_module,
):
    """Lowest terrain-relative primitive support point for each foot.

    NumPy evaluation and JAX rollouts execute this same calculation. Boxes use
    all eight corners; curved primitives use the local terrain normal.
    """
    xp = array_module
    local_down = xp.einsum("gji,gj->gi", geom_rotation, -terrain_up)
    radial = local_down[:, :2]
    radial_norm = xp.linalg.norm(radial, axis=1, keepdims=True)
    unit_radial = xp.where(
        radial_norm > 1.0e-8,
        radial / xp.maximum(radial_norm, 1.0e-8),
        xp.zeros_like(radial),
    )
    sphere = geom_size[:, :1] * local_down
    axial = xp.sign(local_down[:, 2]) * geom_size[:, 1]
    capsule = sphere + xp.stack(
        [xp.zeros_like(axial), xp.zeros_like(axial), axial], axis=1
    )
    ellipsoid = (
        geom_size**2
        * local_down
        / xp.maximum(
            xp.sqrt(xp.sum((geom_size * local_down) ** 2, axis=1, keepdims=True)),
            1.0e-8,
        )
    )
    cylinder = xp.concatenate([geom_size[:, :1] * unit_radial, axial[:, None]], axis=1)
    support = sphere
    for shape, candidate in (
        (mujoco.mjtGeom.mjGEOM_CAPSULE, capsule),
        (mujoco.mjtGeom.mjGEOM_ELLIPSOID, ellipsoid),
        (mujoco.mjtGeom.mjGEOM_CYLINDER, cylinder),
    ):
        support = xp.where((geom_type == int(shape))[:, None], candidate, support)
    support_world = geom_position + xp.einsum("gij,gj->gi", geom_rotation, support)
    signs = xp.asarray(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=xp.float32,
    )
    corners = geom_position[:, None, :] + xp.einsum(
        "gij,gkj->gki", geom_rotation, signs[None] * geom_size[:, None]
    )
    corner_clearance = corners[:, :, 2] - terrain_height(corners[:, :, :2])
    box_support = corners[xp.arange(len(corners)), xp.argmin(corner_clearance, axis=1)]
    support_world = xp.where(
        (geom_type == int(mujoco.mjtGeom.mjGEOM_BOX))[:, None],
        box_support,
        support_world,
    )
    clearance = support_world[:, 2] - terrain_height(support_world[:, :2])
    belongs = foot_body_ids[:, None] == geom_body_ids[None, :]
    lowest = xp.argmin(xp.where(belongs, clearance[None], xp.inf), axis=1)
    return support_world[lowest]


@dataclass(frozen=True)
class SoleGeometry:
    centers: np.ndarray
    corners: np.ndarray
    nominal_width: float
    minimum_separation: float


def sole_geometry(model, margin: float) -> SoleGeometry:
    """Build conservative toe/heel corner samples in each foot body's frame.

    Bounds are analytic primitive bounds in the nominal robot heading frame.
    They describe the sole footprint, not the ankle-link origin or visual mesh.
    """
    mj_model = model.mj_model
    data = mujoco.MjData(mj_model)
    data.qpos[:] = model.default_qpos
    mujoco.mj_forward(mj_model, data)
    forward = np.asarray(model.robot.locomotion_forward_axis)
    root = mj_model.body(model.robot.root_body_name).id
    world_forward = data.xmat[root].reshape(3, 3) @ forward
    world_forward[2] = 0.0
    world_forward /= np.linalg.norm(world_forward)
    left = np.cross([0.0, 0.0, 1.0], world_forward)
    basis = np.column_stack((world_forward, left, [0.0, 0.0, 1.0]))
    centers, corners, widths, world_centers = [], [], [], []
    if len(model.foot_body_ids) != 2:
        raise ValueError("foot-placement rewards require ordered left/right feet")
    for body in model.foot_body_ids:
        lows, highs = [], []
        for geom in model.foot_geom_ids:
            if mj_model.geom_bodyid[geom] != body:
                continue
            rotation = basis.T @ data.geom_xmat[geom].reshape(3, 3)
            center = basis.T @ data.geom_xpos[geom]
            size = mj_model.geom_size[geom]
            kind = mj_model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(rotation) @ size
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = size[0] * np.linalg.norm(rotation[:, :2], axis=1) + size[
                    1
                ] * np.abs(rotation[:, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = np.full(3, size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = size[0] + size[1] * np.abs(rotation[:, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent = np.linalg.norm(rotation * size, axis=1)
            else:
                raise ValueError(
                    "sole reward geometry requires primitive support collisions"
                )
            lows.append(center - extent)
            highs.append(center + extent)
        if not lows:
            raise ValueError("each foot needs registered sole collision primitives")
        low, high = np.min(lows, axis=0), np.max(highs, axis=0)
        center = (low + high) * 0.5
        center[2] = low[2]
        samples = np.asarray(
            [[x, y, low[2]] for x in (low[0], high[0]) for y in (low[1], high[1])]
        )
        body_rotation = data.xmat[body].reshape(3, 3)
        centers.append(body_rotation.T @ (basis @ center - data.xpos[body]))
        corners.append((samples @ basis.T - data.xpos[body]) @ body_rotation)
        widths.append(high[1] - low[1])
        world_centers.append(center)
    nominal_width = float(world_centers[0][1] - world_centers[1][1])
    minimum_separation = float(sum(widths) * 0.5 + margin)
    if nominal_width <= minimum_separation:
        raise ValueError("nominal stance must exceed sole widths plus clearance margin")
    return SoleGeometry(
        np.asarray(centers, dtype=np.float32),
        np.asarray(corners, dtype=np.float32),
        nominal_width,
        minimum_separation,
    )
