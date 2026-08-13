import base64
import json
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import bello_mujoco.training.vast as vast
import bello_mujoco.training.vast_remote as vast_remote
from bello_mujoco.training.vast import (
    GitSelection,
    VastLifecycle,
    VastOffer,
    VastSdkClient,
    VastTrainConfig,
    credential,
    create_available_instance,
    destroy_with_retries,
    find_offers,
    launch_pipeline,
    offer_query,
    run_remote_training,
    safe_exception_message,
    validate_config,
)


def make_offer(
    offer_id: int = 123,
    *,
    price: float = 0.8,
    value: float = 10.0,
    location: str = "Seoul, KR",
):
    return {
        "id": offer_id,
        "gpu_name": "RTX_5090",
        "dph_total": price,
        "dlperf_per_dphtotal": value,
        "reliability": 0.995,
        "geolocation": location,
        "cpu_ram": 65536,
        "gpu_ram": 32768,
        "num_gpus": 1,
        "cuda_max_good": 13.0,
        "disk_space": 100.0,
        "direct_port_count": 4,
    }


class SearchApi:
    def __init__(self, responses):
        self.responses = list(responses)
        self.queries = []

    def search_offers(self, query, *, order, limit, storage):
        self.queries.append((query, order, limit, storage))
        return self.responses.pop(0)


class LifecycleApi:
    def __init__(self):
        self.destroyed = []
        self.created = []

    def create_instance(self, offer_id, **kwargs):
        self.created.append((offer_id, kwargs))
        return 456

    def show_instance(self, instance_id):
        if instance_id in self.destroyed:
            return None
        return {"id": instance_id, "actual_status": "running"}

    def show_instances(self):
        return []

    def destroy_instance(self, instance_id):
        self.destroyed.append(instance_id)


def valid_config(**kwargs) -> VastTrainConfig:
    kwargs.setdefault("experiment", "locomotion")
    return VastTrainConfig(
        max_price=1.25,
        branch="main",
        skip_confirmation=True,
        **kwargs,
    )


def test_fetch_origin_retries_transient_git_failure(monkeypatch, tmp_path) -> None:
    attempts = []
    sleeps = []

    def fake_run_git(repo_root, *args):
        attempts.append((repo_root, args))
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(
                128, ["git", *args], stderr="temporary network failure"
            )
        return ""

    monkeypatch.setattr(vast, "run_git", fake_run_git)

    vast.fetch_origin(tmp_path, sleep=sleeps.append)

    assert len(attempts) == 3
    assert sleeps == [2.0, 4.0]


def test_fetch_origin_reports_git_stderr_after_retries(monkeypatch, tmp_path) -> None:
    def fake_run_git(repo_root, *args):
        raise subprocess.CalledProcessError(
            128, ["git", *args], stderr="could not resolve host: github.com"
        )

    monkeypatch.setattr(vast, "run_git", fake_run_git)

    with pytest.raises(RuntimeError, match="could not resolve host: github.com"):
        vast.fetch_origin(tmp_path, retries=2, sleep=lambda _: None)


def selection() -> GitSelection:
    return GitSelection(
        branch="main",
        commit="a" * 40,
        config_path=PurePosixPath("configs/profiles/train/locomotion.yaml"),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
    )


def motion_config(**kwargs) -> VastTrainConfig:
    kwargs.setdefault("experiment", "motion")
    kwargs.setdefault(
        "motion_path",
        Path(
            "motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz"
        ),
    )
    return VastTrainConfig(
        max_price=1.25,
        branch="main",
        skip_confirmation=True,
        **kwargs,
    )


def motion_selection() -> GitSelection:
    return GitSelection(
        branch="main",
        commit="b" * 40,
        config_path=PurePosixPath("configs/profiles/train/motion.yaml"),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
        motion_path=PurePosixPath(
            "motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz"
        ),
    )


def twist_dataset(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in (
        "train.npz",
        "evaluation.npz",
        "heldout_manifest.json",
        "conversion_report.json",
    ):
        (tmp_path / name).write_bytes(name.encode())
    index = {
        "schema_version": 3,
        "dataset_fingerprint": "sha256:" + "c" * 64,
        "train_sampling": "frame_count_times_manifest_weight",
        "split_files": {
            "train": "train.npz",
            "evaluation": "evaluation.npz",
        },
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return path


def twist_config(tmp_path: Path, *, stage: str = "teacher", **kwargs):
    dataset_path = twist_dataset(tmp_path / "dataset")
    values = {
        "experiment": f"twist_{stage}",
        "dataset_path": dataset_path,
    }
    values.update(kwargs)
    return valid_config(**values)


def sonic_dataset(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    root = tmp_path / "sonic_dataset"
    root.mkdir(parents=True)
    for relative in (
        "heldout_manifest.json",
        "conversion_report.json",
        "train.npz",
        "evaluation.npz",
    ):
        path = root / relative
        path.write_bytes(relative.encode())
    index = {
        "dataset_fingerprint": "sha256:" + "d" * 64,
        "split_files": {
            "train": {"file": "train.npz"},
            "evaluation": {"file": "evaluation.npz"},
        },
    }
    index_path = root / "index.json"
    index_path.write_text("{}", encoding="utf-8")
    import bello_mujoco.motion.sonic_dataset as sonic

    monkeypatch.setattr(sonic, "load_sonic_dataset_index", lambda path: index)
    return index_path, index


def sonic_artifact(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "sonic_artifact"
    root.mkdir()
    for name in ("manifest.json", "weights.npz", "golden.npz"):
        (root / name).write_bytes(name.encode())
    import bello_mujoco.sonic_policy as sonic_policy

    monkeypatch.setattr(
        sonic_policy,
        "load_sonic_release_artifact",
        lambda path: object(),
    )
    return root


@pytest.mark.parametrize("max_price", [None, 0.0, float("nan"), float("inf")])
def test_validate_config_requires_finite_positive_price(max_price) -> None:
    with pytest.raises(ValueError, match="max-price"):
        validate_config(VastTrainConfig(max_price=max_price, branch="main"))


def test_vast_experiment_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="--experiment"):
        validate_config(
            valid_config(experiment="unsupported")  # type: ignore[arg-type]
        )


def test_twist_vast_config_requires_explicit_stage_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset-path is required"):
        validate_config(valid_config(experiment="twist_teacher"))
    with pytest.raises(ValueError, match="teacher-checkpoint-path is required"):
        validate_config(twist_config(tmp_path, stage="student"))
    with pytest.raises(ValueError, match="only valid for TWIST"):
        validate_config(valid_config(dataset_path=Path("dataset/index.json")))
    validate_config(
        valid_config(
            experiment="twist2",
            dataset_path=twist_dataset(tmp_path / "twist2_dataset"),
        )
    )


def test_sonic_vast_config_requires_dataset_and_release_artifact() -> None:
    with pytest.raises(ValueError, match="dataset-path is required"):
        validate_config(valid_config(experiment="sonic"))
    with pytest.raises(ValueError, match="sonic-artifact-path is required"):
        validate_config(
            valid_config(
                experiment="sonic",
                dataset_path=Path("dataset/index.json"),
            )
        )
    validate_config(
        valid_config(
            experiment="sonic",
            dataset_path=Path("dataset/index.json"),
            sonic_artifact_path=Path("artifact"),
            total_timesteps=1_425_408,
            num_envs=4096,
            checkpoint_every_updates=29,
        )
    )


def test_sonic_vast_resume_requires_explicit_continuation_contract() -> None:
    base = {
        "experiment": "sonic",
        "dataset_path": Path("dataset/index.json"),
        "sonic_artifact_path": Path("artifact"),
        "resume_path": Path("checkpoint/final"),
    }
    with pytest.raises(ValueError, match="requires explicit"):
        validate_config(valid_config(**base))
    validate_config(
        valid_config(
            **base,
            allow_git_commit_mismatch=True,
            total_timesteps=71_270_400,
            num_envs=4096,
        )
    )
    with pytest.raises(ValueError, match="only valid for SONIC"):
        validate_config(valid_config(resume_path=Path("checkpoint/final")))


def test_sonic_dataset_selection_preserves_canonical_split_files(
    tmp_path: Path, monkeypatch
) -> None:
    index_path, index = sonic_dataset(tmp_path, monkeypatch)

    fingerprint, paths, root = vast._sonic_dataset_files(index_path)

    assert fingerprint == index["dataset_fingerprint"]
    assert root == index_path.parent.resolve()
    assert {path.relative_to(root).as_posix() for path in paths} == {
        "index.json",
        "heldout_manifest.json",
        "conversion_report.json",
        "train.npz",
        "evaluation.npz",
    }
    (root / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="canonical files"):
        vast._sonic_dataset_files(index_path)


def test_sonic_release_selection_requires_exact_pinned_export(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = sonic_artifact(tmp_path, monkeypatch)

    assert vast._sonic_release_artifact(artifact) == artifact.resolve()

    (artifact / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="files differ"):
        vast._sonic_release_artifact(artifact)


def test_sonic_resume_selection_binds_dataset_release_and_budget(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")
    (checkpoint / "state").mkdir()
    (checkpoint / "continuation").mkdir()
    fingerprint = "sha256:" + "d" * 64
    commit = "a" * 40

    class FakeSonicMetadata:
        dataset = SimpleNamespace(dataset_fingerprint=fingerprint)
        source_revision = "b" * 40
        release_checkpoint_sha256 = "sha256:" + "c" * 64
        release_artifact_sha256 = "sha256:" + "e" * 64
        ppo = SimpleNamespace(num_envs=4096, num_steps=24)

    import bello_mujoco.sonic_policy as sonic_policy
    import bello_mujoco.training.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "SonicCheckpointMetadata", FakeSonicMetadata)
    monkeypatch.setattr(
        checkpoint_module,
        "load_checkpoint_metadata",
        lambda path, **kwargs: FakeSonicMetadata(),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "load_checkpoint_update",
        lambda path, **kwargs: 116,
    )
    monkeypatch.setattr(
        sonic_policy,
        "load_sonic_release_artifact",
        lambda path: SimpleNamespace(
            checkpoint_sha256="c" * 64,
            artifact_sha256="e" * 64,
        ),
    )

    assert (
        vast._sonic_resume_checkpoint(
            checkpoint,
            dataset_fingerprint=fingerprint,
            release_artifact_path=tmp_path / "release",
            selected_commit=commit,
            allow_git_commit_mismatch=True,
            total_timesteps=71_270_400,
            num_envs=4096,
        )
        == checkpoint.resolve()
    )
    with pytest.raises(ValueError, match="budget ends before"):
        vast._sonic_resume_checkpoint(
            checkpoint,
            dataset_fingerprint=fingerprint,
            release_artifact_path=tmp_path / "release",
            selected_commit=commit,
            allow_git_commit_mismatch=True,
            total_timesteps=115 * 4096 * 24,
            num_envs=4096,
        )


def test_twist_dataset_selection_is_strict_and_portable(tmp_path: Path) -> None:
    index_path = twist_dataset(tmp_path)

    fingerprint, paths = vast._twist_dataset_files(index_path)

    assert fingerprint == "sha256:" + "c" * 64
    assert {path.name for path in paths} == {
        "index.json",
        "train.npz",
        "evaluation.npz",
        "heldout_manifest.json",
        "conversion_report.json",
    }
    (tmp_path / "train.npz").unlink()
    with pytest.raises(ValueError, match="files are missing"):
        vast._twist_dataset_files(index_path)


def test_twist_teacher_selection_binds_dataset_fingerprint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()
    (checkpoint / "state").mkdir()
    (checkpoint / "continuation").mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "task": "twist_teacher",
                    "dataset": {"dataset_fingerprint": "sha256:" + "c" * 64},
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        vast._twist_teacher_checkpoint(
            checkpoint, dataset_fingerprint="sha256:" + "c" * 64
        )
        == checkpoint.resolve()
    )
    with pytest.raises(ValueError, match="dataset fingerprint differs"):
        vast._twist_teacher_checkpoint(
            checkpoint, dataset_fingerprint="sha256:" + "d" * 64
        )


def test_motion_git_selection_verifies_clip_at_selected_commit(
    monkeypatch, tmp_path
) -> None:
    commit = "c" * 40
    checked_objects = []

    monkeypatch.setattr(vast, "remote_branches", lambda repo_root: ["main"])

    def fake_run_git(repo_root, *args):
        assert repo_root == tmp_path
        if args == ("rev-parse", "refs/remotes/origin/main^{commit}"):
            return commit
        if args == ("ls-tree", commit, "--", "deps/bello-stack"):
            return f"160000 commit {'s' * 40}\tdeps/bello-stack"
        if args[:2] == ("cat-file", "-e"):
            checked_objects.append(args[2])
            return ""
        if args == ("remote", "get-url", "origin"):
            return "https://github.com/MotomindKR/bello_mujoco.git"
        raise AssertionError(args)

    monkeypatch.setattr(vast, "run_git", fake_run_git)

    resolved = vast.resolve_git_selection(tmp_path, motion_config())

    assert resolved.motion_path == motion_selection().motion_path
    assert f"{commit}:{resolved.motion_path}" in checked_objects
    assert f"{commit}:.gitmodules" in checked_objects


@pytest.mark.parametrize(
    "remote_dir",
    [
        "/",
        "/workspace",
        "/tmp/bello",
        "workspace/bello",
        "/workspace/../etc",
        "/workspace/run/../../etc",
        "/workspace/run/../other",
        "//workspace/run",
        "/workspace//run",
        "/workspace/./run",
        "/workspace/run/",
    ],
)
def test_validate_config_restricts_destructive_remote_path(remote_dir: str) -> None:
    with pytest.raises(ValueError, match="normalized strict child of /workspace"):
        validate_config(valid_config(remote_dir=remote_dir))


@pytest.mark.parametrize(
    "remote_dir",
    ["/workspace/bello", "/workspace/jobs/bello"],
)
def test_validate_config_accepts_normalized_workspace_children(
    remote_dir: str,
) -> None:
    validate_config(valid_config(remote_dir=remote_dir))


@pytest.mark.parametrize(
    "state_dir",
    [Path("/tmp/vast-state"), Path("."), Path("state"), Path(".vast/../state")],
)
def test_validate_config_restricts_lifecycle_state_to_vast_directory(
    state_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="lifecycle-state-dir"):
        validate_config(valid_config(lifecycle_state_dir=state_dir))


def test_launch_pipeline_isolates_lifecycle_state_directory(
    monkeypatch, tmp_path: Path
) -> None:
    original_state = tmp_path / ".vast" / "active_instance.json"
    original_state.parent.mkdir(parents=True)
    original_state.write_text("original-controller")
    observed = {}

    class Lock:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    class Lifecycle:
        def __init__(self, api, config, state_path, label):
            observed["state_path"] = state_path

        def run(self, *args, **kwargs):
            return None

    def fake_pipeline_lock(state_dir):
        observed["state_dir"] = state_dir
        return Lock()

    monkeypatch.setattr(vast, "resolve_git_selection", lambda *args: selection())
    monkeypatch.setattr(vast, "credential", lambda *args, **kwargs: "secret")
    monkeypatch.setattr(vast, "pipeline_lock", fake_pipeline_lock)
    monkeypatch.setattr(vast, "VastLifecycle", Lifecycle)

    assert (
        launch_pipeline(
            tmp_path,
            valid_config(lifecycle_state_dir=Path(".vast/yaw-run")),
            api_factory=lambda key: object(),
        )
        == 0
    )
    assert observed == {
        "state_dir": tmp_path / ".vast" / "yaw-run",
        "state_path": tmp_path / ".vast" / "yaw-run" / "active_instance.json",
    }
    assert original_state.read_text() == "original-controller"


def test_validate_config_requires_bounded_job_timeouts() -> None:
    with pytest.raises(ValueError, match="job and progress"):
        validate_config(valid_config(job_timeout_seconds=0))
    with pytest.raises(ValueError, match="must not exceed"):
        validate_config(
            valid_config(job_timeout_seconds=10, progress_timeout_seconds=11)
        )


def test_offer_query_contains_hardware_and_price_constraints() -> None:
    query = offer_query(valid_config())

    assert "gpu_name=RTX_5090" in query
    assert "cpu_ram>=64" in query
    assert "disk_space>=80" in query
    assert "cuda_max_good>=13.0" in query
    assert "dph_total<=1.25" in query


def test_find_offers_prefers_nearby_regions_before_global_value() -> None:
    api = SearchApi(
        [
            [
                make_offer(1, value=5.0, location="Tokyo, JP"),
                make_offer(2, value=12.0, location="Singapore, SG"),
            ],
        ]
    )

    offers = find_offers(api, valid_config())

    assert [offer.id for offer in offers] == [1, 2]
    assert len(api.queries) == 1
    assert api.queries[0][3] == 80.0


def test_find_offers_rechecks_returned_price_cap() -> None:
    api = SearchApi([[make_offer(price=1.26)]])

    with pytest.raises(RuntimeError, match="no Vast offers"):
        find_offers(api, valid_config())


def test_create_available_instance_tries_next_explicitly_unavailable_offer() -> None:
    class RacingApi(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            if offer_id == 1:
                error = RuntimeError("offer disappeared")
                error.response = type("Response", (), {"status_code": 410})()
                raise error
            return super().create_instance(offer_id, **kwargs)

    api = RacingApi()
    offers = [
        VastOffer.from_mapping(make_offer(1)),
        VastOffer.from_mapping(make_offer(2)),
    ]

    assert (
        create_available_instance(
            api,
            offers,
            valid_config(),
            "test-label",
            on_before_create=lambda: None,
            on_offer_unavailable=lambda: None,
            sleep=lambda _: None,
        )
        == 456
    )
    assert [offer_id for offer_id, _ in api.created] == [2]


def test_safe_exception_message_redacts_query_credentials() -> None:
    error = RuntimeError(
        "410 Gone: https://console.vast.ai/api/v0/asks/1/?api_key=secret&x=1"
    )

    message = safe_exception_message(error)

    assert "secret" not in message
    assert "api_key=<redacted>&x=1" in message


def test_nominal_64_gb_offer_accepts_vast_usable_ram_reporting() -> None:
    api = SearchApi([[make_offer() | {"cpu_ram": 64469}]])

    offers = find_offers(api, valid_config())

    assert [offer.id for offer in offers] == [123]


def test_sdk_client_uses_pinned_sdk_response_contract() -> None:
    class PinnedClient:
        def __init__(self):
            self.destroyed = []

        def search_offers(self, **kwargs):
            assert kwargs["storage"] == 80.0
            return [make_offer()]

        def create_instance(self, **kwargs):
            assert kwargs["id"] == 123
            return {"new_contract": 456}

        def show_instance(self, **kwargs):
            return {"id": kwargs["id"], "actual_status": "running"}

        def show_instances(self):
            return [{"id": 456}]

        def destroy_instance(self, **kwargs):
            self.destroyed.append(kwargs["id"])

    raw_client = PinnedClient()
    client = object.__new__(VastSdkClient)
    client._client = raw_client

    assert client.search_offers("query", order="score-", limit=1, storage=80.0) == [
        make_offer()
    ]
    assert client.create_instance(123, image="test") == 456
    assert client.show_instance(456)["actual_status"] == "running"
    assert client.show_instances() == [{"id": 456}]
    client.destroy_instance(456)
    assert raw_client.destroyed == [456]


def test_credential_prefers_environment_over_keyring(monkeypatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "environment-secret")
    monkeypatch.setattr(
        vast.keyring,
        "get_password",
        lambda *args: (_ for _ in ()).throw(AssertionError("keyring should not run")),
    )

    assert credential("VAST_API_KEY", "vast", "api_key", required=True) == (
        "environment-secret"
    )


def test_credential_reads_python_keyring(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        vast.keyring,
        "get_password",
        lambda service, username: f"{service}-{username}-secret",
    )

    assert credential("WANDB_API_KEY", "wandb", "api_key", required=True) == (
        "wandb-api_key-secret"
    )


def test_wait_for_ssh_uses_direct_port_mapping(monkeypatch) -> None:
    class Api:
        def show_instance(self, instance_id):
            assert instance_id == 123
            return {
                "public_ipaddr": "192.0.2.10",
                "ports": {"22/tcp": [{"HostPort": "22022"}]},
            }

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert 0 < kwargs["timeout"] <= vast_remote.SSH_SUBPROCESS_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vast_remote.subprocess, "run", fake_run)

    target = vast.wait_for_ssh(Api(), 123, 30, sleep=lambda _: None)

    assert target == ("root", "192.0.2.10", 22022)
    assert commands == [vast_remote.ssh_command(target, "true")]


def test_upload_route_probe_measures_payload_throughput(monkeypatch) -> None:
    moments = iter((10.0, 12.0))
    monkeypatch.setattr(vast_remote.time, "monotonic", lambda: next(moments))
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    speed = vast_remote.measure_upload_mib_per_second(
        ("root", "example.com", 22001),
        size_mib=8,
        timeout_seconds=30,
        runner=runner,
    )

    assert speed == 4.0
    assert len(calls[0][1]["input"]) == 8 * 1024 * 1024
    assert calls[0][1]["timeout"] == 30


def test_upload_route_probe_treats_timeout_as_zero_speed() -> None:
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    assert (
        vast_remote.measure_upload_mib_per_second(
            ("root", "example.com", 22001),
            size_mib=1,
            timeout_seconds=30,
            runner=runner,
        )
        == 0.0
    )


def test_destroy_treats_not_found_after_api_error_as_success() -> None:
    class AlreadyGoneApi:
        def destroy_instance(self, instance_id):
            error = RuntimeError("not found")
            error.response = type("Response", (), {"status_code": 404})()
            raise error

        def show_instance(self, instance_id):
            return None

    assert destroy_with_retries(AlreadyGoneApi(), 123, retries=1)


def test_vast_error_classification_prefers_structured_http_status() -> None:
    class Response:
        status_code = 404

    error = RuntimeError("unrelated message")
    error.response = Response()

    assert vast._is_explicit_not_found(error)
    assert vast._is_explicit_offer_unavailable(error)


@pytest.mark.parametrize("status", [400, 409, 422])
def test_vast_definite_create_rejections_are_not_ambiguous(status) -> None:
    error = RuntimeError("request rejected")
    error.response = type("Response", (), {"status_code": status})()

    assert vast._is_explicit_offer_unavailable(error)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_vast_auth_rate_and_server_errors_remain_ambiguous(status) -> None:
    error = RuntimeError("uncertain create result")
    error.response = type("Response", (), {"status_code": status})()

    assert not vast._is_explicit_offer_unavailable(error)


def test_destroy_does_not_treat_transient_api_errors_as_success() -> None:
    class UnreachableApi:
        def destroy_instance(self, instance_id):
            raise TimeoutError("gateway timed out")

        def show_instance(self, instance_id):
            raise TimeoutError("gateway timed out")

        def show_instances(self):
            raise TimeoutError("gateway timed out")

    assert not destroy_with_retries(UnreachableApi(), 123, retries=1)


def test_destroy_requires_status_or_canonical_absence_confirmation() -> None:
    class Api:
        def __init__(self, response):
            self.response = response

        def destroy_instance(self, instance_id):
            pass

        def show_instance(self, instance_id):
            return self.response

    assert not destroy_with_retries(Api({"id": 123}), 123, retries=1)
    assert destroy_with_retries(Api(None), 123, retries=1)


def test_ambiguous_create_never_rents_a_second_offer() -> None:
    class AmbiguousApi(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            self.created.append((offer_id, kwargs))
            raise TimeoutError("response lost")

        def show_instances(self):
            raise TimeoutError("listing unavailable")

    api = AmbiguousApi()
    offers = [
        VastOffer.from_mapping(make_offer(1)),
        VastOffer.from_mapping(make_offer(2)),
    ]

    with pytest.raises(RuntimeError, match="ambiguous Vast rental"):
        create_available_instance(
            api,
            offers,
            valid_config(),
            "test-label",
            on_before_create=lambda: None,
            on_offer_unavailable=lambda: None,
            sleep=lambda _: None,
        )

    assert [offer_id for offer_id, _ in api.created] == [1]


def test_ambiguous_create_with_empty_listing_never_rents_again() -> None:
    class LostResponseApi(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            self.created.append((offer_id, kwargs))
            raise TimeoutError("response lost")

        def show_instances(self):
            return []

    api = LostResponseApi()
    offers = [
        VastOffer.from_mapping(make_offer(1)),
        VastOffer.from_mapping(make_offer(2)),
    ]

    with pytest.raises(RuntimeError, match="result for offer 1 is ambiguous"):
        create_available_instance(
            api,
            offers,
            valid_config(),
            "test-label",
            on_before_create=lambda: None,
            on_offer_unavailable=lambda: None,
            sleep=lambda _: None,
        )

    assert [offer_id for offer_id, _ in api.created] == [1]


def test_ambiguous_create_recovers_exact_labeled_instance() -> None:
    class RecoveredApi(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            self.created.append((offer_id, kwargs))
            raise TimeoutError("response lost")

        def show_instances(self):
            return [{"id": 789, "label": "test-label"}]

    api = RecoveredApi()

    assert (
        create_available_instance(
            api,
            [VastOffer.from_mapping(make_offer(1))],
            valid_config(),
            "test-label",
            on_before_create=lambda: None,
            on_offer_unavailable=lambda: None,
            sleep=lambda _: None,
        )
        == 789
    )
    assert len(api.created) == 1


def test_active_state_write_is_atomic_and_round_trips(tmp_path) -> None:
    state_path = tmp_path / ".vast" / "active_instance.json"

    vast.write_active_state(state_path, 123, "test-label")

    state = vast.read_lifecycle_state(state_path)
    assert state is not None
    assert state.phase == "active"
    assert state.instance_id == 123
    assert state.label == "test-label"
    assert list(state_path.parent.glob("*.tmp")) == []


def test_creating_state_is_durable(tmp_path) -> None:
    state_path = tmp_path / "active_instance.json"
    vast.write_creating_state(state_path, "pending-label")

    pending = vast.read_lifecycle_state(state_path)
    assert pending is not None
    assert pending.phase == "creating"
    assert pending.instance_id is None


def test_watchdog_uses_restricted_container_credentials_and_absolute_deadline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(vast.time, "time", lambda: 1_000.75)
    config = valid_config(
        ready_timeout_seconds=10,
        ssh_timeout_seconds=20,
        job_timeout_seconds=30,
        progress_timeout_seconds=15,
    )

    script = vast.watchdog_onstart_script(config)
    expected_deadline = (
        1_000
        + 10
        + 20
        + config.transfer_timeout_seconds
        + 30
        + vast.WATCHDOG_TEARDOWN_GRACE_SECONDS
    )

    assert f"deadline = {expected_deadline}" in script
    assert "CONTAINER_ID" in script
    assert "CONTAINER_API_KEY" in script
    assert "Authorization" in script
    assert "VAST_API_KEY" not in script
    assert "install -d -m 0700 -o root -g root /root/.ssh" in script
    assert "chown root:root /root/.ssh/authorized_keys" in script
    assert "chmod 0600 /root/.ssh/authorized_keys" in script
    assert "nohup" in script
    assert script.rstrip().endswith("&")


def test_termination_handler_covers_hangup_and_quit(monkeypatch) -> None:
    installed = {}
    restored = {}

    monkeypatch.setattr(vast.signal, "getsignal", lambda candidate: candidate.value)

    def capture(candidate, handler):
        if callable(handler):
            installed[candidate] = handler
        else:
            restored[candidate] = handler

    monkeypatch.setattr(vast.signal, "signal", capture)

    with vast.termination_handler():
        assert set(installed) == {
            vast.signal.SIGTERM,
            vast.signal.SIGHUP,
            vast.signal.SIGQUIT,
        }

    assert restored == {candidate: candidate.value for candidate in installed}


def test_remote_runner_does_not_put_secrets_in_command(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "input" in kwargs:
            return subprocess.CompletedProcess(
                command, 0, stdout="started\n", stderr=""
            )
        response = json.dumps(
            {
                "state": "complete",
                "offset": 0,
                "output": "",
                "returncode": 0,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout=response, stderr="")

    monkeypatch.setattr(vast_remote.subprocess, "run", fake_run)
    config = valid_config()

    run_remote_training(
        ("root", "example.com", 22001),
        selection(),
        config,
        "test-run",
        wandb_key="wandb-secret",
        github_token="github-secret",
    )

    assert all("wandb-secret" not in " ".join(call[0]) for call in calls)
    assert all("github-secret" not in " ".join(call[0]) for call in calls)
    environment = json.loads(calls[0][1]["input"])["environment"]
    assert environment == {
        "WANDB_API_KEY": "wandb-secret",
        "GITHUB_TOKEN": "github-secret",
    }
    assert "start_new_session=True" in calls[0][0][-1]


def test_remote_start_command_detaches_and_records_status(tmp_path) -> None:
    job_dir = tmp_path / "job"
    command = vast_remote.remote_start_command(
        str(job_dir), selection(), valid_config(), "test-run"
    )
    assert shlex.split(command)[-10] == "test-run"
    payload = json.dumps(
        {
            "environment": {"DETACHED_TEST_VALUE": "available"},
            "script": 'printf "%s\\n" "$DETACHED_TEST_VALUE"; exit 7\n',
        }
    )

    result = subprocess.run(
        shlex.split(command),
        input=payload,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "Started detached remote job" in result.stdout
    for _ in range(100):
        if (job_dir / "status").exists():
            break
        time.sleep(0.01)
    assert (job_dir / "status").read_text() == "7\n"
    assert (job_dir / "output.log").read_text() == "available\n"


def test_remote_command_uses_canonical_trainer_for_both_profiles(tmp_path) -> None:
    command = vast_remote.remote_start_command(
        str(tmp_path / "job"),
        motion_selection(),
        motion_config(),
        "motion-test",
    )

    assert shlex.split(command)[-10] == "motion-test"
    assert vast_remote.REMOTE_TRAIN_SCRIPT.count("uv run bello-train") == 3
    assert "git clone --recurse-submodules" in vast_remote.REMOTE_TRAIN_SCRIPT
    assert "submodule status --recursive" in vast_remote.REMOTE_TRAIN_SCRIPT
    overrides_index = vast_remote.REMOTE_TRAIN_SCRIPT.index(
        'training_overrides=("experiment=$experiment")'
    )
    validation_index = vast_remote.REMOTE_TRAIN_SCRIPT.index("total_timesteps=768")
    user_index = vast_remote.REMOTE_TRAIN_SCRIPT.index('"${training_overrides[@]}"')
    forced_index = vast_remote.REMOTE_TRAIN_SCRIPT.rindex("wandb_enabled=true")
    checkpoint_index = vast_remote.REMOTE_TRAIN_SCRIPT.rindex(
        'checkpoint_name="$run_name"'
    )
    assert overrides_index < validation_index
    assert user_index < forced_index < checkpoint_index


def test_twist_remote_command_uses_staged_inputs_and_allocator_flags() -> None:
    command = vast_remote.remote_start_command(
        "/workspace/.bello-vast-jobs/twist",
        selection(),
        valid_config(experiment="twist_student", allow_git_commit_mismatch=True),
        "twist-test",
        dataset_path="/workspace/datasets/twist/index.json",
        teacher_checkpoint_path="/workspace/checkpoints/twist_teacher",
    )
    arguments = shlex.split(command)

    assert arguments[-9:-6] == [
        "/workspace/datasets/twist/index.json",
        "/workspace/checkpoints/twist_teacher",
        "true",
    ]
    assert "TF_GPU_ALLOCATOR=cuda_malloc_async" in vast_remote.REMOTE_TRAIN_SCRIPT
    assert (
        'if [[ "$experiment" != twist_* && "$experiment" != twist2 '
        '&& "$experiment" != sonic ]]' in vast_remote.REMOTE_TRAIN_SCRIPT
    )
    assert 'training_overrides+=("dataset_path=$dataset_path")' in (
        vast_remote.REMOTE_TRAIN_SCRIPT
    )


def test_sonic_remote_command_runs_bounded_baseline_and_evaluation() -> None:
    config = valid_config(
        experiment="sonic",
        dataset_path=Path("/data/sonic/index.json"),
        sonic_artifact_path=Path("/data/artifact"),
        total_timesteps=1_425_408,
        num_envs=4096,
        checkpoint_every_updates=29,
    )
    command = vast_remote.remote_start_command(
        "/workspace/.bello-vast-jobs/sonic",
        selection(),
        config,
        "sonic-test",
        dataset_path="/workspace/datasets/sonic/index.json",
        sonic_artifact_path="/workspace/artifacts/sonic/default_release",
    )
    arguments = shlex.split(command)

    assert arguments[-9:] == [
        "/workspace/datasets/sonic/index.json",
        "",
        "false",
        "/workspace/artifacts/sonic/default_release",
        "",
        "1425408",
        "4096",
        "29",
        "/workspace/.bello-vast-jobs/sonic/results",
    ]
    assert 'actual_overrides+=("total_timesteps=$total_timesteps")' in (
        vast_remote.REMOTE_TRAIN_SCRIPT
    )
    assert "uv run bello-eval" in vast_remote.REMOTE_TRAIN_SCRIPT
    assert "task=sonic" in vast_remote.REMOTE_TRAIN_SCRIPT
    assert (
        '"$experiment" != twist2 && "$experiment" != sonic'
        in vast_remote.REMOTE_TRAIN_SCRIPT
    )
    assert 'output_path="$result_dir/sonic_evaluation.json"' in (
        vast_remote.REMOTE_TRAIN_SCRIPT
    )


def test_sonic_remote_resume_is_applied_only_to_the_real_run() -> None:
    config = valid_config(
        experiment="sonic",
        dataset_path=Path("/data/sonic/index.json"),
        sonic_artifact_path=Path("/data/artifact"),
        resume_path=Path("/data/checkpoint"),
        allow_git_commit_mismatch=True,
        total_timesteps=71_270_400,
        num_envs=4096,
        checkpoint_every_updates=580,
    )
    command = vast_remote.remote_start_command(
        "/workspace/.bello-vast-jobs/sonic-long",
        selection(),
        config,
        "sonic-long",
        dataset_path="/workspace/datasets/sonic/index.json",
        sonic_artifact_path="/workspace/artifacts/sonic/default_release",
        resume_checkpoint_path="/workspace/checkpoints/sonic_resume",
    )
    arguments = shlex.split(command)

    assert arguments[-9:] == [
        "/workspace/datasets/sonic/index.json",
        "",
        "true",
        "/workspace/artifacts/sonic/default_release",
        "/workspace/checkpoints/sonic_resume",
        "71270400",
        "4096",
        "580",
        "/workspace/.bello-vast-jobs/sonic-long/results",
    ]
    script = vast_remote.REMOTE_TRAIN_SCRIPT
    resume_index = script.index(
        'actual_overrides+=("resume_path=$resume_checkpoint_path")'
    )
    preflight_index = script.index("preflight_overrides=(")
    assert resume_index < preflight_index
    assert 'preflight_overrides=("${training_overrides[@]}"' in script
    assert "checkpoint_enabled=false" in script
    assert "checkpoint_config.enabled=false" not in script
    gpu_query = "--query-gpu=utilization.gpu,memory.used,memory.total"
    power_query = "--query-gpu=power.draw"
    assert script.count(gpu_query) == 2
    assert script.count(power_query) == 2
    assert '>> "$benchmark_gpu_log" 2>/dev/null || true' in script
    assert '>> "$benchmark_power_log" 2>/dev/null || true' in script
    assert "remote job heartbeat" in command


def test_sonic_benchmark_accepts_unavailable_power_telemetry(tmp_path) -> None:
    telemetry_path = tmp_path / "gpu.csv"
    power_path = tmp_path / "power.csv"
    telemetry_path.write_text("95,25899,32607\n", encoding="utf-8")
    power_path.write_text("N/A\n", encoding="utf-8")
    script = vast_remote.REMOTE_TRAIN_SCRIPT
    program_start = script.index("import csv\n", script.index("benchmark_gpu_log"))
    program_end = script.index("\nPY\n  rm -f", program_start)

    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(telemetry_path),
            str(power_path),
            "4096",
            "3",
            "240",
            "0",
            str(tmp_path),
        ],
        input=script[program_start:program_end],
        text=True,
        check=True,
        capture_output=True,
    )

    report = json.loads(result.stdout)["sonic_benchmark"]
    assert report["gpu_utilization_peak_percent"] == 95
    assert report["gpu_memory_peak_mib"] == 25899
    assert report["gpu_power_mean_watts"] is None


def test_rsync_transfer_is_resumable_and_retried(monkeypatch) -> None:
    attempts = []
    sleeps = []

    class Executor:
        def run(self, command, *, timeout_seconds):
            attempts.append((command, timeout_seconds))
            if len(attempts) == 1:
                raise subprocess.CalledProcessError(12, command)

    moments = iter((0.0, 1.0, 1.0, 2.0))
    monkeypatch.setattr(vast_remote.time, "monotonic", lambda: next(moments))
    command = vast_remote._rsync_command(
        ("root", "example.com", 22001),
        ["/data/index.json", "/data/train.npz"],
        "/workspace/datasets/twist",
    )

    vast_remote._run_rsync_with_retries(
        command,
        timeout_seconds=30,
        retries=3,
        executor=Executor(),
        sleep=sleeps.append,
    )

    assert len(attempts) == 2
    assert sleeps == [5]
    assert "--partial" in command
    assert "--append-verify" in command
    assert command[-1] == "root@example.com:/workspace/datasets/twist/"


def test_twist_staging_verifies_dataset_before_job_start(tmp_path: Path) -> None:
    dataset_path = twist_dataset(tmp_path)
    fingerprint, paths = vast._twist_dataset_files(dataset_path)
    selected = GitSelection(
        branch="main",
        commit="a" * 40,
        config_path=PurePosixPath("configs/profiles/train/twist_teacher.yaml"),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
        dataset_fingerprint=fingerprint,
        dataset_files=paths,
    )
    remote_commands = []
    rsync_commands = []

    class Executor:
        def run(self, target, remote_command, *, timeout_seconds, input_text=None):
            remote_commands.append(remote_command)
            return subprocess.CompletedProcess(
                remote_command, 0, stdout="verified\n", stderr=""
            )

    class Transfer:
        def run(self, command, *, timeout_seconds):
            rsync_commands.append(command)

    dataset_index, teacher = vast_remote.stage_twist_inputs(
        ("root", "example.com", 22001),
        selected,
        valid_config(experiment="twist_teacher", dataset_path=dataset_path),
        executor=Executor(),
        rsync_executor=Transfer(),
        sleep=lambda _: None,
    )

    assert dataset_index == "/workspace/datasets/twist/index.json"
    assert teacher == ""
    assert len(rsync_commands) == 1
    assert {Path(path).name for path in rsync_commands[0] if path.startswith("/")} >= {
        "index.json",
        "train.npz",
        "evaluation.npz",
    }
    assert fingerprint in remote_commands[-1]


def test_sonic_staging_preserves_split_files_and_verifies_release(
    tmp_path: Path, monkeypatch
) -> None:
    index_path, index = sonic_dataset(tmp_path, monkeypatch)
    fingerprint, paths, dataset_root = vast._sonic_dataset_files(index_path)
    artifact = sonic_artifact(tmp_path, monkeypatch)
    selected = GitSelection(
        branch="main",
        commit="a" * 40,
        config_path=PurePosixPath("configs/profiles/train/sonic.yaml"),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
        dataset_fingerprint=fingerprint,
        dataset_files=paths,
        dataset_root=dataset_root,
        sonic_artifact_path=artifact,
    )
    remote_commands = []
    rsync_commands = []

    class Executor:
        def run(self, target, remote_command, *, timeout_seconds, input_text=None):
            remote_commands.append(remote_command)
            return subprocess.CompletedProcess(
                remote_command, 0, stdout="verified\n", stderr=""
            )

    class Transfer:
        def run(self, command, *, timeout_seconds):
            rsync_commands.append(command)

    resume = tmp_path / "resume"
    (resume / "state").mkdir(parents=True)
    (resume / "continuation").mkdir()
    (resume / "metadata.json").write_text("{}", encoding="utf-8")
    (resume / "state" / "data").write_bytes(b"state")
    (resume / "continuation" / "data").write_bytes(b"continuation")
    selected = replace(selected, resume_checkpoint_path=resume)

    dataset, teacher, release, staged_resume = vast_remote.stage_training_inputs(
        ("root", "example.com", 22001),
        selected,
        valid_config(
            experiment="sonic",
            dataset_path=index_path,
            sonic_artifact_path=artifact,
            resume_path=resume,
            total_timesteps=71_270_400,
            num_envs=4096,
        ),
        executor=Executor(),
        rsync_executor=Transfer(),
        sleep=lambda _: None,
    )

    assert dataset == "/workspace/datasets/sonic/index.json"
    assert teacher == ""
    assert release == "/workspace/artifacts/sonic/default_release"
    assert staged_resume == "/workspace/checkpoints/sonic_resume"
    assert len(rsync_commands) == 3
    assert "--relative" in rsync_commands[0]
    assert any("/./train.npz" in value for value in rsync_commands[0])
    assert fingerprint in remote_commands[1]
    assert any("weights.npz" in command for command in remote_commands)
    assert "metadata.json" in remote_commands[-1]


def test_downloaded_sonic_results_are_hash_and_dataset_bound(tmp_path: Path) -> None:
    fingerprint = "sha256:" + "d" * 64
    report = {
        "task": "sonic",
        "dataset_fingerprint": fingerprint,
        "global": {"completion_rate": 0.5},
        "acceptance": {"passed": False},
        "per_clip": [{"completed": True}, {"completed": False}],
    }
    summary = {
        "task": "sonic",
        "dataset_fingerprint": fingerprint,
        "evaluated_clips": 2,
        "global": report["global"],
        "acceptance": report["acceptance"],
    }
    benchmark = {
        "sonic_benchmark": {
            "num_envs": 4096,
            "updates": 3,
            "exit_status": 0,
        }
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark))
    (tmp_path / "checkpoint_path.txt").write_text("checkpoints/run/final\n")
    (tmp_path / "sonic_evaluation.json").write_text(json.dumps(report))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    files = {
        name: vast_remote._sha256(tmp_path / name)
        for name in (
            "benchmark.json",
            "checkpoint_path.txt",
            "sonic_evaluation.json",
            "summary.json",
        )
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files})
    )

    vast_remote._validate_local_sonic_results(
        tmp_path,
        expected_dataset_fingerprint=fingerprint,
    )

    summary["evaluated_clips"] = 3
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    files["summary.json"] = vast_remote._sha256(tmp_path / "summary.json")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files})
    )
    with pytest.raises(ValueError, match="summary differs"):
        vast_remote._validate_local_sonic_results(
            tmp_path,
            expected_dataset_fingerprint=fingerprint,
        )


def test_remote_checksum_verifier_detects_corruption(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"expected")
    expected = vast_remote._sha256(payload)

    class LocalExecutor:
        def run(self, target, remote_command, *, timeout_seconds, input_text=None):
            return subprocess.run(
                remote_command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    vast_remote._verify_remote_files(
        ("root", "localhost", 22),
        str(tmp_path),
        {"payload.bin": expected},
        executor=LocalExecutor(),
        timeout_seconds=30,
    )
    payload.write_bytes(b"corrupted")
    with pytest.raises(subprocess.CalledProcessError):
        vast_remote._verify_remote_files(
            ("root", "localhost", 22),
            str(tmp_path),
            {"payload.bin": expected},
            executor=LocalExecutor(),
            timeout_seconds=30,
        )


def test_remote_runner_reconnects_and_resumes_log_offset(monkeypatch, capfd) -> None:
    monitor_calls = 0
    monitor_commands = []
    sleeps = []

    def response(state, offset, output=b"", returncode=None):
        return json.dumps(
            {
                "state": state,
                "offset": offset,
                "output": base64.b64encode(output).decode("ascii"),
                "returncode": returncode,
            }
        )

    def fake_run(command, **kwargs):
        nonlocal monitor_calls
        if "input" in kwargs:
            return subprocess.CompletedProcess(
                command, 0, stdout="started\n", stderr=""
            )
        monitor_calls += 1
        monitor_commands.append(command[-1])
        if monitor_calls == 1:
            return subprocess.CompletedProcess(command, 255, stdout="", stderr="closed")
        if monitor_calls == 2:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=response("running", 6, b"setup\n"),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=response("complete", 12, b"done\n", 0),
            stderr="",
        )

    monkeypatch.setattr(vast_remote.subprocess, "run", fake_run)

    run_remote_training(
        ("root", "example.com", 22001),
        selection(),
        valid_config(),
        "test-run",
        wandb_key="wandb-secret",
        github_token="",
        sleep=sleeps.append,
    )

    output = capfd.readouterr().out
    assert "remote training continues while reconnecting" in output
    assert "SSH reconnected" in output
    assert "setup\ndone\n" in output
    assert sleeps == [5, 10]
    assert shlex.split(monitor_commands[-1])[-1] == "6"


def test_remote_runner_aborts_when_progress_deadline_expires(monkeypatch) -> None:
    class Executor:
        def run(self, target, remote_command, *, input_text=None, timeout_seconds):
            assert timeout_seconds > 0
            assert input_text is not None
            return subprocess.CompletedProcess(remote_command, 0, stdout="", stderr="")

    moments = iter((0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(vast_remote.time, "monotonic", lambda: next(moments))

    with pytest.raises(TimeoutError, match="no new output"):
        run_remote_training(
            ("root", "example.com", 22001),
            selection(),
            valid_config(job_timeout_seconds=100, progress_timeout_seconds=1),
            "test-run",
            wandb_key="wandb-secret",
            github_token="",
            executor=Executor(),
        )


def test_remote_runner_enforces_hard_deadline_while_starting(monkeypatch) -> None:
    class Executor:
        def run(self, target, remote_command, *, input_text=None, timeout_seconds):
            assert input_text is not None
            assert timeout_seconds == 1
            raise subprocess.TimeoutExpired(remote_command, timeout_seconds)

    moments = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(vast_remote.time, "monotonic", lambda: next(moments))

    with pytest.raises(TimeoutError, match="exceeded 1 seconds while starting"):
        run_remote_training(
            ("root", "example.com", 22001),
            selection(),
            valid_config(job_timeout_seconds=1, progress_timeout_seconds=1),
            "test-run",
            wandb_key="wandb-secret",
            github_token="",
            sleep=lambda _: None,
            executor=Executor(),
        )


@pytest.mark.parametrize(
    "remote_failure",
    [
        None,
        RuntimeError("training failed"),
        vast.PipelineInterrupted("received SIGTERM"),
    ],
)
def test_pipeline_always_destroys_created_instance(
    monkeypatch, tmp_path, remote_failure
) -> None:
    api = LifecycleApi()
    monkeypatch.setattr(vast, "resolve_git_selection", lambda *args: selection())
    monkeypatch.setattr(
        vast,
        "credential",
        lambda env_name, *args, **kwargs: {
            "VAST_API_KEY": "vast-secret",
            "WANDB_API_KEY": "wandb-secret",
            "GITHUB_TOKEN": "",
        }[env_name],
    )
    monkeypatch.setattr(
        vast,
        "find_offers",
        lambda *args: [VastOffer.from_mapping(make_offer())],
    )
    monkeypatch.setattr(vast, "wait_for_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        vast,
        "wait_for_ssh",
        lambda *args, **kwargs: ("root", "example.com", 22),
    )

    def remote_runner(*args, **kwargs):
        if remote_failure is not None:
            raise remote_failure

    if remote_failure is None:
        assert (
            launch_pipeline(
                tmp_path,
                valid_config(),
                api_factory=lambda key: api,
                remote_runner=remote_runner,
            )
            == 0
        )
    else:
        with pytest.raises(type(remote_failure), match=str(remote_failure)):
            launch_pipeline(
                tmp_path,
                valid_config(),
                api_factory=lambda key: api,
                remote_runner=remote_runner,
            )

    assert api.destroyed == [456]
    assert api.created[0][1]["label"].startswith("bello-")
    assert api.created[0][1]["runtype"] == "ssh_direct"
    assert "CONTAINER_ID" in api.created[0][1]["onstart_cmd"]
    assert "CONTAINER_API_KEY" in api.created[0][1]["onstart_cmd"]
    assert "ssh" not in api.created[0][1]
    assert "direct" not in api.created[0][1]
    assert not (tmp_path / ".vast" / "active_instance.json").exists()


@pytest.mark.parametrize(
    "first_probe",
    [0.3, RuntimeError("route connection reset")],
)
def test_twist_lifecycle_rejects_unusable_route_and_tries_next_offer(
    monkeypatch, tmp_path, first_probe
) -> None:
    class Api(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            self.created.append((offer_id, kwargs))
            return offer_id

    api = Api()
    config = valid_config(
        experiment="twist_teacher",
        dataset_path=tmp_path / "index.json",
    )
    twist_selection = GitSelection(
        branch="main",
        commit="a" * 40,
        config_path=PurePosixPath("configs/profiles/train/twist_teacher.yaml"),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
        dataset_files=(tmp_path / "index.json",),
    )
    monkeypatch.setattr(
        vast,
        "find_offers",
        lambda *args: [
            VastOffer.from_mapping(make_offer(1, location="Seoul, KR")),
            VastOffer.from_mapping(make_offer(2, location="Tokyo, JP")),
        ],
    )
    monkeypatch.setattr(vast, "wait_for_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        vast,
        "wait_for_ssh",
        lambda api, instance_id, timeout: ("root", f"host-{instance_id}", 22),
    )
    probe_results = iter((first_probe, 20.0))

    def measure_upload(*args, **kwargs):
        result = next(probe_results)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(vast, "measure_upload_mib_per_second", measure_upload)
    remote_targets = []

    def remote_runner(target, *args, **kwargs):
        remote_targets.append(target)

    VastLifecycle(
        api,
        config,
        tmp_path / ".vast" / "active_instance.json",
        "route-test",
    ).run(
        twist_selection,
        "route-test",
        remote_runner,
        wandb_key="secret",
        github_token="",
    )

    assert [offer_id for offer_id, _ in api.created] == [1, 2]
    assert api.destroyed == [1, 2]
    assert remote_targets == [("root", "host-2", 22)]


def test_lifecycle_retries_next_offer_when_direct_ssh_times_out(
    monkeypatch, tmp_path
) -> None:
    class Api(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            self.created.append((offer_id, kwargs))
            return offer_id

    api = Api()
    monkeypatch.setattr(
        vast,
        "find_offers",
        lambda *args: [
            VastOffer.from_mapping(make_offer(1, location="Seoul, KR")),
            VastOffer.from_mapping(make_offer(2, location="Tokyo, JP")),
        ],
    )
    monkeypatch.setattr(vast, "wait_for_instance", lambda *args, **kwargs: None)
    ssh_attempts = iter((TimeoutError("direct SSH unavailable"), None))

    def wait_for_ssh(api, instance_id, timeout):
        failure = next(ssh_attempts)
        if failure is not None:
            raise failure
        return "root", f"host-{instance_id}", 22

    monkeypatch.setattr(vast, "wait_for_ssh", wait_for_ssh)
    remote_targets = []

    VastLifecycle(
        api,
        valid_config(),
        tmp_path / ".vast" / "active_instance.json",
        "ssh-test",
    ).run(
        selection(),
        "ssh-test",
        lambda target, *args, **kwargs: remote_targets.append(target),
        wandb_key="secret",
        github_token="",
    )

    assert [offer_id for offer_id, _ in api.created] == [1, 2]
    assert api.destroyed == [1, 2]
    assert remote_targets == [("root", "host-2", 22)]


def test_motion_dry_run_never_reads_credentials_or_rents(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(vast, "resolve_git_selection", lambda *args: motion_selection())
    monkeypatch.setattr(
        vast,
        "credential",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not read credentials")
        ),
    )

    def api_factory(api_key):
        raise AssertionError("dry-run must not create a Vast client")

    assert (
        launch_pipeline(
            tmp_path,
            motion_config(dry_run=True),
            api_factory=api_factory,
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "for motion-tracking" in output
    assert str(motion_selection().motion_path) in output
    assert "no instance was rented" in output
    assert not (tmp_path / ".vast").exists()


def test_lifecycle_retains_ambiguous_create_and_refuses_another_rental(
    monkeypatch, tmp_path
) -> None:
    class AmbiguousApi(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            self.created.append((offer_id, kwargs))
            raise TimeoutError("response lost")

        def show_instances(self):
            return []

    api = AmbiguousApi()
    state_path = tmp_path / ".vast" / "active_instance.json"
    config = valid_config(reconcile_retries=1)
    monkeypatch.setattr(
        vast,
        "find_offers",
        lambda *args: [VastOffer.from_mapping(make_offer())],
    )
    lifecycle = VastLifecycle(api, config, state_path, "bello-pending")

    with pytest.raises(RuntimeError, match="ambiguous"):
        lifecycle.run(
            selection(),
            "pending",
            lambda *args, **kwargs: None,
            wandb_key="wandb-secret",
            github_token="",
        )

    pending = vast.read_lifecycle_state(state_path)
    assert pending is not None
    assert pending.phase == "creating"
    assert pending.label == "bello-pending"
    assert len(api.created) == 1

    with pytest.raises(RuntimeError, match="still unresolved"):
        lifecycle.run(
            selection(),
            "pending",
            lambda *args, **kwargs: None,
            wandb_key="wandb-secret",
            github_token="",
        )

    assert len(api.created) == 1


def test_lifecycle_recovers_and_destroys_pending_labeled_instance(
    monkeypatch, tmp_path
) -> None:
    pending_id = 789

    class PendingApi(LifecycleApi):
        def show_instances(self):
            if pending_id in self.destroyed:
                return []
            return [{"id": pending_id, "label": "bello-pending"}]

    api = PendingApi()
    state_path = tmp_path / ".vast" / "active_instance.json"
    vast.write_creating_state(state_path, "bello-pending")
    monkeypatch.setattr(
        vast,
        "find_offers",
        lambda *args: (_ for _ in ()).throw(RuntimeError("stop after reconciliation")),
    )

    with pytest.raises(RuntimeError, match="stop after reconciliation"):
        VastLifecycle(
            api,
            valid_config(reconcile_retries=1),
            state_path,
            "bello-new",
        ).run(
            selection(),
            "new",
            lambda *args, **kwargs: None,
            wandb_key="wandb-secret",
            github_token="",
        )

    assert api.destroyed == [pending_id]
    assert not state_path.exists()


def test_lifecycle_reconciles_stale_labeled_instance_before_rental(
    monkeypatch, tmp_path
) -> None:
    stale_id = 111
    stale_label = "bello-stale"

    class RecoveringApi(LifecycleApi):
        def show_instances(self):
            if stale_id in self.destroyed:
                return []
            return [{"id": stale_id, "label": stale_label}]

    api = RecoveringApi()
    state_path = tmp_path / ".vast" / "active_instance.json"
    state_path.parent.mkdir(parents=True)
    vast.write_active_state(state_path, stale_id, stale_label)
    monkeypatch.setattr(
        vast,
        "find_offers",
        lambda *args: [VastOffer.from_mapping(make_offer())],
    )
    monkeypatch.setattr(vast, "wait_for_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        vast,
        "wait_for_ssh",
        lambda *args, **kwargs: ("root", "example.com", 22),
    )
    remote_calls = []

    VastLifecycle(api, valid_config(), state_path, "bello-new").run(
        selection(),
        "new",
        lambda *args, **kwargs: remote_calls.append((args, kwargs)),
        wandb_key="wandb-secret",
        github_token="",
    )

    assert api.destroyed == [stale_id, 456]
    assert api.created[0][1]["label"] == "bello-new"
    assert len(remote_calls) == 1
    assert not state_path.exists()


def test_pipeline_retains_recovery_state_when_destroy_fails(
    monkeypatch, tmp_path
) -> None:
    api = LifecycleApi()
    monkeypatch.setattr(vast, "resolve_git_selection", lambda *args: selection())
    monkeypatch.setattr(vast, "credential", lambda *args, **kwargs: "secret")
    monkeypatch.setattr(
        vast,
        "find_offers",
        lambda *args: [VastOffer.from_mapping(make_offer())],
    )
    monkeypatch.setattr(vast, "wait_for_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        vast,
        "wait_for_ssh",
        lambda *args, **kwargs: ("root", "example.com", 22),
    )
    monkeypatch.setattr(vast, "destroy_with_retries", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="could not confirm destruction"):
        launch_pipeline(
            tmp_path,
            valid_config(),
            api_factory=lambda key: api,
            remote_runner=lambda *args, **kwargs: None,
        )

    state_path = tmp_path / ".vast" / "active_instance.json"
    assert json.loads(state_path.read_text())["instance_id"] == 456
