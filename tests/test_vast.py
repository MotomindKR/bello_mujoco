import base64
import json
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath

import pytest

import bello_mujoco.training.vast as vast
import bello_mujoco.training.vast_remote as vast_remote
from bello_mujoco.training.vast import (
    GitSelection,
    VastOffer,
    VastSdkClient,
    VastTrainConfig,
    credential,
    create_available_instance,
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
    kwargs.setdefault("experiment", "bello_locomotion")
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
        config_path=PurePosixPath("configs/profiles/train/bello_locomotion.yaml"),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
    )


def motion_config(**kwargs) -> VastTrainConfig:
    kwargs.setdefault("experiment", "motion")
    kwargs.setdefault(
        "motion_path",
        Path("motion.npz"),
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
    )


@pytest.mark.parametrize("max_price", [None, 0.0, float("nan"), float("inf")])
def test_validate_config_requires_finite_positive_price(max_price) -> None:
    with pytest.raises(ValueError, match="max-price"):
        validate_config(VastTrainConfig(max_price=max_price, branch="main"))


@pytest.mark.parametrize(
    "experiment", ("unsupported", "locomotion", "hardware_locomotion")
)
def test_vast_experiment_rejects_unknown_and_retired_values(experiment: str) -> None:
    with pytest.raises(ValueError, match="--experiment"):
        validate_config(valid_config(experiment=experiment))  # type: ignore[arg-type]


def test_vast_defaults_to_bello_and_accepts_mini_locomotion_profiles() -> None:
    assert VastTrainConfig().experiment == "bello_locomotion"
    validate_config(valid_config(experiment="bello_mini_locomotion"))
    validate_config(valid_config(experiment="bello_mini_standing"))


def test_motion_git_selection_only_pins_source_not_clip(monkeypatch, tmp_path) -> None:
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

    assert resolved.commit == commit
    assert not any(value.endswith(".npz") for value in checked_objects)
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

    raw_client = PinnedClient()
    client = object.__new__(VastSdkClient)
    client._client = raw_client

    assert client.search_offers("query", order="score-", limit=1, storage=80.0) == [
        make_offer()
    ]
    assert client.create_instance(123, image="test") == 456
    assert client.show_instance(456)["actual_status"] == "running"
    assert client.show_instances() == [{"id": 456}]


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


def test_vast_error_classification_prefers_structured_http_status() -> None:
    class Response:
        status_code = 404

    error = RuntimeError("unrelated message")
    error.response = Response()

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
            sleep=lambda _: None,
        )
        == 789
    )
    assert len(api.created) == 1


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
    expected_deadline = 1_000 + 10 + 20 + 30 + vast.WATCHDOG_TEARDOWN_GRACE_SECONDS

    assert f"supervise({expected_deadline})" in script
    assert "CONTAINER_ID" in script
    assert "CONTAINER_API_KEY" in script
    assert "Authorization" in script
    assert "VAST_API_KEY" not in script
    assert "nohup" in script
    assert script.rstrip().endswith("&")


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


@pytest.mark.parametrize(
    ("script", "timeout", "exit_code", "output"),
    [
        ('printf "%s\\n" "$DETACHED_TEST_VALUE"; exit 7\n', 5, 7, "available\n"),
        ("sleep 2\n", 0.05, 124, ""),
    ],
)
def test_remote_start_command_detaches_and_records_status(
    tmp_path, script, timeout, exit_code, output
) -> None:
    import os

    (tmp_path / "armed").write_text("100")
    (tmp_path / "supervisor_pid").write_text(str(os.getpid()))
    job_dir = tmp_path / "job"
    command = vast_remote.remote_start_command(
        str(job_dir), selection(), valid_config(), "test-run"
    )
    assert shlex.split(command)[-3] == "test-run"
    payload = json.dumps(
        {
            "environment": {"DETACHED_TEST_VALUE": "available"},
            "script": script,
            "timeout_seconds": timeout,
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
        if (tmp_path / "status").exists():
            break
        time.sleep(0.01)
    assert (job_dir / "status").read_text() == f"{exit_code}\n"
    assert (tmp_path / "status").read_text() == f"{exit_code}\n"
    assert (job_dir / "output.log").read_text() == output
    retry = subprocess.run(
        shlex.split(command), input=payload, text=True, check=True, capture_output=True
    )
    assert "already completed" in retry.stdout
    assert (job_dir / "output.log").read_text() == output


def test_remote_command_uses_canonical_trainer_for_both_profiles(tmp_path) -> None:
    command = vast_remote.remote_start_command(
        str(tmp_path / "job"),
        motion_selection(),
        motion_config(end_effector_payload_mass_kg=4.5359237),
        "motion-test",
    )

    assert shlex.split(command)[-3] == "motion-test"
    assert float(shlex.split(command)[-4]) == pytest.approx(4.5359237)
    assert (
        vast_remote.REMOTE_TRAIN_SCRIPT.count("uv run --group training bello-train")
        == 2
    )
    assert "git clone --single-branch" in vast_remote.REMOTE_TRAIN_SCRIPT
    assert (
        "submodule update --init --recursive -- deps/bello-stack"
        in vast_remote.REMOTE_TRAIN_SCRIPT
    )
    assert (
        "submodule status --recursive -- deps/bello-stack"
        in vast_remote.REMOTE_TRAIN_SCRIPT
    )
    assert (
        'training_overrides+=("end_effector_payload_mass_kg='
        '$end_effector_payload_mass_kg")' in vast_remote.REMOTE_TRAIN_SCRIPT
    )
    overrides_index = vast_remote.REMOTE_TRAIN_SCRIPT.index(
        'training_overrides=("experiment=$experiment")'
    )
    validation_index = vast_remote.REMOTE_TRAIN_SCRIPT.index("total_timesteps=768")
    user_index = vast_remote.REMOTE_TRAIN_SCRIPT.index('"${training_overrides[@]}"')
    forced_index = vast_remote.REMOTE_TRAIN_SCRIPT.rindex("wandb_enabled=true")
    checkpoint_index = vast_remote.REMOTE_TRAIN_SCRIPT.index(
        'checkpoint_name="$run_name"'
    )
    assert overrides_index < validation_index
    assert user_index < forced_index < checkpoint_index


def test_remote_command_preflights_software_rendering() -> None:
    script = vast_remote.REMOTE_TRAIN_SCRIPT

    sync_index = script.index("uv sync --frozen")
    assert script.index("export UV_HTTP_TIMEOUT=600") < sync_index
    assert script.index("export UV_HTTP_RETRIES=5") < sync_index
    assert "libegl-mesa0" in script
    assert script.index("https://archive.ubuntu.com/ubuntu") < script.index(
        "apt-get update"
    )
    assert "configure_software_rendering()" in script
    assert "validate_software_renderer()" in script
    assert "mujoco.Renderer" in script
    assert "VideoWriter" in script
    assert script.index("mujoco.Renderer") < script.index("total_timesteps=768")
    assert "load_locomotion_policy_runtime(checkpoint, use_viewer_model=True)" in script
    assert "load_checkpoint_update(checkpoint) < 1" in script
    checkpoint_render = script.split(
        "python - \"checkpoints/vast-preflight-$run_name\" <<'PY'", 1
    )[1]
    assert checkpoint_render.index("configure_software_rendering()") < (
        checkpoint_render.index("import jax")
    )
    assert script.index("preflight_checkpoint_restored") < script.rindex(
        "uv run --group training bello-train"
    )


def test_motion_preflight_exports_and_restores_before_training() -> None:
    import ast

    script = vast_remote.REMOTE_TRAIN_SCRIPT
    block = (
        script.split('if [[ -n "$motion_path" ]]; then', 1)[1]
        .split("<<'PY'\n", 1)[1]
        .split("\nPY", 1)[0]
    )
    ast.parse(block)
    assert "load_motion_policy_runtime(checkpoint, use_viewer_model=True)" in block
    assert "load_checkpoint_update(paths[0]) < 1" in block
    assert "render_motion_policy(" in block
    assert "export_checkpoint(ExportConfig(" in block
    assert script.index("motion_preflight_checkpoint_restored") < script.rindex(
        "uv run --group training bello-train"
    )


@pytest.mark.parametrize("payload_mass", [-1.0, float("nan"), float("inf")])
def test_motion_vast_config_rejects_invalid_payload(
    payload_mass: float, tmp_path
) -> None:
    clip = tmp_path / "motion.npz"
    clip.write_bytes(b"clip")
    with pytest.raises(ValueError, match="end-effector-payload-mass-kg"):
        validate_config(
            motion_config(motion_path=clip, end_effector_payload_mass_kg=payload_mass)
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

    clip = tmp_path / "untracked motion.npz"
    clip.write_bytes(b"test clip")
    assert (
        launch_pipeline(
            tmp_path,
            motion_config(dry_run=True, motion_path=clip),
            api_factory=api_factory,
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "for motion-tracking" in output
    assert str(clip) in output
    assert "no instance was rented" in output
    assert not (tmp_path / ".vast").exists()
