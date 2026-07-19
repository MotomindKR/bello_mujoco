import base64
import json
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath

import pytest

import bello_mujoco.vast as vast
from bello_mujoco.vast import (
    GitSelection,
    VastLifecycle,
    VastOffer,
    VastOfferConfig,
    VastSdkClient,
    VastTrainConfig,
    credential,
    create_available_instance,
    destroy_with_retries,
    find_offers,
    launch_pipeline,
    offer_query,
    parse_ssh_url,
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
            raise RuntimeError("not found")
        return {"id": instance_id, "actual_status": "running"}

    def show_instances(self):
        return []

    def destroy_instance(self, instance_id):
        self.destroyed.append(instance_id)


def valid_config(**kwargs) -> VastTrainConfig:
    return VastTrainConfig(
        train_config=Path("experiments/train/locomotion.yaml"),
        max_price=1.25,
        branch="main",
        yes=True,
        **kwargs,
    )


def selection() -> GitSelection:
    return GitSelection(
        branch="main",
        commit="a" * 40,
        config_path=PurePosixPath("experiments/train/locomotion.yaml"),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
    )


def motion_config(**kwargs) -> VastTrainConfig:
    return VastTrainConfig(
        train_config=Path("experiments/train_motion/asap_single_foot_balance.yaml"),
        task="motion-tracking",
        max_price=1.25,
        branch="main",
        yes=True,
        **kwargs,
    )


def motion_selection() -> GitSelection:
    return GitSelection(
        branch="main",
        commit="b" * 40,
        config_path=PurePosixPath(
            "experiments/train_motion/asap_single_foot_balance.yaml"
        ),
        repo_url="https://github.com/MotomindKR/bello_mujoco.git",
        motion_path=PurePosixPath(
            "motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz"
        ),
    )


def test_validate_config_requires_recipe_and_price() -> None:
    with pytest.raises(ValueError, match="train-config"):
        validate_config(VastTrainConfig(max_price=1.0))
    with pytest.raises(ValueError, match="max-price"):
        validate_config(VastTrainConfig(train_config=Path("config.yaml")))


def test_vast_task_defaults_to_locomotion_and_rejects_unknown_values() -> None:
    assert VastTrainConfig().task == "locomotion"
    with pytest.raises(ValueError, match="--task"):
        validate_config(valid_config(task="unsupported"))  # type: ignore[arg-type]


def test_resolve_vast_motion_path_uses_recipe_and_hydra_overrides() -> None:
    recipe = """
motion:
  directory: motions/local/lafan1_fixed_head
  sequence: walk1_subject1
"""

    assert vast.resolve_vast_motion_path(
        recipe,
        (
            'motion.directory="motions/asap_fixed_head"',
            "motion.sequence=video_single_foot_balance_level4_filter_amass",
        ),
    ) == PurePosixPath(
        "motions/asap_fixed_head/video_single_foot_balance_level4_filter_amass.npz"
    )


def test_resolve_vast_motion_path_rejects_ignored_and_unsafe_inputs() -> None:
    local_recipe = """
motion:
  directory: motions/local/lafan1_fixed_head
  sequence: walk1_subject1
"""
    with pytest.raises(ValueError, match="motions/local is ignored"):
        vast.resolve_vast_motion_path(local_recipe)
    with pytest.raises(ValueError, match="without a path or suffix"):
        vast.resolve_vast_motion_path(
            "motion: {directory: motions/asap_fixed_head, sequence: ../clip}"
        )


def test_motion_git_selection_verifies_clip_at_selected_commit(
    monkeypatch, tmp_path
) -> None:
    commit = "c" * 40
    checked_objects = []
    recipe = """
motion:
  directory: motions/asap_fixed_head
  sequence: video_single_foot_balance_level1_filter_amass
"""

    monkeypatch.setattr(vast, "remote_branches", lambda repo_root: ["main"])

    def fake_run_git(repo_root, *args):
        assert repo_root == tmp_path
        if args == ("rev-parse", "refs/remotes/origin/main^{commit}"):
            return commit
        if args[:2] == ("cat-file", "-e"):
            checked_objects.append(args[2])
            return ""
        if args == (
            "show",
            f"{commit}:experiments/train_motion/asap_single_foot_balance.yaml",
        ):
            return recipe
        if args == ("remote", "get-url", "origin"):
            return "git@github.com:MotomindKR/bello_mujoco.git"
        raise AssertionError(args)

    monkeypatch.setattr(vast, "run_git", fake_run_git)

    resolved = vast.resolve_git_selection(tmp_path, motion_config())

    assert resolved.motion_path == motion_selection().motion_path
    assert f"{commit}:{resolved.motion_path}" in checked_objects


def test_validate_config_restricts_destructive_remote_path() -> None:
    with pytest.raises(ValueError, match="child of /workspace"):
        validate_config(valid_config(remote_dir="/"))


def test_validate_config_requires_bounded_job_timeouts() -> None:
    with pytest.raises(ValueError, match="job and progress"):
        validate_config(valid_config(job_timeout_seconds=0))
    with pytest.raises(ValueError, match="must not exceed"):
        validate_config(
            valid_config(job_timeout_seconds=10, progress_timeout_seconds=11)
        )


def test_offer_query_contains_hardware_price_and_region_constraints() -> None:
    query = offer_query(valid_config(), countries=("KR",))

    assert "gpu_name=RTX_5090" in query
    assert "cpu_ram>=64" in query
    assert "disk_space>=80" in query
    assert "cuda_max_good>=13.0" in query
    assert "dph_total<=1.25" in query
    assert "geolocation in [KR]" in query


def test_find_offers_prefers_first_nonempty_region_and_best_value() -> None:
    api = SearchApi(
        [
            [],
            [
                make_offer(1, value=5.0, location="Tokyo, JP"),
                make_offer(2, value=12.0, location="Singapore, SG"),
            ],
        ]
    )

    tier, offers = find_offers(api, valid_config())

    assert tier == "Asia"
    assert [offer.id for offer in offers] == [2, 1]
    assert len(api.queries) == 2
    assert api.queries[0][3] == 80.0


def test_find_offers_rechecks_returned_price_cap() -> None:
    api = SearchApi([[make_offer(price=1.26)], [], []])

    with pytest.raises(RuntimeError, match="no Vast offers"):
        find_offers(api, valid_config())


def test_create_available_instance_falls_back_when_offer_is_gone() -> None:
    class RacingApi(LifecycleApi):
        def create_instance(self, offer_id, **kwargs):
            if offer_id == 1:
                raise RuntimeError("offer disappeared")
            return super().create_instance(offer_id, **kwargs)

    api = RacingApi()
    offers = [
        VastOffer.from_mapping(make_offer(1)),
        VastOffer.from_mapping(make_offer(2)),
    ]

    assert (
        create_available_instance(
            api, offers, valid_config(), "test-label", sleep=lambda _: None
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

    _, offers = find_offers(api, valid_config())

    assert [offer.id for offer in offers] == [123]


def test_sdk_client_normalizes_raw_vast_response_shapes() -> None:
    class RawClient:
        def __init__(self):
            self.destroyed = []

        def search_offers(self, **kwargs):
            assert kwargs["storage"] == 80.0
            return json.dumps({"offers": [make_offer()]})

        def create_instance(self, **kwargs):
            assert kwargs["id"] == 123
            return json.dumps({"new_contract": 456})

        def show_instance(self, **kwargs):
            return json.dumps(
                {"instances": [{"id": kwargs["id"], "actual_status": "running"}]}
            )

        def show_instances(self):
            return json.dumps({"instances": [{"id": 456}]})

        def ssh_url(self, **kwargs):
            assert kwargs["id"] == 456
            return json.dumps({"ssh_url": "ssh://root@example.com:22001"})

        def destroy_instance(self, **kwargs):
            self.destroyed.append(kwargs["id"])

    raw_client = RawClient()
    client = object.__new__(VastSdkClient)
    client._client = raw_client

    assert client.search_offers("query", order="score-", limit=1, storage=80.0) == [
        make_offer()
    ]
    assert client.create_instance(123, image="test") == 456
    assert client.show_instance(456)["actual_status"] == "running"
    assert client.show_instances() == [{"id": 456}]
    assert client.ssh_url(456) == "ssh://root@example.com:22001"
    client.destroy_instance(456)
    assert raw_client.destroyed == [456]


def test_sdk_client_rejects_error_mapping_as_instance_listing() -> None:
    class RawClient:
        def __init__(self, response):
            self.response = response

        def show_instances(self):
            return json.dumps(self.response)

    for response in (
        {"error": "rate limited"},
        {"instances": [{"error": "rate limited"}]},
    ):
        client = object.__new__(VastSdkClient)
        client._client = RawClient(response)
        with pytest.raises(RuntimeError, match="unexpected Vast instances response"):
            client.show_instances()


def test_sdk_client_rejects_error_mapping_as_offer_listing() -> None:
    class RawClient:
        def __init__(self, response):
            self.response = response

        def search_offers(self, **kwargs):
            return json.dumps(self.response)

    for response in (
        {"error": "rate limited"},
        {"offers": [{"error": "rate limited"}]},
    ):
        client = object.__new__(VastSdkClient)
        client._client = RawClient(response)
        with pytest.raises(RuntimeError, match="unexpected Vast offer response"):
            client.search_offers("query", order="score-", limit=1, storage=80.0)


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


def test_parse_ssh_url_requires_complete_ssh_url() -> None:
    assert parse_ssh_url("ssh://root@example.com:22001") == (
        "root",
        "example.com",
        22001,
    )
    with pytest.raises(ValueError, match="invalid Vast SSH URL"):
        parse_ssh_url("example.com")


def test_wait_for_ssh_prefers_direct_port_mapping(monkeypatch) -> None:
    class Api:
        def show_instance(self, instance_id):
            assert instance_id == 123
            return {
                "public_ipaddr": "192.0.2.10",
                "ports": {"22/tcp": [{"HostPort": "22022"}]},
            }

        def ssh_url(self, instance_id):
            assert instance_id == 123
            return "ssh://root@proxy.example.com:22001"

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert 0 < kwargs["timeout"] <= vast.SSH_SUBPROCESS_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vast.subprocess, "run", fake_run)

    target = vast.wait_for_ssh(Api(), 123, 30, sleep=lambda _: None)

    assert target == ("root", "192.0.2.10", 22022)
    assert commands == [vast.ssh_command(target, "true")]


def test_destroy_treats_not_found_after_api_error_as_success() -> None:
    class AlreadyGoneApi:
        def destroy_instance(self, instance_id):
            raise RuntimeError("404")

        def show_instance(self, instance_id):
            raise RuntimeError("404")

    assert destroy_with_retries(AlreadyGoneApi(), 123, retries=1)


def test_vast_error_classification_prefers_structured_http_status() -> None:
    class Response:
        status_code = 404

    error = RuntimeError("unrelated message")
    error.response = Response()

    assert vast._is_explicit_not_found(error)
    assert vast._is_explicit_offer_unavailable(error)


def test_destroy_does_not_treat_transient_api_errors_as_success() -> None:
    class UnreachableApi:
        def destroy_instance(self, instance_id):
            raise TimeoutError("gateway timed out")

        def show_instance(self, instance_id):
            raise TimeoutError("gateway timed out")

        def show_instances(self):
            raise TimeoutError("gateway timed out")

    assert not destroy_with_retries(UnreachableApi(), 123, retries=1)


def test_destroy_requires_status_or_successful_absence_confirmation() -> None:
    class MissingStatusApi:
        def __init__(self, listed):
            self.listed = listed

        def destroy_instance(self, instance_id):
            pass

        def show_instance(self, instance_id):
            return {"id": instance_id}

        def show_instances(self):
            return [{"id": 123}] if self.listed else []

    assert not destroy_with_retries(MissingStatusApi(True), 123, retries=1)
    assert destroy_with_retries(MissingStatusApi(False), 123, retries=1)


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
            api, offers, valid_config(), "test-label", sleep=lambda _: None
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
            api, offers, valid_config(), "test-label", sleep=lambda _: None
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


def test_active_state_write_is_atomic_and_round_trips(tmp_path) -> None:
    state_path = tmp_path / ".vast" / "active_instance.json"

    vast.write_active_state(state_path, 123, "test-label")

    state = vast.read_lifecycle_state(state_path)
    assert state is not None
    assert state.phase == "active"
    assert state.instance_id == 123
    assert state.label == "test-label"
    assert list(state_path.parent.glob("*.tmp")) == []


def test_creating_state_is_durable_and_legacy_active_state_remains_readable(
    tmp_path,
) -> None:
    state_path = tmp_path / "active_instance.json"
    vast.write_creating_state(state_path, "pending-label")

    pending = vast.read_lifecycle_state(state_path)
    assert pending is not None
    assert pending.phase == "creating"
    assert pending.instance_id is None

    state_path.write_text(
        json.dumps({"instance_id": 321, "label": "legacy-label"}),
        encoding="utf-8",
    )
    legacy = vast.read_lifecycle_state(state_path)
    assert legacy is not None
    assert legacy.phase == "active"
    assert legacy.instance_id == 321
    assert legacy.label == "legacy-label"


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

    assert f"deadline = {expected_deadline}" in script
    assert "CONTAINER_ID" in script
    assert "CONTAINER_API_KEY" in script
    assert "Authorization" in script
    assert "VAST_API_KEY" not in script
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

    monkeypatch.setattr(vast.subprocess, "run", fake_run)
    config = valid_config(hydra_overrides=("config.seed=2",))

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
    assert environment["WANDB_API_KEY"] == "wandb-secret"
    assert environment["GITHUB_TOKEN"] == "github-secret"
    assert environment["BELLO_HYDRA_OVERRIDES"] == '["config.seed=2"]'
    assert "start_new_session=True" in calls[0][0][-1]


def test_remote_start_command_detaches_and_records_status(tmp_path) -> None:
    job_dir = tmp_path / "job"
    command = vast._remote_start_command(
        str(job_dir), selection(), valid_config(), "test-run"
    )
    assert shlex.split(command)[-1] == "locomotion"
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


def test_motion_remote_command_uses_motion_trainer_for_validation_and_training(
    tmp_path,
) -> None:
    command = vast._remote_start_command(
        str(tmp_path / "job"),
        motion_selection(),
        motion_config(),
        "motion-test",
    )

    assert shlex.split(command)[-1] == "motion-tracking"
    assert (
        'motion-tracking) train_command="bello-train-motion-ppo"'
        in vast.REMOTE_TRAIN_SCRIPT
    )
    assert vast.REMOTE_TRAIN_SCRIPT.count('uv run "$train_command"') == 2
    overrides_index = vast.REMOTE_TRAIN_SCRIPT.index("mapfile -t user_overrides")
    validation_index = vast.REMOTE_TRAIN_SCRIPT.index("dry_run=true")
    assert overrides_index < validation_index


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

    monkeypatch.setattr(vast.subprocess, "run", fake_run)

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
    monkeypatch.setattr(vast.time, "monotonic", lambda: next(moments))

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
    monkeypatch.setattr(vast.time, "monotonic", lambda: next(moments))

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
        lambda *args: (
            "Korea",
            [VastOffer.from_mapping(make_offer())],
        ),
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
    assert api.created[0][1]["runtype"] == "ssh_direc ssh_proxy"
    assert "CONTAINER_ID" in api.created[0][1]["onstart_cmd"]
    assert "CONTAINER_API_KEY" in api.created[0][1]["onstart_cmd"]
    assert "ssh" not in api.created[0][1]
    assert "direct" not in api.created[0][1]
    assert not (tmp_path / ".vast" / "active_instance.json").exists()


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
        lambda *args: ("Korea", [VastOffer.from_mapping(make_offer())]),
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
        lambda *args: ("Korea", [VastOffer.from_mapping(make_offer())]),
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
        lambda *args: ("Korea", [VastOffer.from_mapping(make_offer())]),
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


def test_mandatory_training_overrides_follow_user_overrides() -> None:
    user_index = vast.REMOTE_TRAIN_SCRIPT.index('"${user_overrides[@]}"')
    forced_index = vast.REMOTE_TRAIN_SCRIPT.index("wandb_config.enabled=true")
    checkpoint_index = vast.REMOTE_TRAIN_SCRIPT.index(
        "checkpoint_config.upload_wandb_artifact=true"
    )

    assert user_index < forced_index < checkpoint_index


def test_offer_config_can_override_asia_tier() -> None:
    config = valid_config(
        offer=VastOfferConfig(asia_countries=("JP", "TH")),
    )

    assert "geolocation in [JP,TH]" in offer_query(
        config, countries=config.offer.asia_countries
    )
