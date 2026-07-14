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
    VastOffer,
    VastOfferConfig,
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


def test_validate_config_requires_recipe_and_price() -> None:
    with pytest.raises(ValueError, match="train-config"):
        validate_config(VastTrainConfig(max_price=1.0))
    with pytest.raises(ValueError, match="max-price"):
        validate_config(VastTrainConfig(train_config=Path("config.yaml")))


def test_validate_config_restricts_destructive_remote_path() -> None:
    with pytest.raises(ValueError, match="child of /workspace"):
        validate_config(valid_config(remote_dir="/"))


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

    assert create_available_instance(api, offers, valid_config(), "test-label") == 456
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


def test_destroy_treats_not_found_after_api_error_as_success() -> None:
    class AlreadyGoneApi:
        def destroy_instance(self, instance_id):
            raise RuntimeError("404")

        def show_instance(self, instance_id):
            raise RuntimeError("404")

    assert destroy_with_retries(AlreadyGoneApi(), 123, retries=1)


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


@pytest.mark.parametrize("remote_failure", [None, RuntimeError("training failed")])
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
        with pytest.raises(RuntimeError, match="training failed"):
            launch_pipeline(
                tmp_path,
                valid_config(),
                api_factory=lambda key: api,
                remote_runner=remote_runner,
            )

    assert api.destroyed == [456]
    assert api.created[0][1]["runtype"] == "ssh_direc ssh_proxy"
    assert "ssh" not in api.created[0][1]
    assert "direct" not in api.created[0][1]
    assert not (tmp_path / ".vast" / "active_instance.json").exists()


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
