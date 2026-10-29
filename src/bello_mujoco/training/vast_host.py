"""Standalone host supervisor, installed by Vast's onstart hook before SSH setup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


STATE_ROOT = Path("/workspace/.bello-vast")


def shutdown_reason(root: Path, deadline: float, now: float) -> str | None:
    if now >= deadline:
        return "hard timeout"
    status = root / "status"
    if status.is_file():
        return f"job exited with status {int(status.read_text().strip())}"
    pid_file = root / "job_pid"
    if pid_file.is_file():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "job process disappeared"
        stat = Path(f"/proc/{pid}/stat")
        try:
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return "job process is a zombie"
        except FileNotFoundError:
            return "job process disappeared"
    return None


def destroy_once(instance_id: str, api_key: str) -> bool:
    request = urllib.request.Request(
        f"https://cloud.vast.ai/api/v0/instances/{instance_id}/",
        method="DELETE",
        data=b"{}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "bello-mujoco-supervisor",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            confirmed = json.load(response).get("success") is True
            if not confirmed:
                print("Vast teardown response did not confirm success", flush=True)
            return confirmed
    except urllib.error.HTTPError as error:
        if error.code != 404:
            print(f"Vast teardown HTTP status {error.code}", flush=True)
        return error.code == 404
    except (OSError, urllib.error.URLError, ValueError) as error:
        print(f"Vast teardown request failed: {type(error).__name__}", flush=True)
        return False


def supervise(deadline: int, root: Path = STATE_ROOT) -> None:
    instance_id = os.environ.get("CONTAINER_ID", "")
    api_key = os.environ.get("CONTAINER_API_KEY", "")
    if not instance_id.isdigit() or not api_key:
        raise RuntimeError("Vast's per-instance teardown credentials are missing")
    root.mkdir(parents=True, exist_ok=True)
    (root / "supervisor_pid").write_text(str(os.getpid()))
    (root / "armed").write_text(str(deadline))
    while True:
        reason = shutdown_reason(root, deadline, time.time())
        if reason is not None:
            break
        time.sleep(2)
    print(f"Vast host teardown: {reason}", flush=True)
    # Give the local monitor time to consume the final status. Uploads must have
    # completed inside the job before it writes status; teardown is irreversible.
    if reason != "hard timeout":
        time.sleep(30)
    delay = 5
    while not destroy_once(instance_id, api_key):
        print("Vast teardown not confirmed; retrying", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 300)
