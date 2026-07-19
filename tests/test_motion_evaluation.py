from pathlib import Path

from bello_mujoco import motion_evaluation
from bello_mujoco.motion_evaluation import (
    MotionEvaluationRequest,
    MotionEvaluationWorker,
)


class _FakeQueue:
    def __init__(self) -> None:
        self.values = []
        self.closed = False

    def put(self, value) -> None:
        self.values.append(value)

    def get(self, *, timeout: float):
        del timeout
        return self.values.pop(0)

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        pass


class _FakeProcess:
    def __init__(self, target, args, name: str) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.exitcode = None
        self.closed = False

    def start(self) -> None:
        self.target(*self.args)
        self.exitcode = 0

    def is_alive(self) -> bool:
        return self.exitcode is None

    def join(self, timeout: float) -> None:
        del timeout

    def terminate(self) -> None:
        self.exitcode = -15

    def close(self) -> None:
        self.closed = True


class _FakeContext:
    def __init__(self) -> None:
        self.processes = []
        self.queues = []

    def Queue(self, *, maxsize: int):
        assert maxsize == 1
        queue = _FakeQueue()
        self.queues.append(queue)
        return queue

    def Process(self, *, target, args, name: str):
        process = _FakeProcess(target, args, name)
        self.processes.append(process)
        return process


def test_motion_evaluation_uses_a_fresh_process_per_video(monkeypatch) -> None:
    context = _FakeContext()
    monkeypatch.setattr(motion_evaluation.mp, "get_context", lambda method: context)
    monkeypatch.setattr(
        motion_evaluation,
        "render_checkpoint_with_stats",
        lambda *args, **kwargs: {
            "rendered_steps": 1,
            "episode_return": 0.0,
            "terminal": False,
            "timeout": False,
            "completion_fraction": 1.0,
            "root_position_rms_m": 0.0,
            "root_orientation_rms_deg": 0.0,
            "joint_position_rms_rad": 0.0,
        },
    )
    request = MotionEvaluationRequest(
        checkpoint=Path("checkpoint"),
        output_path=Path("video.mp4"),
        steps=1,
        width=32,
        height=24,
        fps=20,
    )
    worker = MotionEvaluationWorker()

    worker.evaluate(request)
    worker.evaluate(request)

    assert len(context.processes) == 2
    assert all(process.closed for process in context.processes)
    assert all(queue.closed for queue in context.queues)
    assert worker._process is None
