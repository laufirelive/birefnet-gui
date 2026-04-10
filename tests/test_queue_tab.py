from src.core.config import InputType, ProcessingConfig
from src.core.queue_task import QueueTask
from src.gui.queue_tab import QueueTab


class _DummyQueueManager:
    def __init__(self, tasks):
        self._tasks = tasks
        self.save_calls = 0

    @property
    def tasks(self):
        return list(self._tasks)

    def next_pending_task(self):
        return None

    def get_task(self, task_id: str):
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def save(self):
        self.save_calls += 1


def test_progress_save_throttled_10s_and_phase_change_saves_immediately(qtbot, monkeypatch):
    task = QueueTask.create(
        input_path="/tmp/demo.mp4",
        input_type=InputType.VIDEO,
        config=ProcessingConfig(),
    )
    qm = _DummyQueueManager([task])
    tab = QueueTab(qm, lambda: ProcessingConfig())
    qtbot.addWidget(tab)

    now = [100.0]
    monkeypatch.setattr("src.gui.queue_tab.time.time", lambda: now[0])

    # First update changes phase from None -> inference, should save immediately.
    tab._on_task_progress(task.id, 1, 100, "inference")
    assert qm.save_calls == 1

    # Same phase within 10s should NOT trigger another save.
    now[0] = 105.0
    tab._on_task_progress(task.id, 2, 100, "inference")
    assert qm.save_calls == 1

    # Same phase after 10s should trigger periodic save.
    now[0] = 110.1
    tab._on_task_progress(task.id, 3, 100, "inference")
    assert qm.save_calls == 2

    # Phase transition should save immediately even within 10s.
    now[0] = 111.0
    tab._on_task_progress(task.id, 4, 100, "temporal_fix")
    assert qm.save_calls == 3
