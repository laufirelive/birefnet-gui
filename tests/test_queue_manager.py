import json
import os

import pytest

from src.core.config import InputType, OutputFormat, ProcessingConfig
from src.core.queue_manager import QueueManager
from src.core.queue_task import ProcessingPhase, QueueTask, TaskStatus


@pytest.fixture
def qm(tmp_path):
    brm_path = os.path.join(str(tmp_path), "queue.brm")
    return QueueManager(brm_path=brm_path)


def _make_task(input_path="/tmp/video.mp4", **kwargs) -> QueueTask:
    return QueueTask.create(
        input_path=input_path,
        input_type=kwargs.get("input_type", InputType.VIDEO),
        config=kwargs.get("config", ProcessingConfig()),
        output_dir=kwargs.get("output_dir"),
    )


class TestQueueManagerTaskList:
    def test_add_task(self, qm):
        t = _make_task()
        qm.add_task(t)
        assert len(qm.tasks) == 1
        assert qm.tasks[0].id == t.id

    def test_remove_task(self, qm):
        t = _make_task()
        qm.add_task(t)
        qm.remove_task(t.id)
        assert len(qm.tasks) == 0

    def test_remove_nonexistent_is_noop(self, qm):
        qm.remove_task("ghost")

    def test_move_task(self, qm):
        t1 = _make_task("/tmp/a.mp4")
        t2 = _make_task("/tmp/b.mp4")
        t3 = _make_task("/tmp/c.mp4")
        qm.add_task(t1)
        qm.add_task(t2)
        qm.add_task(t3)
        qm.move_task(t3.id, 0)
        assert [t.id for t in qm.tasks] == [t3.id, t1.id, t2.id]

    def test_move_task_to_end(self, qm):
        t1 = _make_task("/tmp/a.mp4")
        t2 = _make_task("/tmp/b.mp4")
        qm.add_task(t1)
        qm.add_task(t2)
        qm.move_task(t1.id, 1)
        assert [t.id for t in qm.tasks] == [t2.id, t1.id]

    def test_clear_all(self, qm):
        qm.add_task(_make_task("/tmp/a.mp4"))
        qm.add_task(_make_task("/tmp/b.mp4"))
        qm.clear_all()
        assert len(qm.tasks) == 0

    def test_get_task(self, qm):
        t = _make_task()
        qm.add_task(t)
        found = qm.get_task(t.id)
        assert found is t

    def test_get_task_missing_returns_none(self, qm):
        assert qm.get_task("nope") is None


class TestQueueManagerPersistence:
    def test_save_and_load(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "queue.brm")
        qm1 = QueueManager(brm_path=brm_path)
        t1 = _make_task("/tmp/a.mp4")
        t1.status = TaskStatus.COMPLETED
        t2 = _make_task("/tmp/b.mp4")
        t2.status = TaskStatus.PROCESSING
        t2.progress = 50
        t2.total = 100
        qm1.add_task(t1)
        qm1.add_task(t2)
        qm1.save()
        assert os.path.exists(brm_path)
        qm2 = QueueManager(brm_path=brm_path)
        qm2.load()
        assert len(qm2.tasks) == 2
        assert qm2.tasks[0].id == t1.id
        assert qm2.tasks[0].status == TaskStatus.COMPLETED
        assert qm2.tasks[1].progress == 50

    def test_load_nonexistent_file_is_noop(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "nope.brm")
        qm = QueueManager(brm_path=brm_path)
        qm.load()
        assert len(qm.tasks) == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "sub", "dir", "queue.brm")
        qm = QueueManager(brm_path=brm_path)
        qm.add_task(_make_task())
        qm.save()
        assert os.path.exists(brm_path)

    def test_processing_tasks_saved_as_paused(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "queue.brm")
        qm1 = QueueManager(brm_path=brm_path)
        t = _make_task()
        t.status = TaskStatus.PROCESSING
        qm1.add_task(t)
        qm1.save()
        qm2 = QueueManager(brm_path=brm_path)
        qm2.load()
        assert qm2.tasks[0].status == TaskStatus.PAUSED
