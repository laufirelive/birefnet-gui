import os
import tempfile

import cv2
import numpy as np
import pytest


@pytest.fixture
def test_video_path():
    """Create a tiny 10-frame 64x64 test video and return its path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (64, 64))
    for i in range(10):
        frame = np.full((64, 64, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    yield path
    if os.path.exists(path):
        os.remove(path)
    os.rmdir(tmpdir)


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for output files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
