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
def test_video_with_audio_path():
    """Create a 10-frame 64x64 test video WITH a silent audio track."""
    tmpdir = tempfile.mkdtemp()
    raw_path = os.path.join(tmpdir, "raw.mp4")
    path = os.path.join(tmpdir, "test_with_audio.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, 30.0, (64, 64))
    for i in range(10):
        frame = np.full((64, 64, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    import subprocess
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-i", raw_path,
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    os.remove(raw_path)

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


from PIL import Image


@pytest.fixture
def test_image_path():
    """Create a single 64x64 test image and return its path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_image.png")
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(path)
    yield path
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def test_image_folder(tmp_path):
    """Create a folder with 3 test images."""
    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(tmp_path / f"img_{i:03d}.png")
    return str(tmp_path)
