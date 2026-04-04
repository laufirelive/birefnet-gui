import numpy as np
import pytest

from src.core.compositing import compose_frame
from src.core.config import BackgroundMode


@pytest.fixture
def sample_frame():
    """A 4x4 BGR frame with known pixel values."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 100  # B
    frame[:, :, 1] = 150  # G
    frame[:, :, 2] = 200  # R
    return frame


@pytest.fixture
def sample_alpha():
    """A 4x4 alpha mask: top half opaque (255), bottom half transparent (0)."""
    alpha = np.zeros((4, 4), dtype=np.uint8)
    alpha[:2, :] = 255  # top half opaque
    return alpha


class TestComposeTransparent:
    def test_output_shape_is_rgba(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.TRANSPARENT)
        assert result.shape == (4, 4, 4)
        assert result.dtype == np.uint8

    def test_rgb_channels_are_bgr_to_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.TRANSPARENT)
        assert result[0, 0, 0] == 200  # R
        assert result[0, 0, 1] == 150  # G
        assert result[0, 0, 2] == 100  # B

    def test_alpha_channel_preserved(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.TRANSPARENT)
        assert result[0, 0, 3] == 255  # top half opaque
        assert result[3, 0, 3] == 0    # bottom half transparent


class TestComposeGreen:
    def test_output_shape_is_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.GREEN)
        assert result.shape == (4, 4, 3)
        assert result.dtype == np.uint8

    def test_opaque_pixels_show_foreground(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.GREEN)
        assert result[0, 0, 0] == 200  # R
        assert result[0, 0, 1] == 150  # G
        assert result[0, 0, 2] == 100  # B

    def test_transparent_pixels_show_green(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.GREEN)
        assert result[3, 0, 0] == 0    # R
        assert result[3, 0, 1] == 255  # G
        assert result[3, 0, 2] == 0    # B


class TestComposeBlue:
    def test_transparent_pixels_show_blue(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.BLUE)
        assert result.shape == (4, 4, 3)
        assert result[3, 0, 0] == 0
        assert result[3, 0, 1] == 0
        assert result[3, 0, 2] == 255


class TestComposeMaskBW:
    def test_output_is_grayscale_as_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.MASK_BW)
        assert result.shape == (4, 4, 3)
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255
        assert result[3, 0, 0] == 0
        assert result[3, 0, 1] == 0
        assert result[3, 0, 2] == 0


class TestComposeMaskWB:
    def test_inverted_mask(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.MASK_WB)
        assert result.shape == (4, 4, 3)
        assert result[0, 0, 0] == 0
        assert result[3, 0, 0] == 255


class TestComposeSideBySide:
    def test_output_width_doubled(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.SIDE_BY_SIDE)
        assert result.shape == (4, 8, 3)
        assert result.dtype == np.uint8

    def test_left_half_is_original_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.SIDE_BY_SIDE)
        assert result[0, 0, 0] == 200  # R
        assert result[0, 0, 1] == 150  # G
        assert result[0, 0, 2] == 100  # B

    def test_right_half_is_mask(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.SIDE_BY_SIDE)
        assert result[0, 4, 0] == 255  # top half opaque
        assert result[3, 4, 0] == 0    # bottom half transparent
