import numpy as np

from src.core.config import BackgroundMode


def compose_frame(
    bgr_frame: np.ndarray,
    alpha_mask: np.ndarray,
    mode: BackgroundMode,
) -> np.ndarray:
    """Compose a BGR frame with an alpha mask according to the background mode.

    Args:
        bgr_frame: BGR uint8 array, shape (H, W, 3).
        alpha_mask: uint8 array, shape (H, W), values 0-255.
        mode: How to compose the output.

    Returns:
        Composed frame as uint8 array.
        - TRANSPARENT: RGBA shape (H, W, 4)
        - GREEN/BLUE: RGB shape (H, W, 3)
        - MASK_BW/MASK_WB: RGB shape (H, W, 3)
        - SIDE_BY_SIDE: RGB shape (H, W*2, 3)
    """
    rgb = bgr_frame[:, :, ::-1]  # BGR -> RGB

    if mode == BackgroundMode.TRANSPARENT:
        return np.dstack([rgb, alpha_mask])

    if mode in (BackgroundMode.GREEN, BackgroundMode.BLUE):
        bg_color = np.array([0, 255, 0], dtype=np.uint8) if mode == BackgroundMode.GREEN \
            else np.array([0, 0, 255], dtype=np.uint8)
        alpha_f = alpha_mask.astype(np.float32) / 255.0
        alpha_3 = alpha_f[:, :, np.newaxis]
        bg = np.full_like(rgb, bg_color)
        blended = (rgb.astype(np.float32) * alpha_3 + bg.astype(np.float32) * (1.0 - alpha_3))
        return blended.clip(0, 255).astype(np.uint8)

    if mode == BackgroundMode.MASK_BW:
        return np.dstack([alpha_mask, alpha_mask, alpha_mask])

    if mode == BackgroundMode.MASK_WB:
        inverted = 255 - alpha_mask
        return np.dstack([inverted, inverted, inverted])

    if mode == BackgroundMode.SIDE_BY_SIDE:
        mask_rgb = np.dstack([alpha_mask, alpha_mask, alpha_mask])
        return np.hstack([rgb, mask_rgb])

    raise ValueError(f"Unknown background mode: {mode}")
