import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from src.core.config import MODELS


def get_model_path(model_name: str, models_dir: str) -> str:
    """Map a model display name to its local directory path."""
    dir_name = MODELS[model_name]
    return os.path.join(models_dir, dir_name)


def detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str) -> AutoModelForImageSegmentation:
    """Load BiRefNet model from a local directory."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = AutoModelForImageSegmentation.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    # Ensure float32 to avoid MPS float16/float32 mismatch
    model.float()
    model.eval()
    return model


def _make_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

_transform_cache: dict[int, transforms.Compose] = {}

def _get_transform(resolution: int) -> transforms.Compose:
    if resolution not in _transform_cache:
        _transform_cache[resolution] = _make_transform(resolution)
    return _transform_cache[resolution]


def predict(model, frame: np.ndarray, device: str, resolution: int = 1024) -> np.ndarray:
    """Run BiRefNet on a single BGR frame, return alpha mask at original resolution.

    Args:
        model: Loaded BiRefNet model.
        frame: BGR uint8 numpy array, shape (H, W, 3).
        device: 'cuda', 'mps', or 'cpu'.
        resolution: Input resolution for the model (default 1024).

    Returns:
        Alpha mask as uint8 numpy array, shape (H, W), values 0-255.
    """
    orig_h, orig_w = frame.shape[:2]
    transform = _get_transform(resolution)

    # BGR -> RGB -> PIL
    rgb = frame[:, :, ::-1]
    image = Image.fromarray(rgb)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                preds = model(input_tensor)[-1]
        else:
            preds = model(input_tensor)[-1]
        pred = torch.sigmoid(preds[0, 0])

    # Resize back to original resolution
    pred_resized = torch.nn.functional.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Convert to uint8 numpy
    alpha = (pred_resized * 255).clamp(0, 255).byte().cpu().numpy()
    return alpha


def predict_batch(
    model,
    frames: list[np.ndarray],
    device: str,
    resolution: int = 1024,
) -> list[np.ndarray]:
    """Run BiRefNet on a batch of BGR frames, return alpha masks at original resolutions.

    Args:
        model: Loaded BiRefNet model.
        frames: List of BGR uint8 numpy arrays, each shape (H, W, 3).
        device: 'cuda', 'mps', or 'cpu'.
        resolution: Input resolution for the model (default 1024).

    Returns:
        List of alpha masks as uint8 numpy arrays, each shape (H, W), values 0-255.
    """
    if len(frames) == 0:
        return []
    transform = _get_transform(resolution)
    orig_sizes = [(f.shape[0], f.shape[1]) for f in frames]
    tensors = []
    for frame in frames:
        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        tensors.append(transform(image))
    batch_tensor = torch.stack(tensors).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                preds = model(batch_tensor)[-1]
        else:
            preds = model(batch_tensor)[-1]
        preds = torch.sigmoid(preds[:, 0])
    masks = []
    for i, (orig_h, orig_w) in enumerate(orig_sizes):
        pred_resized = torch.nn.functional.interpolate(
            preds[i].unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        alpha = (pred_resized * 255).clamp(0, 255).byte().cpu().numpy()
        masks.append(alpha)
    return masks
