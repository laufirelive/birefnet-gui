import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


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


_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, frame: np.ndarray, device: str) -> np.ndarray:
    """Run BiRefNet on a single BGR frame, return alpha mask at original resolution.

    Args:
        model: Loaded BiRefNet model.
        frame: BGR uint8 numpy array, shape (H, W, 3).
        device: 'cuda', 'mps', or 'cpu'.

    Returns:
        Alpha mask as uint8 numpy array, shape (H, W), values 0-255.
    """
    orig_h, orig_w = frame.shape[:2]

    # BGR -> RGB -> PIL
    rgb = frame[:, :, ::-1]
    image = Image.fromarray(rgb)

    # Preprocess
    input_tensor = _transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
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
