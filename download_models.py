#!/usr/bin/env python3
"""Download BiRefNet-general model for offline use.

Run: python download_models.py
Downloads to: ./models/birefnet-general/
"""

import os

from huggingface_hub import snapshot_download


def main():
    model_dir = "./models/birefnet-general"
    os.makedirs(model_dir, exist_ok=True)

    print("Downloading BiRefNet-general model...")
    print(f"Destination: {os.path.abspath(model_dir)}")

    snapshot_download(
        repo_id="zhengpeng7/BiRefNet",
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Done. Model saved to {os.path.abspath(model_dir)}")


if __name__ == "__main__":
    main()
