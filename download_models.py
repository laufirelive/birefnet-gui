#!/usr/bin/env python3
"""Download BiRefNet models for offline use.

Run: python download_models.py              # download general (default)
     python download_models.py --all        # download all models
     python download_models.py general lite # download specific models
"""

import os
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "general": ("birefnet-general", "zhengpeng7/BiRefNet"),
    "lite": ("birefnet-lite", "zhengpeng7/BiRefNet_lite"),
    "matting": ("birefnet-matting", "zhengpeng7/BiRefNet-matting"),
    "hr": ("birefnet-hr", "zhengpeng7/BiRefNet_HR"),
    "hr-matting": ("birefnet-hr-matting", "zhengpeng7/BiRefNet_HR-matting"),
    "dynamic": ("birefnet-dynamic", "zhengpeng7/BiRefNet_dynamic"),
}


def download_model(key: str, models_dir: str = "./models"):
    dir_name, repo_id = MODELS[key]
    local_path = os.path.join(models_dir, dir_name)
    os.makedirs(local_path, exist_ok=True)

    print(f"\nDownloading {key} from {repo_id}...")
    print(f"  -> {os.path.abspath(local_path)}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"  Done: {key}")


def main():
    args = sys.argv[1:]

    if not args:
        print("No arguments. Downloading birefnet-general (default).")
        print("Use --all to download all models, or specify: general lite matting hr hr-matting dynamic")
        download_model("general")
        return

    if "--all" in args:
        keys = list(MODELS.keys())
    else:
        keys = []
        for arg in args:
            if arg not in MODELS:
                print(f"Unknown model: {arg}")
                print(f"Available: {', '.join(MODELS.keys())}")
                sys.exit(1)
            keys.append(arg)

    for key in keys:
        download_model(key)

    print(f"\nAll done. Models saved to {os.path.abspath('./models')}")


if __name__ == "__main__":
    main()
