#!/usr/bin/env python3
"""Download BiRefNet models for offline use.

Run: python download_models.py              # download general (default)
     python download_models.py --all        # download all models
     python download_models.py general lite # download specific models
"""

import sys

from src.core.config import MODEL_REGISTRY
from src.core.model_downloader import ModelDownloader

MODELS_DIR = "./models"


def main():
    args = sys.argv[1:]
    downloader = ModelDownloader(MODELS_DIR)

    if not args:
        print("No arguments. Downloading birefnet-general (default).")
        print(f"Use --all to download all models, or specify: {', '.join(MODEL_REGISTRY.keys())}")
        downloader.download_model("general")
        print("Done.")
        return

    if "--all" in args:
        keys = list(MODEL_REGISTRY.keys())
    else:
        keys = []
        for arg in args:
            if arg not in MODEL_REGISTRY:
                print(f"Unknown model: {arg}")
                print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
                sys.exit(1)
            keys.append(arg)

    for key in keys:
        info = MODEL_REGISTRY[key]
        print(f"\nDownloading {key} ({info.display_name})...")
        downloader.download_model(key)
        print(f"  Done: {key}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
