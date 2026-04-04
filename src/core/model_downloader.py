import os
import shutil

from huggingface_hub import snapshot_download

from src.core.config import MODEL_REGISTRY

HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"


class ModelDownloader:
    """Manages model installation: check status, download, delete."""

    def __init__(self, models_dir: str):
        self._models_dir = models_dir

    def get_installed_models(self) -> list[str]:
        """Return list of installed model keys."""
        installed = []
        for key, info in MODEL_REGISTRY.items():
            model_path = os.path.join(self._models_dir, info.dir_name)
            if os.path.isdir(model_path):
                installed.append(key)
        return installed

    def is_installed(self, model_key: str) -> bool:
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            return False
        return os.path.isdir(os.path.join(self._models_dir, info.dir_name))

    def delete_model(self, model_key: str) -> None:
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            raise FileNotFoundError(f"Unknown model key: {model_key}")
        model_path = os.path.join(self._models_dir, info.dir_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model not installed: {model_key}")
        shutil.rmtree(model_path)

    def download_model(self, model_key: str, use_mirror: bool = True) -> str:
        """Download a model. Tries hf-mirror first, falls back to official.
        Returns the local path of the downloaded model.
        """
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            raise ValueError(f"Unknown model key: {model_key}")

        local_path = os.path.join(self._models_dir, info.dir_name)
        os.makedirs(local_path, exist_ok=True)

        if use_mirror:
            try:
                return self._do_download(info.repo_id, local_path, endpoint=HF_MIRROR)
            except Exception:
                pass

        return self._do_download(info.repo_id, local_path, endpoint=HF_OFFICIAL)

    def _do_download(self, repo_id: str, local_path: str, endpoint: str) -> str:
        old_endpoint = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        finally:
            if old_endpoint is not None:
                os.environ["HF_ENDPOINT"] = old_endpoint
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
        return local_path
