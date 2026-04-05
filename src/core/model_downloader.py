import os
import shutil

from huggingface_hub import snapshot_download

from src.core.config import MODEL_REGISTRY

HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"

ENDPOINTS = {
    "hf-mirror": HF_MIRROR,
    "huggingface": HF_OFFICIAL,
}


class ModelDownloader:
    """Manages model installation: check status, download, delete."""

    def __init__(self, models_dir: str):
        self._models_dir = models_dir

    def get_installed_models(self) -> list[str]:
        installed = []
        for key in MODEL_REGISTRY:
            if self.is_installed(key):
                installed.append(key)
        return installed

    def is_installed(self, model_key: str) -> bool:
        """A model is installed if its directory contains config.json."""
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            return False
        model_path = os.path.join(self._models_dir, info.dir_name)
        return os.path.isfile(os.path.join(model_path, "config.json"))

    def is_partial(self, model_key: str) -> bool:
        """Directory exists but download is incomplete (no config.json)."""
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            return False
        model_path = os.path.join(self._models_dir, info.dir_name)
        return os.path.isdir(model_path) and not os.path.isfile(
            os.path.join(model_path, "config.json")
        )

    def delete_model(self, model_key: str) -> None:
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            raise FileNotFoundError(f"Unknown model key: {model_key}")
        model_path = os.path.join(self._models_dir, info.dir_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model not installed: {model_key}")
        shutil.rmtree(model_path)

    def download_model(
        self,
        model_key: str,
        endpoint: str | None = None,
        tqdm_class=None,
    ) -> str:
        """Download a model. If no endpoint given, tries hf-mirror then official."""
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            raise ValueError(f"Unknown model key: {model_key}")
        local_path = os.path.join(self._models_dir, info.dir_name)
        os.makedirs(local_path, exist_ok=True)

        if endpoint:
            return self._do_download(info.repo_id, local_path, endpoint, tqdm_class)

        try:
            return self._do_download(info.repo_id, local_path, HF_MIRROR, tqdm_class)
        except Exception:
            pass
        return self._do_download(info.repo_id, local_path, HF_OFFICIAL, tqdm_class)

    def _do_download(
        self, repo_id: str, local_path: str, endpoint: str, tqdm_class=None,
    ) -> str:
        old_endpoint = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            kwargs = dict(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            if tqdm_class is not None:
                kwargs["tqdm_class"] = tqdm_class
            snapshot_download(**kwargs)
        finally:
            if old_endpoint is not None:
                os.environ["HF_ENDPOINT"] = old_endpoint
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
        return local_path
