"""
Persisted, GUI-only dataset configuration store.

Remembers, across GUI restarts:
  * the last-used ``root_path`` for each predefined dataset name, and
  * user-defined custom (CSV-backed) image datasets, so a name typed once
    can be reused later without re-entering all its fields.

This file is read/written only by the GUI process (``gui.py``). The actual
training config handed to the worker subprocess is self-contained (see
``AutoMAXApp._build_config()``), so the worker never needs to open this
store -- it only exists for the GUI's own convenience/memory.
"""
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("gui_sessions") / "dataset_config.json"


class DatasetConfigStore:
    def __init__(self, path: str = str(_DEFAULT_PATH)):
        self.path = Path(path)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"root_paths": {}, "custom_image_datasets": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading dataset config from {self.path}: {e}")
            return {"root_paths": {}, "custom_image_datasets": {}}
        data.setdefault("root_paths", {})
        data.setdefault("custom_image_datasets", {})
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dataset config to {self.path}: {e}")

    # ------------------------------------------------------------------
    # Root paths (predefined datasets)
    # ------------------------------------------------------------------

    def get_root_path(self, dataset_name: str) -> str:
        with self._lock:
            return self._load()["root_paths"].get(dataset_name, "")

    def set_root_path(self, dataset_name: str, root_path: str) -> None:
        if not dataset_name or not root_path:
            return
        with self._lock:
            data = self._load()
            if data["root_paths"].get(dataset_name) != root_path:
                data["root_paths"][dataset_name] = root_path
                self._save(data)

    # ------------------------------------------------------------------
    # Custom (CSV-backed) image datasets
    # ------------------------------------------------------------------

    def get_custom_image_dataset(self, name: str) -> Optional[dict]:
        with self._lock:
            return self._load()["custom_image_datasets"].get(name)

    def custom_image_dataset_names(self) -> List[str]:
        with self._lock:
            return list(self._load()["custom_image_datasets"].keys())

    def add_custom_image_dataset(
        self, name: str, definition: dict, reserved_names: Iterable[str] = ()
    ) -> None:
        """
        Persist a new custom image dataset definition.

        Raises:
            ValueError: if ``name`` is blank, or collides (case-insensitively)
                with a name in ``reserved_names`` (the predefined dataset
                list) or with an already-added custom dataset.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("Dataset name cannot be blank.")

        lname = name.lower()
        if lname in {n.lower() for n in reserved_names}:
            raise ValueError(f"'{name}' is already a predefined dataset name.")

        with self._lock:
            data = self._load()
            if lname in {n.lower() for n in data["custom_image_datasets"]}:
                raise ValueError(f"A custom dataset named '{name}' has already been added.")
            data["custom_image_datasets"][name] = definition
            self._save(data)
        logger.info(f"Added custom image dataset '{name}'")


_store: Optional[DatasetConfigStore] = None


def get_dataset_config_store() -> DatasetConfigStore:
    """Get the global (GUI-process-wide) DatasetConfigStore instance."""
    global _store
    if _store is None:
        _store = DatasetConfigStore()
    return _store
