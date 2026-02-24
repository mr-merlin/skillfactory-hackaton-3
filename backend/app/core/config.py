import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if _PROJECT_ROOT.name == "backend":
    _PROJECT_ROOT = _PROJECT_ROOT.parent


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_perfume_dir: Path
    data_organ_dir: Path


def get_settings() -> Settings:
    perfume_dir = Path(os.environ.get("DATA_PERFUME_DIR", _PROJECT_ROOT / "data"))
    organ_dir = Path(os.environ.get("DATA_ORGAN_DIR", _PROJECT_ROOT / "data"))
    return Settings(project_root=_PROJECT_ROOT, data_perfume_dir=perfume_dir, data_organ_dir=organ_dir)
