"""YAML config I/O and path resolution for trainer specs."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from trainer.spec import TrainSpec

##### PATHS #####

TRAINER_ROOT: Final[Path] = Path(__file__).resolve().parent
TRAINER_MODELS_DIR: Final[Path] = TRAINER_ROOT / "models"
LEGACY_DIR: Final[Path] = TRAINER_ROOT / "legacy"
MODELS_WW_DIR: Final[Path] = TRAINER_ROOT.parent / "models" / "ww"

##### WAKEWORD DIRECTORIES #####


def wakeword_dir(word: str) -> Path:
    """Artifact directory: trainer/models/<word>/."""
    return TRAINER_MODELS_DIR / word


def spec_path(word: str) -> Path:
    """Spec YAML path: trainer/<word>/spec.yaml."""
    return wakeword_dir(word) / "spec.yaml"


def oww_config_path(word: str) -> Path:
    """Generated openwakeword config: trainer/<word>/oww_config.yaml."""
    return wakeword_dir(word) / "oww_config.yaml"


##### LEGACY PATH RESOLUTION #####


def resolve_legacy_paths() -> dict[str, object]:
    """Resolve legacy artifact paths to absolute strings for openwakeword YAML."""
    return {
        "piper_sample_generator_path": str(LEGACY_DIR / "piper-sample-generator"),
        "background_paths": [str(LEGACY_DIR / "audioset" / "audio")],
        "background_paths_duplication_rate": [1],
        "rir_paths": [str(LEGACY_DIR / "MIT_environmental_impulse_responses" / "16khz")],
        "false_positive_validation_data_path": str(LEGACY_DIR / "validation_set_features.npy"),
        "feature_data_files": {
            "ACAV100M_sample": str(LEGACY_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
        },
    }


##### LOAD / SAVE #####


def load_spec(word: str) -> TrainSpec:
    """Load a TrainSpec from trainer/<word>/spec.yaml."""
    path = spec_path(word)
    if not path.exists():
        msg = f"No training spec found at {path}"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(path.read_text())
    return TrainSpec.model_validate(data)


def save_spec(word: str, spec: TrainSpec) -> Path:
    """Save a TrainSpec to trainer/<word>/spec.yaml."""
    wd = wakeword_dir(word)
    wd.mkdir(parents=True, exist_ok=True)
    path = spec_path(word)
    path.write_text(yaml.dump(spec.model_dump(mode="json"), default_flow_style=False, sort_keys=False))
    return path


def list_specs() -> list[str]:
    """List all wakewords that have a spec.yaml."""
    if not TRAINER_MODELS_DIR.exists():
        return []
    return sorted(d.name for d in TRAINER_MODELS_DIR.iterdir() if d.is_dir() and (d / "spec.yaml").exists())


##### OWW CONFIG GENERATION #####


def build_oww_config(spec: TrainSpec, word: str) -> dict[str, object]:
    """Build openwakeword-compatible YAML dict from a TrainSpec."""
    legacy = resolve_legacy_paths()
    exclude = {"tts", "langs", "evoice_url", "target_accuracy", "target_recall"}
    return {
        **spec.model_dump(exclude=exclude),
        **legacy,
        "output_dir": str(wakeword_dir(word)),
    }


def write_oww_config(spec: TrainSpec, word: str) -> Path:
    """Write openwakeword YAML config to trainer/<word>/oww_config.yaml."""
    wd = wakeword_dir(word)
    wd.mkdir(parents=True, exist_ok=True)
    data = build_oww_config(spec, word)
    path = oww_config_path(word)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return path
