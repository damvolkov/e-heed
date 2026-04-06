"""Training specification models for wakeword trainer."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated

from annotated_types import Ge, Gt, Le
from pydantic import BaseModel, ConfigDict

##### TYPES #####

Probability = Annotated[float, Ge(0.0), Le(1.0)]
PositiveInt = Annotated[int, Gt(0)]

##### ENUMS #####


class TrainPhase(StrEnum):
    GENERATE = auto()
    AUGMENT = auto()
    TRAIN = auto()


class TTSProvider(StrEnum):
    PIPER = auto()
    EVOICE = auto()


##### VOICE MAPS #####

EVOICE_VOICE_MAP: dict[str, str] = {
    "en": "af_heart",
    "es": "ef_dora",
    "fr": "ff_siwis",
    "de": "af_heart",
    "it": "if_sara",
    "pt": "pf_dora",
    "nl": "af_heart",
    "ja": "jf_alpha",
    "hi": "hf_alpha",
    "zh": "zf_xiaobei",
}

PIPER_MODEL_MAP: dict[str, str] = {
    "en": "en_US-libritts_r-medium.pt",
    "es": "es_ES-mls-medium.pt",
    "de": "de_DE-mls-medium.pt",
    "fr": "fr_FR-mls-medium.pt",
    "nl": "nl_NL-mls-medium.pt",
}

##### SPEC MODELS #####


class BatchSpec(BaseModel):
    """Batch size per class for training data loader."""

    model_config = ConfigDict(extra="forbid")

    ACAV100M_sample: PositiveInt = 1024
    adversarial_negative: PositiveInt = 50
    positive: PositiveInt = 50


class TrainSpec(BaseModel):
    """Full training specification — portable, no absolute paths."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    model_name: str
    target_phrase: list[str]

    # TTS
    tts: TTSProvider = TTSProvider.EVOICE
    langs: list[str] = ["en"]
    evoice_url: str = "http://localhost:45140"

    # Generation
    n_samples: PositiveInt = 10000
    n_samples_val: PositiveInt = 2000
    tts_batch_size: PositiveInt = 50
    custom_negative_phrases: list[str] = []

    # Augmentation
    augmentation_batch_size: PositiveInt = 16
    augmentation_rounds: PositiveInt = 1

    # Model architecture
    model_type: str = "dnn"
    layer_size: PositiveInt = 32

    # Training
    steps: PositiveInt = 20000
    max_negative_weight: PositiveInt = 1500
    target_false_positives_per_hour: float = 0.2
    target_accuracy: Probability = 0.6
    target_recall: Probability = 0.25

    # Batch config
    batch_n_per_class: BatchSpec = BatchSpec()
