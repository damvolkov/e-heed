"""Pipeline orchestrator — run training phases via subprocess to openwakeword."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import structlog
import yaml
from trainer.config import LEGACY_DIR, MODELS_WW_DIR, wakeword_dir, write_oww_config
from trainer.generate import generate
from trainer.spec import TrainPhase, TrainSpec, TTSProvider

log = structlog.get_logger("trainer.pipeline")

_TRAIN_PY: str = str(LEGACY_DIR / "openwakeword" / "openwakeword" / "train.py")


_OWW_REPO: str = str(LEGACY_DIR / "openwakeword")


def _run_oww(config_path: Path, *flags: str) -> subprocess.CompletedProcess[bytes]:
    """Run openwakeword train.py as a subprocess with legacy repo on PYTHONPATH."""
    cmd = [sys.executable, _TRAIN_PY, "--training_config", str(config_path), *flags]
    env = {**os.environ, "PYTHONPATH": _OWW_REPO}
    log.info("subprocess_start", cmd=" ".join(cmd))
    result = subprocess.run(cmd, env=env)
    # Allow tflite conversion failure (onnx_tf not installed) — we only need ONNX
    if result.returncode != 0:
        log.warning("subprocess_nonzero", returncode=result.returncode, cmd=cmd[2])
    return result


def run_phase(spec: TrainSpec, word: str, phase: TrainPhase) -> None:
    """Run a single training phase."""
    oww_yaml = write_oww_config(spec, word)
    log.info("phase_start", phase=phase.value, word=word)

    match phase:
        case TrainPhase.GENERATE:
            if spec.tts == TTSProvider.EVOICE:
                generate(spec, word)
            else:
                _run_oww(oww_yaml, "--generate_clips")

        case TrainPhase.AUGMENT:
            _run_oww(oww_yaml, "--augment_clips")

        case TrainPhase.TRAIN:
            _run_oww(oww_yaml, "--train_model")

    log.info("phase_complete", phase=phase.value, word=word)


def run_all(spec: TrainSpec, word: str) -> Path:
    """Run full pipeline: generate -> augment -> train. Returns path to deployed .onnx."""
    for phase in TrainPhase:
        run_phase(spec, word, phase)

    return deploy_model(spec, word)


def deploy_model(spec: TrainSpec, word: str) -> Path:
    """Copy trained .onnx + spec to models/ww/."""
    wd = wakeword_dir(word)
    onnx_src = wd / spec.model_name / f"{spec.model_name}.onnx"

    if not onnx_src.exists():
        msg = f"Training did not produce {onnx_src}"
        raise FileNotFoundError(msg)

    MODELS_WW_DIR.mkdir(parents=True, exist_ok=True)

    onnx_dest = MODELS_WW_DIR / f"{spec.model_name}.onnx"
    shutil.copy2(onnx_src, onnx_dest)
    log.info("deploy_onnx", src=str(onnx_src), dest=str(onnx_dest))

    spec_dest = MODELS_WW_DIR / f"{spec.model_name}_config.yaml"
    spec_dest.write_text(yaml.dump(spec.model_dump(), default_flow_style=False, sort_keys=False))
    log.info("deploy_spec", dest=str(spec_dest))

    return onnx_dest
