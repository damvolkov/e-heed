"""CLI commands for wakeword training — train and record."""

from __future__ import annotations

import sys
from typing import Annotated

import cyclopts
from rich.console import Console
from rich.table import Table
from trainer.config import list_specs, load_spec, save_spec, spec_path
from trainer.pipeline import run_all, run_phase
from trainer.spec import TrainPhase, TrainSpec, TTSProvider

train_app = cyclopts.App(name="train", help="Wakeword model training pipeline.")
record_app = cyclopts.App(name="record", help="Interactive VAD-based wakeword recorder.")
console = Console(stderr=True)


def _print_tagged(step: str, msg: str, *, error: bool = False) -> None:
    style = "red" if error else "green"
    console.print(f"[cyan]\\[TRAIN][/cyan] [bold {style}]\\[{step}][/bold {style}] {msg}")


##### TRAIN COMMAND #####


@train_app.default
def train(
    word: Annotated[str | None, cyclopts.Parameter(name="word", help="Wakeword name")] = None,
    *,
    phase: Annotated[
        TrainPhase | None, cyclopts.Parameter(name="--phase", help="Single phase (generate|augment|train)")
    ] = None,
    list_configs: Annotated[bool, cyclopts.Parameter(name="--list", help="List available training specs")] = False,
    tts: Annotated[
        TTSProvider, cyclopts.Parameter(name="--tts", help="TTS provider (piper|evoice)")
    ] = TTSProvider.EVOICE,
    langs: Annotated[str | None, cyclopts.Parameter(name="--langs", help="Comma-separated language codes")] = None,
    samples: Annotated[int | None, cyclopts.Parameter(name="--samples", help="Number of training samples")] = None,
    steps: Annotated[int | None, cyclopts.Parameter(name="--steps", help="Number of training steps")] = None,
) -> None:
    """Train a wakeword model.

    Usage:
      eheed train --list                              # list specs
      eheed train eager                               # full pipeline
      eheed train eager --tts evoice --langs en,es    # new spec
      eheed train eager --phase generate              # single phase
      eheed train eager --samples 5000 --steps 20000  # override params
    """
    if list_configs:
        specs = list_specs()
        if not specs:
            _print_tagged("LIST", "No training specs found under trainer/")
            return
        table = Table(title="Training Specs")
        table.add_column("Wakeword", style="cyan")
        table.add_column("Path", style="dim")
        for name in specs:
            table.add_row(name, str(spec_path(name)))
        console.print(table)
        return

    if word is None:
        _print_tagged("ERROR", "Provide a wakeword name or use --list", error=True)
        sys.exit(1)

    # Load or create spec
    try:
        spec = load_spec(word)
        _print_tagged("LOAD", f"Loaded spec for '{word}'")
    except FileNotFoundError:
        _print_tagged("NEW", f"Creating new spec for '{word}'")
        spec = TrainSpec(model_name=word, target_phrase=[word])

    # Apply CLI overrides
    spec.tts = tts
    if langs is not None:
        spec.langs = [lang.strip() for lang in langs.split(",")]
    if samples is not None:
        spec.n_samples = samples
        spec.n_samples_val = max(100, samples // 5)
    if steps is not None:
        spec.steps = steps

    save_spec(word, spec)

    # Run
    if phase is not None:
        _print_tagged("RUN", f"Phase: {phase.value}")
        run_phase(spec, word, phase)
        _print_tagged("DONE", f"Phase '{phase.value}' complete")
    else:
        _print_tagged("RUN", "Full pipeline: generate → augment → train")
        onnx_path = run_all(spec, word)
        _print_tagged("DONE", f"Model deployed to {onnx_path}")


##### RECORD COMMAND #####


@record_app.default
def record(
    word: Annotated[str, cyclopts.Parameter(name="word", help="Wakeword name to record")],
    *,
    duration: Annotated[float, cyclopts.Parameter(name="--duration", help="Capture window (seconds)")] = 3.0,
    target: Annotated[float, cyclopts.Parameter(name="--target", help="Target clip duration (seconds)")] = 1.0,
) -> None:
    """Record wakeword clips interactively using VAD.

    Usage:
      eheed record eager                # record clips for 'eager'
      eheed record eager --duration 5   # longer capture window
    """
    from trainer.recorder import run_recorder

    run_recorder(word, duration=duration, target=target)
