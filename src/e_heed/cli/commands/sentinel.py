"""Sentinel command — run only the wakeword detection pipeline (foreground)."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated

import cyclopts
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from e_heed.shared.settings import AppConfig, settings

app = cyclopts.App(name="sentinel", help="Run the sentinel pipeline only (wakeword testing).")
console = Console(stderr=True)

_MODELS_DIR = settings.MODELS_DIR / "ww"


def _print_tagged(pipeline: str, step: str, msg: str, *, error: bool = False) -> None:
    style = "red" if error else "green"
    console.print(f"[cyan]\\[{pipeline}][/cyan] [bold {style}]\\[{step}][/bold {style}] {msg}")


def _available_models() -> list[str]:
    """Discover .onnx wakeword models in models/ww/."""
    if not _MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in _MODELS_DIR.glob("*.onnx"))


async def _run_sentinel_loop(config: AppConfig) -> None:
    """Standalone sentinel loop — detects wakewords, logs, resets, repeats."""
    from e_heed.daemon.adapters.audio import AudioAdapter
    from e_heed.daemon.sentinel.pipeline import SentinelPipeline
    from e_heed.shared.logger import configure_logging, logger

    configure_logging(
        "debug",
        idle_interval=config.logging.idle_interval,
        turn_interval=config.logging.turn_interval,
    )

    audio = AudioAdapter(config.audio)
    sentinel = SentinelPipeline(config.sentinel)

    logger.system("START", "starting audio device...")
    await audio.start()

    logger.system("START", f"sentinel-only mode — listening for '{sentinel.wakeword_name}'")

    try:
        while True:
            await sentinel.run(audio.queue)

            if sentinel.last_event:
                event = sentinel.last_event
                logger.sentinel(
                    "WAKEWORD",
                    f"detected! confidence={event.confidence:.2f} energy={event.energy:.4f}",
                )
    except asyncio.CancelledError:
        pass
    finally:
        await sentinel.stop()
        await audio.stop()
        logger.system("STOP", "sentinel stopped")


@app.default
def sentinel(
    *,
    model: Annotated[
        str | None, cyclopts.Parameter(name="--model", help="Wakeword model name (from models/ww/)")
    ] = None,
    threshold: Annotated[
        float | None, cyclopts.Parameter(name="--threshold", help="Wakeword confidence threshold (0.0-1.0)")
    ] = None,
    list_models: Annotated[
        bool, cyclopts.Parameter(name="--list", help="List available wakeword models and exit")
    ] = False,
) -> None:
    """Run sentinel pipeline only — wakeword detection without STT/LLM/TTS.

    Usage:
      eheed sentinel                        # default model from config.yaml
      eheed sentinel --model eager          # test 'eager' wakeword
      eheed sentinel --model maia --threshold 0.3
      eheed sentinel --list                 # show available models
    """
    if list_models:
        models = _available_models()
        if not models:
            console.print(f"[yellow]No .onnx models found in {_MODELS_DIR}[/yellow]")
            return
        table = Table(title="Available wakeword models")
        table.add_column("Model", style="cyan")
        table.add_column("Path", style="dim")
        for name in models:
            table.add_row(name, str(_MODELS_DIR / f"{name}.onnx"))
        console.print(table)
        return

    try:
        config = AppConfig.load()
    except ValidationError as exc:
        _print_tagged("SYSTEM", "ERROR", "invalid config.yaml", error=True)
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"])
            console.print(f"  [red]{loc}:[/red] {err['msg']}")
        sys.exit(1)

    if model is not None:
        available = _available_models()
        if available and model not in available:
            _print_tagged("SYSTEM", "ERROR", f"model '{model}' not found", error=True)
            console.print(f"  Available: {', '.join(available)}")
            sys.exit(1)
        config.sentinel.wakeword.model = model

    if threshold is not None:
        config.sentinel.wakeword.threshold = threshold

    try:
        asyncio.run(_run_sentinel_loop(config))
    except KeyboardInterrupt:
        _print_tagged("SYSTEM", "STOP", "interrupted")
