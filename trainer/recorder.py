"""VAD-based smart voice recorder — trims silence, saves 16kHz PCM WAV clips."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import torch
from rich.console import Console
from silero_vad import get_speech_timestamps, load_silero_vad
from trainer.config import wakeword_dir

##### CONSTANTS #####

_SAMPLE_RATE: Final[int] = 16000
_VAD_THRESHOLD: Final[float] = 0.35
_SPEECH_PAD_MS: Final[int] = 50
_MIN_SPEECH_MS: Final[int] = 150
_MIN_SILENCE_MS: Final[int] = 100
_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"_(\d+)\.wav$")

console = Console(stderr=True)

##### HELPERS #####


def _find_next_index(output_dir: Path, name: str) -> int:
    """Scan existing files and return the next available index."""
    max_idx = 0
    for f in output_dir.glob(f"{name}_*.wav"):
        if m := _FILENAME_PATTERN.search(f.name):
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _record_raw(duration: float) -> np.ndarray:
    """Record raw audio from the default input device."""
    n_samples = int(duration * _SAMPLE_RATE)
    audio = sd.rec(n_samples, samplerate=_SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def _extract_voice(audio: np.ndarray, vad_model: object, target_duration: float) -> np.ndarray | None:
    """Extract voice segment via VAD, pad or keep as-is based on target duration."""
    tensor = torch.from_numpy(audio).float()

    timestamps = get_speech_timestamps(
        tensor,
        vad_model,
        threshold=_VAD_THRESHOLD,
        sampling_rate=_SAMPLE_RATE,
        min_speech_duration_ms=_MIN_SPEECH_MS,
        min_silence_duration_ms=_MIN_SILENCE_MS,
        speech_pad_ms=_SPEECH_PAD_MS,
    )

    if not timestamps:
        return None

    start = timestamps[0]["start"]
    end = timestamps[-1]["end"]
    voice = audio[start:end]

    target_samples = int(target_duration * _SAMPLE_RATE)
    if len(voice) < target_samples:
        pad_total = target_samples - len(voice)
        pad_left = pad_total // 2
        voice = np.pad(voice, (pad_left, pad_total - pad_left), mode="constant", constant_values=0.0)

    return voice


def _save_wav(path: Path, audio: np.ndarray) -> None:
    """Save float32 audio as 16-bit PCM WAV at 16kHz mono."""
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), _SAMPLE_RATE, pcm)


##### PUBLIC #####


def run_recorder(word: str, *, duration: float = 3.0, target: float = 1.0) -> None:
    """Interactive VAD-based recorder — saves clips to trainer/<word>/<word>/positive_train/."""
    output_dir = wakeword_dir(word) / word / "positive_train"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"real_{word}"
    idx = _find_next_index(output_dir, prefix)

    console.print("[cyan]Loading Silero VAD...[/cyan]")
    vad_model = load_silero_vad(onnx=True)

    console.print(
        f"[green]Recording to:[/green] {output_dir}\n"
        f"  Capture: {duration}s | Target clip: {target}s | VAD: {_VAD_THRESHOLD}\n"
        f"  Starting from take #{idx}\n"
    )

    while True:
        filename = f"{prefix}_{idx:03d}.wav"
        try:
            user_input = input(f"Enter to record '{filename}' (q to quit): ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() == "q":
            break

        console.print("[yellow]Recording...[/yellow]")
        raw_audio = _record_raw(duration)

        voice = _extract_voice(raw_audio, vad_model, target)
        if voice is None:
            console.print("[red]No voice detected — skipping.[/red]\n")
            continue

        out_path = output_dir / filename
        _save_wav(out_path, voice)
        voice_ms = len(voice) / _SAMPLE_RATE * 1000
        console.print(f"[green]Saved:[/green] {out_path} ({voice_ms:.0f}ms)\n")
        idx += 1

    console.print("[cyan]Done.[/cyan]")


##### STANDALONE #####


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python recorder.py <word> [--duration N] [--target N][/red]")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--target", type=float, default=1.0)
    args = parser.parse_args()
    run_recorder(args.name, duration=args.duration, target=args.target)
