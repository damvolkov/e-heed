"""TTS clip generation — e-voice (HTTP) and Piper (legacy) providers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import soundfile as sf
import structlog
from trainer.spec import EVOICE_VOICE_MAP, PIPER_MODEL_MAP, TTSProvider

log = structlog.get_logger("trainer.tts")

_TARGET_SAMPLE_RATE: Final[int] = 16000

##### VOICE RESOLUTION #####


def resolve_voice(provider: TTSProvider, lang: str) -> str:
    """Resolve voice ID or model path for a given language and provider."""
    match provider:
        case TTSProvider.EVOICE:
            if (voice := EVOICE_VOICE_MAP.get(lang)) is None:
                msg = f"No e-voice voice mapped for language '{lang}'"
                raise ValueError(msg)
            return voice
        case TTSProvider.PIPER:
            if (model := PIPER_MODEL_MAP.get(lang)) is None:
                msg = f"No Piper model mapped for language '{lang}'"
                raise ValueError(msg)
            return model


##### E-VOICE GENERATION #####


def _resample_to_16k(wav_bytes: bytes) -> bytes:
    """Resample WAV bytes to 16kHz 16-bit PCM mono if needed."""
    data, source_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

    if data.ndim > 1:
        data = data[:, 0]

    if source_sr != _TARGET_SAMPLE_RATE:
        n_samples = int(len(data) * _TARGET_SAMPLE_RATE / source_sr)
        data = scipy.signal.resample(data, n_samples)

    pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, _TARGET_SAMPLE_RATE, pcm)
    return buf.getvalue()


def generate_clips_evoice(
    texts: list[str],
    output_dir: Path,
    *,
    voice: str,
    url: str = "http://localhost:45140",
    max_samples: int | None = None,
) -> int:
    """Generate 16kHz mono WAV clips via e-voice HTTP API. Returns clip count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = max_samples or len(texts)
    count = 0

    with httpx.Client(base_url=url, timeout=30.0) as client:
        for text in texts:
            if count >= limit:
                break
            resp = client.post(
                "/v1/audio/speech",
                json={
                    "input": text,
                    "voice": voice,
                    "response_format": "wav",
                    "stream": False,
                },
            )
            resp.raise_for_status()

            wav_16k = _resample_to_16k(resp.content)
            filename = f"{uuid4().hex}.wav"
            (output_dir / filename).write_bytes(wav_16k)
            count += 1

            if count % 100 == 0:
                log.info("generate_progress", count=count, limit=limit, voice=voice)

    log.info("generate_complete", count=count, voice=voice, output_dir=str(output_dir))
    return count
