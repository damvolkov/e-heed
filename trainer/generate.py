"""Phase 1: Generate synthetic audio clips of the wakeword via e-voice."""

from __future__ import annotations

from pathlib import Path

import structlog
from trainer.adversarial import generate_adversarial_texts
from trainer.config import wakeword_dir
from trainer.spec import TrainSpec, TTSProvider
from trainer.tts import generate_clips_evoice, resolve_voice

log = structlog.get_logger("trainer.generate")


def generate(spec: TrainSpec, word: str) -> None:
    """Generate positive and adversarial negative clips.

    Only runs for TTSProvider.EVOICE — Piper generation is delegated
    to openwakeword subprocess (--generate_clips) in pipeline.py.
    """
    if spec.tts != TTSProvider.EVOICE:
        return

    wd = wakeword_dir(word)
    model_dir = wd / spec.model_name

    _generate_positive(spec, model_dir)
    _generate_adversarial(spec, model_dir)


##### POSITIVE CLIPS #####


def _generate_positive(spec: TrainSpec, model_dir: Path) -> None:
    """Generate positive clips (target phrase) across all languages."""
    for subdir, n_samples in [("positive_train", spec.n_samples), ("positive_test", spec.n_samples_val)]:
        out = model_dir / subdir
        existing = len(list(out.glob("*.wav"))) if out.exists() else 0

        if existing >= n_samples:
            log.info("skip_existing", subdir=subdir, existing=existing)
            continue

        remaining = n_samples - existing
        per_lang = max(1, remaining // len(spec.langs))

        for lang in spec.langs:
            voice = resolve_voice(TTSProvider.EVOICE, lang)
            texts = [phrase for phrase in spec.target_phrase for _ in range(per_lang)]
            log.info("generate_positive", subdir=subdir, lang=lang, voice=voice, count=per_lang)
            generate_clips_evoice(texts, out, voice=voice, url=spec.evoice_url, max_samples=per_lang)


##### ADVERSARIAL NEGATIVE CLIPS #####


def _generate_adversarial(spec: TrainSpec, model_dir: Path) -> None:
    """Generate adversarial negative clips (phonemically similar words)."""
    texts = list(spec.custom_negative_phrases)
    for phrase in spec.target_phrase:
        texts.extend(
            generate_adversarial_texts(
                phrase,
                n=spec.n_samples // max(1, len(spec.target_phrase)),
                include_partial_phrase=1.0,
                include_input_words=0.2,
            )
        )

    if not texts:
        log.warning("no_adversarial_texts", word=spec.model_name)
        return

    for subdir, n_samples in [("negative_train", spec.n_samples), ("negative_test", spec.n_samples_val)]:
        out = model_dir / subdir
        existing = len(list(out.glob("*.wav"))) if out.exists() else 0

        if existing >= n_samples:
            log.info("skip_existing", subdir=subdir, existing=existing)
            continue

        remaining = n_samples - existing
        per_lang = max(1, remaining // len(spec.langs))

        for lang in spec.langs:
            voice = resolve_voice(TTSProvider.EVOICE, lang)
            log.info("generate_adversarial", subdir=subdir, lang=lang, voice=voice, count=per_lang)
            generate_clips_evoice(texts, out, voice=voice, url=spec.evoice_url, max_samples=per_lang)
