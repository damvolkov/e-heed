"""Adversarial text generation — phonemically similar words for negative training.

Adapted from openwakeword.data.generate_adversarial_texts.
Pure text logic — zero torch/audio dependencies. Only requires `pronouncing` (CMUdict).
"""

from __future__ import annotations

import itertools
import random
import re

import pronouncing

##### CONSTANTS #####

_VOWEL_PHONES = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AX",
        "AXR",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IX",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
        "UX",
    }
)

_VOWEL_PATTERN = "|".join(sorted(_VOWEL_PHONES))


##### PUBLIC #####


def generate_adversarial_texts(
    input_text: str,
    n: int,
    *,
    include_partial_phrase: float = 0.0,
    include_input_words: float = 0.0,
) -> list[str]:
    """Generate phonemically similar but non-matching words/phrases."""
    words = input_text.split()
    word_phones_list = [pronouncing.phones_for_word(w) for w in words]

    resolved_phones: list[str] = []
    for phones, _word in zip(word_phones_list, words, strict=True):
        if phones:
            resolved_phones.append(phones[0])
        else:
            resolved_phones.append("")

    # Add all possible lexical stresses to vowels
    stressed = [
        re.sub(_VOWEL_PATTERN, lambda m: m.group(0) + "[0|1|2]", re.sub(r"\d+", "", p)) for p in resolved_phones
    ]

    # Build adversarial word lists per position
    adversarial_per_pos: list[list[str]] = []
    for phones_str, word in zip(stressed, words, strict=True):
        phone_parts = phones_str.split()
        queries = (
            [" ".join(phone_parts)]
            if len(phone_parts) <= 2
            else _phoneme_replacement(phone_parts, max_replace=max(0, len(phone_parts) - 2))
        )

        candidates: list[str] = []
        for query in queries:
            matches = pronouncing.search(query)
            for match in matches:
                match_phones = pronouncing.phones_for_word(match)
                if match_phones and match_phones[0] != phones_str and match.lower() != word.lower():
                    candidates.append(match)

        if candidates:
            adversarial_per_pos.append(candidates)

    if not adversarial_per_pos:
        return []

    # Build N combinations
    result: list[str] = []
    for _ in range(n):
        txts: list[str] = []
        for j, word in zip(adversarial_per_pos, words, strict=False):
            if random.random() <= include_input_words:
                txts.append(word)
            else:
                txts.append(random.choice(j))

        if len(words) > 1 and random.random() <= include_partial_phrase:
            n_words = random.randint(1, len(words))
            txts = random.sample(txts, n_words)

        result.append(" ".join(txts))

    return [t for t in result if t != input_text]


##### INTERNAL #####


def _phoneme_replacement(phones: list[str], max_replace: int, replace_char: str = "(.){1,3}") -> list[str]:
    """Generate regex queries with phoneme positions replaced by wildcards."""
    results: list[str] = []
    for r in range(1, max_replace + 1):
        for indices in itertools.combinations(range(len(phones)), r):
            copy = list(phones)
            for i in indices:
                copy[i] = replace_char
            results.append(" ".join(copy))
    return results
