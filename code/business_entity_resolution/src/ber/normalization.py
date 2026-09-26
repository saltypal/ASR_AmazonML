from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd


LEGAL_SUFFIXES = frozenset(
    {
        "co", "company", "corp", "corporation", "inc", "incorporated", "llc", "llp",
        "limited", "ltd", "private", "pvt", "plc", "pc", "foundation", "trust",
    }
)
NUMBER_PATTERN = re.compile(r"\d+")
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

SCRIPT_PREFIXES = (
    "LATIN", "DEVANAGARI", "BENGALI", "GUJARATI", "GURMUKHI", "KANNADA",
    "MALAYALAM", "TAMIL", "TELUGU", "ARABIC", "CYRILLIC", "GREEK", "HEBREW",
)


def normalize_basic(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def normalize_punctuation(value: object) -> str:
    text = normalize_basic(value)
    characters = (
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in text
    )
    return " ".join("".join(characters).split())


def fold_latin_accents(value: object) -> str:
    text = normalize_basic(value)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )


def tokenize(value: object) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(normalize_punctuation(value)))


def remove_legal_suffixes(value: object) -> str:
    tokens = list(tokenize(value))
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def extract_numbers(value: object) -> tuple[str, ...]:
    return tuple(NUMBER_PATTERN.findall(normalize_basic(value)))


def number_signature(value: object) -> str:
    return "|".join(extract_numbers(value))


def probable_postal_token(value: object) -> str:
    numbers = extract_numbers(value)
    long_numbers = [number for number in numbers if 5 <= len(number) <= 6]
    return long_numbers[-1] if long_numbers else ""


@lru_cache(maxsize=65_536)
def _script_for_character(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    for prefix in SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return "OTHER"


def scripts_in(value: object) -> tuple[str, ...]:
    text = "" if value is None else str(value)
    return tuple(sorted({script for char in text if (script := _script_for_character(char))}))


def script_signature(value: object) -> str:
    scripts = scripts_in(value)
    return "|".join(scripts) if scripts else "NONE"


def contains_non_latin(value: object) -> bool:
    scripts = scripts_in(value)
    return any(script != "LATIN" for script in scripts)


def prepare_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source frame is missing columns: {sorted(missing)}")
    prepared = frame.copy()
    for column in required:
        prepared[column] = prepared[column].fillna("").astype(str)
    prepared["name_norm"] = prepared["business_name"].map(normalize_basic)
    prepared["name_punct"] = prepared["business_name"].map(normalize_punctuation)
    prepared["name_ascii"] = prepared["business_name"].map(fold_latin_accents)
    prepared["name_core"] = prepared["business_name"].map(remove_legal_suffixes)
    prepared["address_norm"] = prepared["business_address"].map(normalize_basic)
    prepared["address_punct"] = prepared["business_address"].map(normalize_punctuation)
    prepared["address_numbers"] = prepared["business_address"].map(number_signature)
    prepared["postal_token"] = prepared["business_address"].map(probable_postal_token)
    prepared["name_script"] = prepared["business_name"].map(script_signature)
    prepared["address_script"] = prepared["business_address"].map(script_signature)
    prepared["has_non_latin_name"] = prepared["business_name"].map(contains_non_latin)
    prepared["country"] = prepared["country"].str.strip()
    return prepared


def token_jaccard(left: object, right: object) -> float:
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def number_jaccard(left: object, right: object) -> float:
    left_numbers, right_numbers = set(extract_numbers(left)), set(extract_numbers(right))
    if not left_numbers and not right_numbers:
        return 0.0
    union = left_numbers | right_numbers
    return len(left_numbers & right_numbers) / len(union) if union else 0.0
