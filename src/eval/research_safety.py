"""Minimal public artifact projection: URLs never enter new research text artifacts."""
import re


def public_text(value):
    return re.sub(r"https?://[^\s<>\"']+", "[source URL omitted]", str(value))


def public_value(value):
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        return public_text(value)
    return value
