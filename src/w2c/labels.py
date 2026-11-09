"""Canonical four-way When2Call decision labels.

Source labels can arrive as letters (A-D), canonical names, or loose aliases
from the SFT response format. Anything unrecognised or explicitly ambiguous is
mapped to ``None`` and must be adjudicated or excluded -- never guessed.
"""

from __future__ import annotations

CANONICAL = ("direct", "tool_call", "request_for_info", "cannot_answer")
LABEL_TO_ID = {name: i for i, name in enumerate(CANONICAL)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

ALIASES = {
    # letters used in the plan
    "a": "direct",
    "b": "tool_call",
    "c": "request_for_info",
    "d": "cannot_answer",
    # canonical
    "direct": "direct",
    "tool_call": "tool_call",
    "toolcall": "tool_call",
    "request_for_info": "request_for_info",
    "request_info": "request_for_info",
    "cannot_answer": "cannot_answer",
    # audited aliases only; extend deliberately, and log every mapping
    "answer": "direct",
    "no_tool": "direct",
    "call_tool": "tool_call",
    "invoke_tool": "tool_call",
    "ask_for_info": "request_for_info",
    "clarify": "request_for_info",
    "unanswerable": "cannot_answer",
    "refuse": "cannot_answer",
}

# Explicit non-labels. Kept separate from typos: these are expected and dropped.
AMBIGUOUS = {
    "", "?", "n/a", "na", "none", "null", "unknown", "ambiguous",
    "unclear", "mixed", "other",
}


def normalize_label(value) -> str | None:
    """Return a canonical label name, or None if the value is not usable."""
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in AMBIGUOUS:
        return None
    return ALIASES.get(key)


def label_id(value) -> int | None:
    name = normalize_label(value)
    return None if name is None else LABEL_TO_ID[name]
