"""Guardrails: input screening, output redaction and indexing checks.

Pattern-based filters are a defence layer, not a guarantee. Privacy is mainly
enforced by architecture: only allow-listed public documents are indexed,
private data is read through owner-scoped SQL queries, and the model has no
tool that can read other users' data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 2000

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above) ",
    r"forget (all |your )?(previous )?(instructions|rules)",
    r"(reveal|show|print|repeat|leak)\b.{0,30}\b(system prompt|instructions|prompt)",
    r"\bsystem prompt\b",
    r"you are now\b",
    r"\bact as (an? )?(admin|administrator|developer|system)\b",
    r"\b(developer|jailbreak|dan) mode\b",
    r"\bapprove (my|this|the) (own )?(reservation|request|booking) (yourself|automatically)\b",
    r"\b(set|change|mark)\b.{0,30}\bstatus\b.{0,20}\bapproved\b",
    r"<\s*/?\s*(system|assistant)\s*>",
]

_PRIVATE_DATA_PATTERNS = [
    r"\b(other|another|all|every|someone else'?s?|others'?)\b.{0,30}"
    r"\b(users?|customers?|people|persons?|clients?|drivers?)\b.{0,40}"
    r"\b(reservations?|bookings?|requests?|names?|car numbers?|plates?|data|details)\b",
    r"\b(list|show|give|export|dump|print)\b(?!.*\bmy\b).{0,20}\b(all|every)\b.{0,20}"
    r"\b(reservations?|bookings?|requests?|customers?|users?)\b",
    r"\b(other|another)\s+(users?|customers?|people|clients?|drivers?|visitors?)\b"
    r".*\b(book\w*|reserv\w*|cars?|plates?|names?|data|details|park\w*)\b",
    r"\b(book\w*|reserv\w*|cars?|plates?|names?|data|details)\b.*"
    r"\b(other|another)\s+(users?|customers?|people|clients?|drivers?|visitors?)\b",
    r"\bwho (has )?(booked|reserved|parked)\b",
    r"\b(whose|which person'?s?|owner of)\b.{0,30}\b(car|plate|vehicle)\b",
    r"\b(admin|administrator|api)\s*(token|password|key|credentials?)\b",
    r"\b(openai|secret)\s*key\b",
    r"\bselect\s+(\*|\w+(\s*,\s*\w+)*)\s+from\s+\w+",
    r"\b(car number|plate)\s+(of|belonging to|owned by)\b(?!\s+(my|me|mine)\b)",
    r"\b(drop\s+table|delete\s+from|insert\s+into|union\s+select)\b",
    r"\bupdate\s+\w+\s+set\b",
    r"\b(database|db)\b.{0,20}\b(dump|contents|password|credentials)\b",
    r"\.env\b",
]

_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_PRIVATE = [re.compile(p, re.IGNORECASE) for p in _PRIVATE_DATA_PATTERNS]

_SENSITIVE_OUTPUT = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<![\w-])\+?\d[\d\s().-]{7,}\d(?![\w-])"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),
    "bearer": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "password": re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE),
}

# Datetimes such as 2026-10-20T10:00:00+04:00 must not be mistaken for phones.
_ISO_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)?"
)

REFUSAL_PRIVATE = (
    "I can't share information about other customers, internal credentials or "
    "database contents. I can help with public parking information or with your "
    "own reservation in this chat."
)
REFUSAL_INJECTION = (
    "I can't follow instructions that try to change my rules or permissions. "
    "I can answer questions about the parking or help you make a reservation."
)
REFUSAL_TOO_LONG = (
    f"Your message is too long. Please keep it under {MAX_INPUT_CHARS} characters."
)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    category: str = "ok"
    message: str = ""


def check_user_input(text: str) -> GuardResult:
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(False, "too_long", REFUSAL_TOO_LONG)
    if any(p.search(text) for p in _INJECTION):
        return GuardResult(False, "prompt_injection", REFUSAL_INJECTION)
    if any(p.search(text) for p in _PRIVATE):
        return GuardResult(False, "private_data", REFUSAL_PRIVATE)
    return GuardResult(True)


def find_sensitive(text: str) -> list[str]:
    """Return categories of sensitive data found in ``text``."""
    scrubbed = _ISO_DATETIME.sub(" ", text)
    return [
        name for name, pattern in _SENSITIVE_OUTPUT.items() if pattern.search(scrubbed)
    ]


def redact(text: str, allowed: tuple[str, ...] = ()) -> str:
    """Mask sensitive values in model output; ``allowed`` values are kept."""
    protected: dict[str, str] = {}

    def placeholder(value: str) -> str:
        key = f"\x00{len(protected)}\x00"
        protected[key] = value
        return key

    result = _ISO_DATETIME.sub(lambda m: placeholder(m.group(0)), text)
    for value in allowed:
        if value and value in result:
            result = result.replace(value, placeholder(value))
    for name, pattern in _SENSITIVE_OUTPUT.items():
        result = pattern.sub(f"[redacted {name}]", result)
    for key, value in protected.items():
        result = result.replace(key, value)
    return result
