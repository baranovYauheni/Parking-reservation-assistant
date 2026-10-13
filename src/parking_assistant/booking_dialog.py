"""Deterministic slot-filling dialog that collects reservation details.

The dialog state is a plain JSON-serialisable dict, so it can be persisted
between turns. Fields already supplied are kept until the user confirms or cancels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from parking_assistant.reservations import (
    ReservationRequest,
    ReservationService,
    validate_car_number,
    validate_name,
    validate_period,
)
from parking_assistant.timeutils import (
    DateParseError,
    from_utc_text,
    human_local,
    parse_user_datetime,
    to_utc_text,
)

FIELDS: tuple[str, ...] = ("first_name", "last_name", "car_number", "start", "end")

LABELS = {
    "first_name": "first name",
    "last_name": "last name",
    "car_number": "car number",
    "start": "start",
    "end": "end",
}

QUESTIONS = {
    "first_name": "What is your first name?",
    "last_name": "What is your last name?",
    "car_number": "What is your car number (for example AB123CD)?",
    "start": "When should the reservation start? Use YYYY-MM-DD HH:MM in Tbilisi "
    "time (for example 2026-10-20 09:00) or 'tomorrow 09:00'.",
    "end": "When should it end? (YYYY-MM-DD HH:MM)",
}

_ALIASES = {
    "first name": "first_name",
    "firstname": "first_name",
    "given name": "first_name",
    "last name": "last_name",
    "lastname": "last_name",
    "surname": "last_name",
    "family name": "last_name",
    "full name": "name",
    "name": "name",
    "car number": "car_number",
    "license plate": "car_number",
    "plate number": "car_number",
    "plate": "car_number",
    "registration": "car_number",
    "car": "car_number",
    "start": "start",
    "from": "start",
    "arrival": "start",
    "end": "end",
    "until": "end",
    "till": "end",
    "to": "end",
    "departure": "end",
}
_KEY_ALT = "|".join(sorted((re.escape(k) for k in _ALIASES), key=len, reverse=True))
_PAIR = re.compile(rf"\b({_KEY_ALT})\s*[:=]\s*([^,;\n]+)", re.IGNORECASE)
_FROM_TO = re.compile(r"\bfrom\s+(.+?)\s+(?:to|until|till)\s+(.+?)\s*$", re.IGNORECASE)
_EDIT = re.compile(
    r"^\s*(?:edit|change|update|fix|correct)\s+(?:the\s+|my\s+)?"
    r"(first name|last name|surname|name|car number|car|plate|start|end|period|dates?)",
    re.IGNORECASE,
)
_CANCEL = re.compile(r"^\s*(cancel|stop|abort|never ?mind|discard)\b", re.IGNORECASE)
_CONFIRM = re.compile(
    r"^\s*(yes|y|yep|confirm|confirmed|correct|submit|send( it)?|ok|okay|sure)\b[.! ]*$",
    re.IGNORECASE,
)
_DECLINE = re.compile(r"^\s*(no|n|nope|not correct|wrong)\b", re.IGNORECASE)

_BOOKING_INTENT = re.compile(
    r"\b(book|booking|reserve|reservation|make a reservation|"
    r"бронь|брониров\w*|забронир\w*)\b",
    re.IGNORECASE,
)
_STATUS_INTENT = re.compile(
    r"\b(status|my (reservation|booking|request)s?|approved|rejected|"
    r"was it (approved|rejected))\b",
    re.IGNORECASE,
)


def new_booking_state() -> dict:
    return {"mode": "idle", "draft": {}, "awaiting": None}


def wants_booking(text: str) -> bool:
    return bool(_BOOKING_INTENT.search(text)) and not wants_status(text)


def wants_status(text: str) -> bool:
    return bool(_STATUS_INTENT.search(text)) and not re.search(
        r"\b(make|new|want to|would like to|i'd like to)\b", text, re.IGNORECASE
    )


def is_question(text: str) -> bool:
    return text.strip().endswith("?")


@dataclass
class BookingStep:
    reply: str
    state: dict
    submit: ReservationRequest | None = None
    errors: list[str] = field(default_factory=list)


def missing(draft: dict) -> list[str]:
    return [f for f in FIELDS if not draft.get(f)]


def _apply(draft: dict, key: str, value: str, now: datetime) -> str | None:
    """Validate and store one field. Returns an error message or ``None``."""
    value = value.strip().strip(".")
    try:
        if key == "name":
            parts = value.split()
            if len(parts) < 2:
                return "Please give both first and last name."
            draft["first_name"] = validate_name(parts[0], "First name")
            draft["last_name"] = validate_name(" ".join(parts[1:]), "Last name")
        elif key in ("first_name", "last_name"):
            draft[key] = validate_name(value, LABELS[key].capitalize())
        elif key == "car_number":
            draft[key] = validate_car_number(value)[0]
        elif key in ("start", "end"):
            draft[key] = to_utc_text(parse_user_datetime(value, now))
    except (ValueError, DateParseError) as exc:
        return str(exc)
    return None


def _check_period(draft: dict, now: datetime) -> str | None:
    if not (draft.get("start") and draft.get("end")):
        return None
    start, end = from_utc_text(draft["start"]), from_utc_text(draft["end"])
    try:
        validate_period(start, end, now)
    except ValueError as exc:
        message = str(exc)
        if "start" in message.lower() and "future" in message.lower():
            draft.pop("start", None)
        else:
            draft.pop("end", None)
        return message
    return None


def summary(draft: dict, service: ReservationService) -> str:
    start, end = from_utc_text(draft["start"]), from_utc_text(draft["end"])
    free = service.free_spaces(start, end)
    price = service.quote(start, end)
    availability = (
        f"{free} space(s) currently free for this period"
        if free > 0
        else "no spaces are currently free for this period - the administrator "
        "may reject the request"
    )
    return (
        "Please check your reservation request:\n"
        f"- Name: {draft['first_name']} {draft['last_name']}\n"
        f"- Car number: {draft['car_number']}\n"
        f"- From: {human_local(start)}\n"
        f"- To: {human_local(end)}\n"
        f"- Estimated price: GEL {price:.2f}\n"
        f"- Availability: {availability}\n"
        "Reply 'confirm' to send it to the administrator, 'edit <field>' to change "
        "something (e.g. 'edit car number'), or 'cancel'."
    )


def _next_prompt(state: dict, service: ReservationService) -> str:
    draft = state["draft"]
    todo = missing(draft)
    if todo:
        state["mode"] = "collecting"
        state["awaiting"] = todo[0]
        return QUESTIONS[todo[0]]
    state["mode"] = "confirming"
    state["awaiting"] = "confirmation"
    return summary(draft, service)


def current_prompt(state: dict, service: ReservationService) -> str:
    return _next_prompt(state, service)


def start_booking(state: dict, text: str, service: ReservationService) -> BookingStep:
    state = _copy(state)
    now = service.clock()
    state["mode"] = "collecting"
    errors = _extract_pairs(state["draft"], text, now)
    period_error = _check_period(state["draft"], now)
    if period_error:
        errors.append(period_error)
    intro = "Let's make a reservation. "
    if state["draft"] and not errors:
        intro += "I've kept the details you already gave me. "
    reply = intro + _format_errors(errors) + _next_prompt(state, service)
    return BookingStep(reply, state, errors=errors)


def handle_booking_message(
    state: dict, text: str, service: ReservationService
) -> BookingStep:
    state = _copy(state)
    now = service.clock()
    draft = state["draft"]

    if _CANCEL.match(text):
        return BookingStep(
            "The reservation draft was cancelled. Ask me anything or say 'book' to "
            "start again.",
            new_booking_state(),
        )

    edit = _EDIT.match(text)
    if edit:
        target = edit.group(1).lower()
        keys = {
            "first name": ["first_name"],
            "last name": ["last_name"],
            "surname": ["last_name"],
            "name": ["first_name", "last_name"],
            "car number": ["car_number"],
            "car": ["car_number"],
            "plate": ["car_number"],
            "start": ["start"],
            "end": ["end"],
            "period": ["start", "end"],
            "date": ["start", "end"],
            "dates": ["start", "end"],
        }[target]
        rest = re.sub(r"^\s*(?:to\b|[:=])?\s*", "", text[edit.end() :]).strip()
        for key in keys:
            draft.pop(key, None)
        errors: list[str] = []
        if rest and len(keys) == 1:
            error = _apply(draft, keys[0], rest, now)
            if error:
                errors.append(error)
        period_error = _check_period(draft, now)
        if period_error:
            errors.append(period_error)
        return BookingStep(
            _format_errors(errors) + _next_prompt(state, service), state, errors=errors
        )

    if state["mode"] == "confirming":
        if _CONFIRM.match(text):
            period_error = _check_period(draft, now)
            if period_error:
                return BookingStep(
                    _format_errors([period_error]) + _next_prompt(state, service),
                    state,
                    errors=[period_error],
                )
            request = ReservationRequest(
                first_name=draft["first_name"],
                last_name=draft["last_name"],
                car_number=draft["car_number"],
                start=from_utc_text(draft["start"]),
                end=from_utc_text(draft["end"]),
            )
            return BookingStep("", state, submit=request)
        if _DECLINE.match(text):
            return BookingStep(
                "What would you like to change? Write for example 'edit start' or "
                "'edit car number', or 'cancel'.",
                state,
            )

    errors = _extract_pairs(draft, text, now)
    if not errors and not _PAIR.search(text) and not _FROM_TO.search(text):
        awaiting = state.get("awaiting")
        if awaiting in FIELDS:
            key = awaiting
            if (
                awaiting == "first_name"
                and len(text.split()) >= 2
                and not draft.get("last_name")
            ):
                key = "name"
            error = _apply(draft, key, text, now)
            if error:
                errors.append(error)
        elif state["mode"] == "confirming":
            errors.append("Please reply 'confirm', 'edit <field>' or 'cancel'.")
    period_error = _check_period(draft, now)
    if period_error:
        errors.append(period_error)
    return BookingStep(
        _format_errors(errors) + _next_prompt(state, service), state, errors=errors
    )


def _extract_pairs(draft: dict, text: str, now: datetime) -> list[str]:
    errors = []
    from_to = _FROM_TO.search(text)
    if from_to and not _PAIR.search(text):
        for key, value in (("start", from_to.group(1)), ("end", from_to.group(2))):
            error = _apply(draft, key, value, now)
            if error:
                errors.append(error)
        return errors
    for raw_key, value in _PAIR.findall(text):
        error = _apply(draft, _ALIASES[raw_key.lower()], value, now)
        if error:
            errors.append(error)
    return errors


def _format_errors(errors: list[str]) -> str:
    return "".join(f"{e} " for e in errors)


def _copy(state: dict) -> dict:
    return {
        "mode": state.get("mode", "idle"),
        "draft": dict(state.get("draft", {})),
        "awaiting": state.get("awaiting"),
    }
