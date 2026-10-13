from datetime import UTC, datetime

import pytest

from parking_assistant.timeutils import (
    DateParseError,
    from_utc_text,
    local_iso,
    parse_user_datetime,
    to_utc_text,
)

NOW = datetime(2026, 10, 13, 8, 0, tzinfo=UTC)  # 12:00 in Tbilisi


def test_local_input_is_interpreted_in_tbilisi_time():
    assert parse_user_datetime("2026-10-20 14:00") == datetime(
        2026, 10, 20, 10, 0, tzinfo=UTC
    )
    assert parse_user_datetime("20.10.2026 14:00") == datetime(
        2026, 10, 20, 10, 0, tzinfo=UTC
    )


def test_explicit_offset_is_respected():
    assert parse_user_datetime("2026-10-20T14:00+02:00") == datetime(
        2026, 10, 20, 12, 0, tzinfo=UTC
    )
    assert parse_user_datetime("2026-10-20T14:00Z") == datetime(
        2026, 10, 20, 14, 0, tzinfo=UTC
    )


def test_relative_tomorrow_uses_local_date():
    assert parse_user_datetime("tomorrow 09:30", NOW) == datetime(
        2026, 10, 14, 5, 30, tzinfo=UTC
    )


@pytest.mark.parametrize("bad", ["2026-10-20", "next week", "", "tomorrow 25:00"])
def test_ambiguous_or_invalid_input_is_rejected(bad):
    with pytest.raises(DateParseError):
        parse_user_datetime(bad, NOW)


def test_utc_round_trip_and_local_iso_has_offset():
    value = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    assert from_utc_text(to_utc_text(value)) == value
    assert local_iso(value) == "2026-10-20T14:00:00+04:00"
