from decimal import Decimal

import pytest

from parking_assistant import facility


def test_settings_are_seeded_from_facility_defaults(db):
    settings = db.get_settings()
    assert settings.total_spaces == facility.TOTAL_SPACES
    assert settings.hourly_rate_gel == Decimal(5)
    assert settings.timezone == "Asia/Tbilisi"


def test_set_setting_updates_value_and_rejects_unknown_keys(db):
    db.set_setting("hourly_rate_gel", "6.50")
    assert db.get_settings().hourly_rate_gel == Decimal("6.50")
    with pytest.raises(KeyError):
        db.set_setting("admin_password", "x")


def test_init_is_idempotent_and_keeps_changes(db):
    db.set_setting("total_spaces", "4")
    db.init()
    assert db.get_settings().total_spaces == 4


def test_transaction_rolls_back_on_error(db):
    with pytest.raises(RuntimeError), db.transaction() as conn:
        conn.execute(
            "UPDATE facility_settings SET value = '1' WHERE key = 'total_spaces'"
        )
        raise RuntimeError("boom")
    assert db.get_settings().total_spaces == facility.TOTAL_SPACES
