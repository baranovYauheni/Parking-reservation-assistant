import pytest

from parking_assistant.car_number import is_valid_car_number, normalize_car_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" ab-123 cd ", "AB123CD"), ("xy999zz", "XY999ZZ"), ("   ", "")],
)
def test_normalize_removes_spaces_hyphens_and_uppercases(raw, expected):
    assert normalize_car_number(raw) == expected


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("AB123CD", True),
        ("ABC", False),
        ("A" * 11, False),
        ("AB12€", False),
        ("AB_123", False),
    ],
)
def test_is_valid_car_number_checks_length_and_characters(value, valid):
    assert is_valid_car_number(value) is valid
