from parking_assistant.booking_dialog import (
    handle_booking_message,
    new_booking_state,
    start_booking,
    wants_booking,
    wants_status,
)


def run(service, *messages):
    step = start_booking(new_booking_state(), messages[0], service)
    for text in messages[1:]:
        step = handle_booking_message(step.state, text, service)
    return step


def test_collects_fields_step_by_step_and_shows_summary(service):
    step = run(
        service,
        "I want to book",
        "John",
        "Smith",
        "ab-123 cd",
        "2026-10-20 10:00",
        "2026-10-20 12:30",
    )
    assert step.state["mode"] == "confirming"
    assert "Estimated price: GEL 15.00" in step.reply
    assert "AB-123 CD" in step.reply


def test_several_fields_in_one_message_and_full_name(service):
    step = run(
        service,
        "book please",
        "name: Anna Maria Lee, car: XY999ZZ, from: 2026-10-20 09:00, to: 2026-10-20 10:00",
    )
    assert step.state["draft"]["first_name"] == "Anna"
    assert step.state["draft"]["last_name"] == "Maria Lee"
    assert step.state["mode"] == "confirming"


def test_invalid_value_is_explained_and_field_asked_again(service):
    step = run(service, "book", "John Smith", "A!")
    assert "Car number must contain" in step.reply
    assert step.state["awaiting"] == "car_number"
    assert "first_name" in step.state["draft"]


def test_end_before_start_is_rejected(service):
    step = run(
        service, "book", "John Smith", "AB123CD", "2026-10-20 12:00", "2026-10-20 11:00"
    )
    assert "end must be after the start" in step.reply
    assert step.state["awaiting"] == "end"


def test_from_to_phrase_and_edit_then_confirm(service):
    step = run(
        service,
        "I'd like to book from 2026-10-20 10:00 to 2026-10-20 12:00",
        "John Smith",
        "AB123CD",
        "edit end 2026-10-20 13:00",
    )
    assert step.state["mode"] == "confirming"
    step = handle_booking_message(step.state, "confirm", service)
    assert step.submit is not None
    assert (step.submit.end - step.submit.start).total_seconds() == 3 * 3600


def test_cancel_discards_draft(service):
    step = run(service, "book", "John", "cancel")
    assert step.state == new_booking_state()
    assert step.submit is None


def test_intent_detection():
    assert wants_booking("I want to reserve a space")
    assert not wants_booking("what is the status of my reservation?")
    assert wants_status("status of my booking")
    assert not wants_status("I want to make a new reservation")
