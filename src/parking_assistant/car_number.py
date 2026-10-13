MIN_CAR_LEN_NUMBER: int = 4
MAX_CAR_LEN_NUMBER: int = 10


def normalize_car_number(raw: str) -> str:
    return raw.strip().replace(" ", "").replace("-", "").upper()


def is_valid_car_number(car_number: str) -> bool:
    return (
        MIN_CAR_LEN_NUMBER <= len(car_number) <= MAX_CAR_LEN_NUMBER
        and car_number.isascii()
        and car_number.isalnum()
    )
