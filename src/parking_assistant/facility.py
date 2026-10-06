from decimal import Decimal

FACILITY_NAME: str = "Demo Central Parking"
TOTAL_SPACES: int = 10
HOURLY_RATE_GEL: Decimal = Decimal(5)
OPENING_HOURS: str = "24/7"
TIMEZONE_NAME: str = "Asia/Tbilisi"


def format_facility_summary() -> str:
    full_day_rate = HOURLY_RATE_GEL * 24

    return (
        f"{FACILITY_NAME}\n"
        f"Spaces:          {TOTAL_SPACES}\n"
        f"Hourly rate:     GEL {HOURLY_RATE_GEL:.2f}\n"
        f"Full day (24 h): GEL {full_day_rate:.2f}\n"
        f"Opening hours:   {OPENING_HOURS}\n"
        f"Timezone:        {TIMEZONE_NAME}"
    )
