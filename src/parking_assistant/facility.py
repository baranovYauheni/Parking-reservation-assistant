"""Default settings of the demo facility.

They seed the ``facility_settings`` SQL table on first start; the running
application always reads the current values from SQL.
"""

from decimal import Decimal

FACILITY_NAME: str = "Demo Central Parking"
TOTAL_SPACES: int = 10
HOURLY_RATE_GEL: Decimal = Decimal(5)
OPENING_HOURS: str = "24/7"
TIMEZONE_NAME: str = "Asia/Tbilisi"
