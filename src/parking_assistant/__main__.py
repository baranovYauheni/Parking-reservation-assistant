import sys


def main() -> int:
    print("Parking Reservation Assistant")
    print("Facility: Demo Central Parking")
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"Interpreter: {sys.executable}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
