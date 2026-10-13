"""Command line entry point: ``parking-assistant serve``."""

import argparse
import logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parking-assistant")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="run the web application (default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    from parking_assistant.api import create_app

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    uvicorn.run(create_app(), host=host, port=port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
