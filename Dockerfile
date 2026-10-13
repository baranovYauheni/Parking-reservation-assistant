FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

RUN useradd --create-home app && mkdir -p /app/var && chown app /app/var
USER app

ENV PARKING_DATA_DIR=/app/var
EXPOSE 8000
CMD ["/app/.venv/bin/parking-assistant", "serve", "--host", "0.0.0.0", "--port", "8000"]
