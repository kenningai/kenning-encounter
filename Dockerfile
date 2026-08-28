FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install from local source (the package is not published to PyPI). Copy only
# what the build needs so the image rebuilds cleanly without the venv/test tree.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --system .

EXPOSE 8000

CMD ["kenning-encounter"]
