FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

WORKDIR /app

# Install from local source (the package is not published to PyPI). Copy only
# what the build needs so the image rebuilds cleanly without the venv/test tree.
#
# Dependencies come from uv.lock, frozen, never from a fresh resolution: a
# fresh `uv pip install .` carries whatever is newest at build time, so the
# suites test one set of versions, the server runs another, and dependency
# alerts grade the lock rather than the image. The project installs last,
# --no-deps.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.txt \
 && uv pip install --system -r /tmp/requirements.txt \
 && uv pip install --system --no-deps . \
 && rm /tmp/requirements.txt

EXPOSE 8000

CMD ["kenning-encounter"]
