# Stage 1: Python environment with your application
FROM python:3.12-slim-bookworm AS python-base

# Install uv properly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Git and build essentials for any potential dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Copy and install project requirements
COPY pyproject.toml .
COPY src/ ./src/
COPY README.md .

# Remove the sources section from pyproject.toml for Docker build
RUN sed -i '/\[tool.uv.sources\]/,/\[/d' pyproject.toml

# Install project in development mode using uv
RUN uv sync
