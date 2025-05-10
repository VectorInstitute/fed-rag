# Stage 1: Python environment with your application
FROM python:3.13-slim-bookworm AS python-base

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

################################################################################
# Stage 2: Final image combining Qdrant with Python application
FROM qdrant/qdrant:latest

USER root

# Install necessary runtime libraries and build tools
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libsqlite3-0 \
    libgssapi-krb5-2 \
    libssl3 \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libonig-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust compiler
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy Python from the first stage
COPY --from=python-base /usr/local/bin/python* /usr/local/bin/
COPY --from=python-base /usr/local/bin/pip* /usr/local/bin/
COPY --from=python-base /bin/uv /bin/uv
COPY --from=python-base /bin/uvx /bin/uvx
COPY --from=python-base /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=python-base /usr/local/lib/libpython3.13* /usr/local/lib/
COPY --from=python-base /usr/local/include/python3.13 /usr/local/include/python3.13

# Set Python environment variables
ENV PYTHONPATH=/usr/local/lib/python3.13/site-packages
ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy your installed application
COPY --from=python-base /app /app
WORKDIR /app

# Copy Docker-specific scripts
COPY docker/startup.sh .
COPY docker/healthcheck.sh .
RUN chmod +x startup.sh healthcheck.sh

# Create a folder for Atlas data
RUN mkdir -p /data/atlas

# Set the default model parameters
ENV MODEL_NAME=""
ENV QUERY_MODEL_NAME="nthakur/dragon-plus-query-encoder"
ENV CONTEXT_MODEL_NAME="nthakur/dragon-plus-context-encoder"
ENV BATCH_SIZE=5000
ENV CLEAR_FIRST=True
ENV FILENAME="text-list-100-sec.jsonl"

# for startup
ENV SKIP_DOWNLOAD="false"
ENV TINY_SAMPLE="false"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 CMD ["./healthcheck.sh"]


# Set the entrypoint to our custom startup script
ENTRYPOINT ["./startup.sh"]

# Inherit the CMD from the Qdrant image which will be passed to startup.sh
