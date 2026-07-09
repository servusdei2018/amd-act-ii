# Use the official ROCm Ubuntu 22.04 base image matching the uv.lock version (ROCm 7.2.3)
FROM docker.io/rocm/dev-ubuntu-22.04:7.2.3

# Set shell and noninteractive variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for ROCm / vLLM compatibility
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV VLLM_USE_MODELSCOPE=False
ENV CONTAINER_PREBUILT=true
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    libnuma-dev \
    libopenmpi-dev \
    openmpi-bin \
    rocm-hip-sdk \
    rocm-ml-libraries \
    rocm-smi-lib \
    procps \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package and project manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Synchronize the virtual environment (installs python 3.12 and lockfile dependencies)
RUN uv sync --frozen

# Copy the rest of the application code
COPY . .

# Ensure scripts are executable
RUN chmod +x /app/run_server.sh

# Expose ports
# 8001: Triduum Gateway (FastAPI router)
# 8000: Internal vLLM server
EXPOSE 8001
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/run_server.sh"]
