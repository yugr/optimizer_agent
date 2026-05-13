# ============================================================
# optimizer_agent – Docker image
# https://github.com/yugr/optimizer_agent
#
# Build:
#   docker build -t optimizer_agent .
#
# Run:
#   docker run --rm -i \
#     -e ANTHROPIC_API_KEY="sk-..." \
#     [-e ANTHROPIC_AUTH_TOKEN="..."] \
#     [-e ANTHROPIC_BASE_URL="https://..."] \
#     -v $PWD:/work
#     optimizer_agent -m sonnet --max-trials 10 -v < kernel.c
#
# Volumes:
#   /work
# ============================================================

FROM debian:bookworm-slim

LABEL maintainer="optimizer_agent image" \
      description="LLM-based C optimizer agent with gem5/ARM cross-build support"

ARG DEBIAN_FRONTEND=noninteractive

# Standard development tools

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ clang lld python3 \
    make cmake scons ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Cross-compilation tools for AArch64

RUN apt-get update && apt-get install -y --no-install-recommends \
    qemu-user \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    libc6-arm64-cross \
    libc6-dev-arm64-cross \
 && rm -rf /var/lib/apt/lists/*

# gem5 dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential m4 zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev \
    libgoogle-perftools-dev \
    python3-dev python3-six python3-pydot \
    libboost-all-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Python pip (for the Anthropic SDK) and misc utilities

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Anthropic Python SDK

RUN pip install --no-cache-dir --break-system-packages anthropic

# Build gem5 (~6 CPU-hours, do this before agent clone so agent changes do not invalidate gem5 layer)

ENV GEM5_PATH=/opt/gem5

RUN git clone --depth 1 https://github.com/gem5/gem5.git ${GEM5_PATH} \
 && cd ${GEM5_PATH} \
 && scons build/ARM/gem5.opt -j$(nproc) < /dev/null \
 && find ${GEM5_PATH} -name '*.o' -o -name '*.a' -delete

# Dedicated user

RUN useradd --create-home --shell /bin/bash agent \
 && mkdir -p /work \
 && chown agent:agent /work

VOLUME /work

USER agent

WORKDIR /home/agent

# Clone agent code

RUN git clone --depth 1 https://github.com/yugr/optimizer_agent.git \
    /home/agent/optimizer_agent

WORKDIR /home/agent/optimizer_agent

# Runtime environment (GEM5_PATH needed by optimizer_agent.py)

ENV GEM5_PATH=/opt/gem5

# Anthropic credentials, to be supplied at `docker run` time via -e flags.

ENV ANTHROPIC_API_KEY="" \
    ANTHROPIC_AUTH_TOKEN="" \
    ANTHROPIC_BASE_URL=""

ENTRYPOINT ["python3", "optimizer_agent.py"]
