# syntax=docker/dockerfile:1.7
ARG DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch

FROM ${DEEPSTREAM_IMAGE} AS runtime-base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    DS_CONFIG=/workspace/configs/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gir1.2-gst-plugins-base-1.0 \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-tools \
        libpq-dev \
        postgresql-client \
        procps \
        python3 \
        python3-dev \
        python3-gi \
        python3-gst-1.0 \
        python3-pip \
        tini \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-base AS bindings-builder

# NVIDIA's DeepStream 9.0 branch reports PyDS 1.2.3 but no longer publishes a
# wheel. Pin the exact upstream commit so a rebuild cannot silently change ABI.
ARG PYDS_SOURCE_REF=8ad0349ed7a496fae35ebb21c350641727070b89
ARG PYCUDA_VERSION=2025.1.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        cmake \
        g++ \
        git \
        libcairo2-dev \
        libglib2.0-dev \
        libgirepository-2.0-dev \
        libgstreamer1.0-dev \
        libtool \
        m4 \
        meson \
        ninja-build \
        python-gi-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global \
        url.https://github.com/GStreamer/gstreamer.git.insteadOf \
        https://gitlab.freedesktop.org/gstreamer/gstreamer.git \
    && for attempt in 1 2 3; do \
        rm -rf /tmp/deepstream_python_apps; \
        if mkdir -p /tmp/deepstream_python_apps \
            && cd /tmp/deepstream_python_apps \
            && git init \
            && git remote add origin \
                https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git \
            && git fetch --depth 1 origin "${PYDS_SOURCE_REF}" \
            && git checkout --detach FETCH_HEAD \
            && git submodule update --init --depth 1 --filter=blob:none \
                bindings/3rdparty/git-partial-submodule \
            && python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py clone \
                bindings/3rdparty/pybind11 bindings/3rdparty/gstreamer; then \
            exit 0; \
        fi; \
        echo "PyDS source checkout attempt ${attempt} failed; retrying" >&2; \
        sleep "$((attempt * 5))"; \
    done; \
    exit 1

# PyDS 1.2.3's pinned binding omits two fields that exist in the DS9 C ABI and
# are required to invert nvinfer ROI scaling for face landmarks. Apply a small,
# reviewed patch and fail the build if upstream no longer matches it exactly.
COPY docker/patches/pyds-tensor-meta-scaling.patch /tmp/pyds-tensor-meta-scaling.patch
RUN cd /tmp/deepstream_python_apps \
    && git apply --check /tmp/pyds-tensor-meta-scaling.patch \
    && git apply /tmp/pyds-tensor-meta-scaling.patch

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "build>=1,<2" "numpy==1.26.0" \
    && cd /tmp/deepstream_python_apps/bindings \
    && CMAKE_ARGS="-DDS_VERSION=9.0 -DDS_PATH=/opt/nvidia/deepstream/deepstream" \
       CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
       python3 -m build --wheel \
    && mkdir -p /tmp/wheels \
    && cp dist/pyds-1.2.3-*.whl /tmp/wheels/ \
    && python3 -m pip wheel --no-cache-dir --no-deps --no-build-isolation \
        --wheel-dir /tmp/wheels \
        "pycuda==${PYCUDA_VERSION}"

COPY native/yolo_parser /tmp/yolo-parser
RUN make -C /tmp/yolo-parser
COPY native/scrfd_parser /tmp/scrfd-parser
RUN make -C /tmp/scrfd-parser

FROM runtime-base AS final

ARG PYDS_SOURCE_REF=8ad0349ed7a496fae35ebb21c350641727070b89
ARG PYCUDA_VERSION=2025.1.2
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

LABEL org.opencontainers.image.title="DeepStream AI Platform" \
      org.opencontainers.image.description="Configuration-driven DeepStream industrial video analytics runtime" \
      org.opencontainers.image.vendor="Local project" \
      org.opencontainers.image.deepstream.version="9.0" \
      org.opencontainers.image.pyds.version="1.2.3" \
      org.opencontainers.image.pyds.revision="${PYDS_SOURCE_REF}" \
      org.opencontainers.image.pycuda.version="${PYCUDA_VERSION}"

COPY --from=bindings-builder /tmp/wheels /tmp/wheels
COPY --from=bindings-builder /tmp/yolo-parser/libnvdsinfer_custom_yolo_dynamic.so \
    /opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo_dynamic.so
COPY --from=bindings-builder /tmp/scrfd-parser/libnvdsinfer_custom_scrfd.so \
    /opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_scrfd.so
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "numpy==1.26.0" \
        "debugpy>=1.8,<2" \
        /tmp/wheels/pyds-1.2.3-*.whl \
        /tmp/wheels/pycuda-*.whl \
    && rm -rf /tmp/wheels \
    && python3 -m pip check \
    && python3 -c "import importlib.metadata as m, gi, numpy; gi.require_version('Gst', '1.0'); assert numpy.__version__ == '1.26.0'; assert m.version('pyds') == '1.2.3'; assert m.version('pycuda') == '${PYCUDA_VERSION}'"

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --break-system-packages ".[dev,onnx]" \
    && python3 -c "import numpy, onnxruntime; assert numpy.__version__ == '1.26.0'"

COPY configs ./configs
COPY database ./database
COPY scripts ./scripts
RUN chmod +x /workspace/scripts/*.sh \
    && mkdir -p /workspace/models /workspace/videos /workspace/uploads /workspace/output/snapshot

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "deepstream_ai", "serve", "--config", "/workspace/configs/config.yaml", "--host", "0.0.0.0", "--port", "8080"]
HEALTHCHECK --interval=20s --timeout=5s --start-period=120s --retries=5 \
    CMD ["/workspace/scripts/healthcheck.sh"]

# Tool-only image. Ultralytics/PyTorch are deliberately excluded from the
# production runtime image so model conversion cannot increase steady-state
# memory, attack surface or startup time of the recognition service.
FROM final AS model-converter
RUN python3 -m pip install --no-cache-dir --break-system-packages ".[convert]" \
    && python3 -m pip check
CMD ["python3", "/workspace/scripts/prepare-local-behavior-models.py", "--model-root", "/workspace/models", "--config-root", "/workspace/configs/nvinfer", "--device", "0"]
