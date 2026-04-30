# Build context: workspace root (required for proto/ and plugin-stt-sherpa-onnx/).
FROM python:3.13-slim
WORKDIR /app

# libgomp1 is required by sherpa-onnx (OpenMP runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Install plugin-stt framework dependencies from lockfile (cached layer).
COPY plugin-stt/pyproject.toml plugin-stt/uv.lock ./
RUN uv sync --frozen --no-dev

# Install proto and sherpa-onnx engine adapter AFTER sync so sync does not remove them.
# plugin-stt-sherpa-onnx pulls in sherpa-onnx as a dependency.
COPY proto/gen/python/ /tmp/proto/
COPY plugin-stt-sherpa-onnx/ /tmp/plugin-stt-sherpa-onnx/
RUN uv pip install /tmp/proto/ /tmp/plugin-stt-sherpa-onnx/

# Install plugin-stt source in editable mode.
COPY plugin-stt/src/ src/
RUN uv pip install --no-deps -e .

# Models are mounted at /models at runtime — no model files baked into the image.
VOLUME ["/models"]

EXPOSE 50061

ENTRYPOINT ["uv", "run", "python", "-m", "speechmux_plugin_stt.main"]
CMD ["--config", "/etc/speechmux/inference.yaml"]
