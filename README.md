# speechmux/plugin-stt

STT (Speech-to-Text) inference plugin base framework. Provides the gRPC servicer, the `InferenceEngine` and `StreamingInferenceEngine` Protocols (interfaces), a Dummy engine for load testing, and an `entry_points`-based engine auto-discovery system.

## Structure

```
plugin-stt/
├── src/speechmux_plugin_stt/
│   ├── main.py                   # UDS gRPC server entry point
│   ├── service/
│   │   └── inference_servicer.py # Transcribe + TranscribeStream + GetCapabilities + HealthCheck
│   └── engine/
│       ├── base.py               # InferenceEngine and StreamingInferenceEngine Protocols
│       ├── dummy.py              # DummySTTEngine (no model, fixed text)
│       └── registry.py           # entry_points-based engine auto-discovery
├── tests/
├── config/
│   ├── inference-onnx.yaml       # sherpa-onnx Zipformer streaming engine
│   ├── inference-mlx.yaml        # mlx-whisper batch engine (Apple Silicon)
│   └── inference-dummy.yaml      # Dummy engine (load testing)
└── pyproject.toml
```

## Install

```bash
pip install -e ".[dev]"
```

## Run

All configuration — socket path, engine selection, and engine parameters — is read from the YAML config file. Run from the `plugin-stt/` directory:

```bash
# sherpa-onnx Zipformer streaming engine
python -m speechmux_plugin_stt.main --config config/inference-onnx.yaml

# mlx-whisper batch engine (Apple Silicon, after installing plugin-stt-mlx-whisper)
python -m speechmux_plugin_stt.main --config config/inference-mlx.yaml

# Dummy engine (load testing, no model required)
python -m speechmux_plugin_stt.main --config config/inference-dummy.yaml
```

Socket path and engine name are set in the config file under `server.socket` and `server.engine`.

## Test

```bash
python -m pytest tests/ -v
ruff check src/
mypy src/
```

## gRPC Services

### Transcribe (unary RPC)

Used by batch engines (`InferenceEngine`). Core sends a complete audio segment extracted from the ring buffer after EPD triggers and receives the full transcription.

**Request fields:**
- `request_id` — unique per decode request
- `session_id` — owning session
- `audio_data` — PCM S16LE bytes
- `sample_rate` — sample rate in Hz
- `language_code` — BCP-47 code (e.g. `"ko"`, `"en"`)
- `task` — `TASK_TRANSCRIBE` or `TASK_TRANSLATE`
- `decode_options` — beam size, temperature, thresholds
- `is_final` / `is_partial` — decode type flags

**Response fields:**
- `text` — full transcription
- `segments[]` — per-segment text, start/end timestamps, log probabilities
- `inference_sec` — model inference wall time
- `audio_duration_sec` — input audio length
- `real_time_factor` — `inference_sec / audio_duration_sec`
- `no_speech_detected` — true if audio is silence

### TranscribeStream (bidi-stream RPC)

Used by streaming engines (`StreamingInferenceEngine`). Core and the engine exchange audio frames and hypotheses in a continuous bidirectional stream for the lifetime of a session. The engine manages its own utterance boundary detection and emits `is_final` responses autonomously.

### GetCapabilities

Returns `InferenceCapabilities`:
- `engine_name` — e.g. `"sherpa_onnx_zipformer"`, `"mlx_whisper"`, `"dummy_stt"`
- `model_size` — e.g. `"large-v3-turbo"`
- `device` — `"cpu"`, `"cuda"`, `"mps"`, `"mlx"`
- `supported_languages` — list of language codes
- `max_concurrent_requests` — declared capacity
- `supports_partial_decode` — whether engine can handle partial (mid-speech) decodes

### HealthCheck

Returns `PluginHealthStatus` with `PluginState`, in-flight request count, and last error. Core uses this for routing and circuit breaker decisions.

## Configuration

```yaml
server:
  socket: /tmp/speechmux/stt-<engine>.sock  # UDS socket path. Must match plugins.yaml endpoint socket.
  engine: <engine_entry_point>              # Engine entry-point name (speechmux.stt_engine group).
  log_level: INFO                           # DEBUG | INFO | WARNING | ERROR
  max_concurrent_sessions: 10              # Max concurrent requests (batch) or sessions (streaming).
  log_transcription_text: true             # Set to false to redact text from logs.

engine:
  dummy:
    latency_ms: 50       # Artificial inference delay (ms).
    text: "hello world"  # Fixed transcript returned for every non-silent chunk.
```

Per-engine settings live under their respective `engine.<name>` key. See `config/inference-onnx.yaml` and `config/inference-mlx.yaml` for full examples.

## Engine Protocols

### InferenceEngine (batch)

For engines that receive a complete audio segment and return a single result:

```python
class MySTTEngine:
    engine_name: str = "my_engine"
    model_size: str = "base"
    device: str = "cpu"
    supported_languages: list[str] = ["en", "ko"]
    max_concurrent_requests: int = 4
    supports_partial_decode: bool = True

    def load(self) -> None:
        """Load model weights before the gRPC server starts accepting requests."""
        ...

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int,
        language_code: str,
        task: str,
        decode_options: dict[str, float | int | bool],
        is_final: bool,
        is_partial: bool,
    ) -> TranscribeResult:
        """Transcribe audio. Returns TranscribeResult with text, segments, timing."""
        ...
```

### StreamingInferenceEngine (native streaming)

For engines that consume audio frame-by-frame and emit hypotheses in real time, managing their own utterance boundaries:

```python
class MyStreamingEngine:
    engine_name: str = "my_streaming_engine"
    supported_languages: list[str] = ["ko"]
    max_concurrent_sessions: int = 10
    streaming_mode: int = STREAMING_MODE_NATIVE
    endpointing_capability: int = ENDPOINTING_ENGINE

    def load(self) -> None:
        """Load model weights before the gRPC server starts."""
        ...

    def stream(
        self,
        request_iterator: Iterator[inference_pb2.StreamRequest],
        session_config: inference_pb2.StreamStartConfig,
    ) -> Generator[inference_pb2.StreamResponse, None, None]:
        """Handle one TranscribeStream bidi session; yield partial and final hypotheses."""
        ...
```

The servicer uses `isinstance(engine, StreamingInferenceEngine)` to route to `TranscribeStream` vs `Transcribe` at startup.

## Engine Plugin System

Register a new engine under the `speechmux.stt_engine` entry_points group:

```toml
[project.entry-points."speechmux.stt_engine"]
my_engine = "my_package:MySTTEngine"
```

After `pip install my-stt-engine`, set `server.engine: my_engine` in the config YAML.

## Deployment Scenarios

| Environment | Install Command | Config `server.engine` |
|-------------|----------------|------------------------|
| Any CPU/ARM (streaming) | `pip install speechmux-plugin-stt-sherpa-onnx` | `sherpa_onnx_zipformer` |
| Apple Silicon (Metal/MLX) | `pip install speechmux-plugin-stt-mlx-whisper` | `mlx_whisper` |
| CUDA server | `pip install speechmux-plugin-stt-faster-whisper` | `faster_whisper` |
| CPU/MPS (PyTorch) | `pip install speechmux-plugin-stt-torch-whisper` | `torch_whisper` |
| Load testing | (base only) | `dummy` |

## Error Handling

The servicer catches engine exceptions and maps them to `PluginErrorCode`:

| Exception | PluginErrorCode | gRPC Status |
|-----------|----------------|-------------|
| `MemoryError` | `PLUGIN_ERROR_MODEL_OOM` | `RESOURCE_EXHAUSTED` |
| `ValueError` / `TypeError` | `PLUGIN_ERROR_INVALID_AUDIO` | `INVALID_ARGUMENT` |
| Other `Exception` | `PLUGIN_ERROR_INFERENCE_FAILED` | `INTERNAL` |

Core translates these to `ERR####` codes at the boundary. Plugins never use Core error codes directly.

## Dummy Engine

`DummySTTEngine` returns a fixed text string with configurable artificial latency. Edit `config/inference-dummy.yaml` to set `latency_ms` and `text`, then run:

```bash
python -m speechmux_plugin_stt.main --config config/inference-dummy.yaml
```

## License

MIT
