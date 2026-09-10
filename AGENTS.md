# AGENTS.md — plugin-stt

STT plugin **host framework**. Assumes the workspace root `AGENTS.md`; only what is
specific to this repository is here.

Root instructions: `../AGENTS.md` · Architecture:
`../docs/architecture/plugin-system.md` · Wire contract:
`../docs/api/plugin-protocol.md`

---

## Role

The gRPC server, the config loader and the engine contracts for speech recognition. Ships
no real model — only a `dummy` engine. Actual decoding lives in engine adapter repos
(`plugin-stt-sherpa-onnx`, `plugin-stt-mlx-whisper`, `plugin-stt-faster-whisper`).

Unlike `plugin-vad`, this repo serves **two** engine contracts from one servicer, and it is
what tells Core which decode path a session takes.

## Two engine contracts

| Protocol | RPC served | Capability reported |
|----------|-----------|---------------------|
| `InferenceEngine` | `Transcribe` (unary) | `STREAMING_MODE_BATCH_ONLY` |
| `StreamingInferenceEngine` | `TranscribeStream` (bidi) | the engine's own `streaming_mode` and `endpointing_capability` |

`InferencePluginServicer.__init__` decides with
`isinstance(engine, StreamingInferenceEngine)` and caches the result in `self._is_streaming`.
That one line determines which branch `GetCapabilities` takes, which in turn determines
whether Core builds a `batchDecodeEngine` or a `streamingDecodeEngine` for every session
(`../docs/decisions/0007-runtime-capability-discovery.md`). Get it wrong and Core opens a
`TranscribeStream` nothing serves, or sends unary requests to a streaming-only engine.

## Boundary with Core

- Core owns session state, scheduling and concurrency. `Transcribe` is **stateless** — it
  decodes the bytes it is given and returns.
- Never emit `ERR####`. Report a `common.v1.PluginErrorCode`; Core translates it.
- For `TranscribeStream`, the servicer consumes and validates the first `StreamStartConfig`
  before handing the iterator to `engine.stream()`. The engine sees only `AudioChunk` and
  `StreamControl`.
- `StreamStartConfig.endpointing_source` must be honoured by the engine: in `CORE` mode it
  finalizes **only** on `KIND_FINALIZE_UTTERANCE`
  (`../docs/decisions/0014-endpointing-source.md`).

## Entry point

```bash
python -m speechmux_plugin_stt.main --config plugin-stt/config/inference-onnx.yaml
```

`--config` is the **only** CLI flag. Do not add others.

## Layout

| Path | Contains |
|------|----------|
| `src/speechmux_plugin_stt/main.py` | Arg parsing, YAML loading, engine construction, `engine.load()`, gRPC server |
| `src/speechmux_plugin_stt/service/inference_servicer.py` | `Transcribe`, `TranscribeStream`, `GetCapabilities`, `HealthCheck`, semaphore, error mapping |
| `src/speechmux_plugin_stt/engine/base.py` | `InferenceEngine`, `StreamingInferenceEngine`, `TranscribeResult`, `pcm16_rms` — **public API for every engine repo** |
| `src/speechmux_plugin_stt/engine/registry.py` | Entry-point lookup (`speechmux.stt_engine`) |
| `src/speechmux_plugin_stt/engine/dummy.py` | No-model engine for load testing |
| `config/inference-*.yaml` | Reference configs, one per engine |
| `templates/AGENTS.md` | **Canonical `AGENTS.md` for every `plugin-stt-*` engine repo** — copied verbatim into each |
| `templates/ENGINE.md` | Skeleton for an engine repo's engine-specific document |

`engine/base.py` is a published API. Downstream engine repos import it by path.

## Engine repos share one `AGENTS.md`, owned here

Every `plugin-stt-*` repository carries a byte-identical `AGENTS.md` copied from
`templates/AGENTS.md`, plus its own `ENGINE.md` for engine-specific rules, plus a
`CLAUDE.md` containing only `@AGENTS.md`. A new engine author copies both templates, adds
that one-line file, and fills in `ENGINE.md` only.

When you change `templates/AGENTS.md`, re-copy it into **every** engine repo in the same
piece of work, and verify with:

```bash
for d in ../plugin-stt-*/; do cmp -s templates/AGENTS.md "$d/AGENTS.md" || echo "DRIFT: $d"; done
```

Never put something engine-specific in the template, and never put a common rule in an
engine's `ENGINE.md`.

## Config files live here for engines that do not own one

`config/inference-onnx.yaml` and `config/inference-mlx.yaml` sit in this repo even though
they configure engines in other repos, because that is where the process is launched from.
A file may contain sections for engines that are not installed — only
`engine.<server.engine>` is read.

`config/inference-onnx.yaml` currently carries a `torch_whisper:` section for an engine
repo that does not exist (`../docs/plans/roadmap.md`).

## Build, test, lint

```bash
make install     # uv pip install -e ".[dev]" into ../.venv
make test        # pytest tests/ -v          → 38 passing
make lint        # ruff check src/
make typecheck   # mypy src/                 → clean
make run-dummy   # start with config/inference-dummy.yaml
```

Known baseline: `make lint` reports 5 findings under ruff 0.16.5 (`RUF012`, `I001`) — the
pin is `ruff>=0.4` with no upper bound. See `../docs/plans/test-and-lint-gaps.md`.

`make typecheck` passes because `pyproject.toml` carries a `[[tool.mypy.overrides]]` block
setting `ignore_missing_imports` for `grpc.*`, `stt_proto.*` and `google.protobuf.*`.
Keep it — `plugin-vad` lacks it and fails.

## Dependencies

`grpcio`, `protobuf`, `pyyaml`, `speechmux-proto`. **No ML runtime, no numpy** — those
belong in engine repos.

## Changing the engine Protocols

Both `InferenceEngine` and `StreamingInferenceEngine` are contracts implemented in other
repositories. A change here fails downstream at runtime, not at build time.

- Prefer optional attributes with a framework-side default.
- Update `engine/dummy.py`, `tests/dummy_streaming_engine.py` and every engine repo in the
  same piece of work.
- Identity attributes are **class attributes, not methods** — the servicer reads them
  directly for `GetCapabilities`. Keep it that way.
- Both Protocols are `@runtime_checkable`; the `isinstance` dispatch depends on it.

## Concurrency and error mapping

`max_workers = max_concurrent_sessions + 4`. The `+ 4` prevents `HealthCheck` starving
behind occupied worker threads, which Core would read as a dead plugin.

The semaphore is acquired **non-blocking**; over capacity aborts with `RESOURCE_EXHAUSTED`
/ `PLUGIN_ERROR_CAPACITY_FULL` (→ Core ERR2008) rather than queueing. Queueing here would
be invisible to Core's `FairDecodeDispatcher`, which exists precisely to control that
queue (`../docs/decisions/0011-fair-decode-dispatcher.md`).

`context.abort()` **raises**, so it exits before any `try/finally`. That is why the
semaphore release is placed deliberately and why the code has `raise RuntimeError(...)`
lines marked unreachable after each abort — they are there for the type checker. Do not
"clean them up" without re-checking the release path.

Error mapping in `Transcribe`: OOM → `RESOURCE_EXHAUSTED` and plugin state `ERROR`; bad
audio → `INVALID_ARGUMENT`; anything else → `INTERNAL` with `last_error` set but the state
left `READY`.

## Privacy

`server.log_transcription_text: false` logs a character count instead of the transcript.
Any new log line that could include recognised text must honour the same flag.

## Testing

`tests/` covers the servicer with dummy engines — `dummy_streaming_engine.py` is the
streaming stand-in. No model, no network.

When adding a test:

- Streaming: the first message must be `StreamStartConfig`; assert the servicer rejects a
  stream that starts with anything else.
- Assert `GetCapabilities` reports `STREAMING_MODE_BATCH_ONLY` on the batch path — Core's
  `RouteBatch()` filters strictly on it.
- Cover the semaphore-refusal path.
- Never import a real ML runtime.

## Do not

- Add a CLI flag other than `--config`.
- Add an ML dependency.
- Make `Transcribe` stateful, or cache anything across requests beyond the loaded model.
- Emit `ERR####` codes.
- Queue inside the plugin instead of refusing at the semaphore.
- Change the `isinstance` dispatch without tracing the consequence through
  `GetCapabilities` → `PluginRouter.RouteBatch` → `StreamProcessor.ProcessSession`.

## Related

- Adding an engine: `../.codex/skills/add-engine-plugin/SKILL.md`
- Proto changes: `../.codex/skills/change-proto/SKILL.md`
- Config keys: `../.codex/skills/add-config-option/SKILL.md`
