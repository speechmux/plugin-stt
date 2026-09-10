# AGENTS.md — STT engine adapter

**This file is identical in every `plugin-stt-*` repository.** It is copied verbatim from
`plugin-stt/templates/AGENTS.md`. Do not edit it in an engine repo — change the template
and re-copy it everywhere. Everything specific to *this* engine is in `ENGINE.md` beside
this file.

Assumes the workspace root `AGENTS.md` and the host framework rules in
`../plugin-stt/AGENTS.md`. Read both, then `ENGINE.md`.

---

## What this repository is

One engine adapter: a single class that implements one of the two STT Protocols from
`speechmux_plugin_stt.engine.base` and registers itself through a Python entry point.

| Protocol | For | RPC the host serves |
|----------|-----|---------------------|
| `InferenceEngine` | Engines that decode a whole segment at once (Whisper family) | `Transcribe` (unary) |
| `StreamingInferenceEngine` | Engines that decode continuously | `TranscribeStream` (bidi) |

`ENGINE.md` states which one this repo implements.

The host framework (`plugin-stt`) owns the gRPC server, the `--config` YAML loader, the
semaphore, `HealthCheck` and `GetCapabilities`. **None of that lives here.** This repo has
no `main.py`, no server code, and reads no config file itself.

## Standard layout

```
plugin-stt-<impl>/
├── AGENTS.md                 # this file — verbatim copy of plugin-stt/templates/AGENTS.md
├── ENGINE.md                 # engine-specific rules, pitfalls and rationale
├── README.md                 # user-facing: install, config keys, model download
├── LICENSE
├── Makefile                  # identical across every engine repo
├── pyproject.toml            # dependency + entry point
├── config/                   # OPTIONAL — only when the engine ships its own inference-*.yaml
├── src/speechmux_plugin_stt_<impl>/
│   ├── __init__.py
│   └── engine.py             # the adapter class (+ helpers if needed)
└── tests/
```

Most engines are configured from a file in `plugin-stt/config/` because that is where the
process is launched from. Add a `config/` here only if the engine needs its own file.

## Entry point

```toml
[project.entry-points."speechmux.stt_engine"]
<engine_name> = "speechmux_plugin_stt_<impl>.engine:<ClassName>"
```

`<engine_name>` is snake_case and is what `server.engine:` in the plugin YAML selects. The
package must be **installed** (`uv pip install -e .`) for the entry point to exist; a
directory on disk is not enough.

Two different names exist and are easy to confuse:

| Name | Where | Example |
|------|-------|---------|
| Entry-point name | `pyproject.toml`, `server.engine:` | `mlx_whisper` |
| Core endpoint `id` | `core/config/plugins.yaml`; what clients pass as `engine_hint` | `whisper-mlx` |

## Engine contract

- Identity fields (`engine_name`, `device`, `streaming_mode`, …) are **class attributes,
  not methods**. The host servicer reads them directly to build `GetCapabilities`.
- `from_config(cls, config)` is the **only** way YAML reaches the engine. It receives the
  `engine.<engine_name>:` section as a dict. Never open a file yourself. Use
  `config.get("key", default)` so an older config still starts.
- `load()` must be **eager**. The host calls it once before the gRPC server accepts
  requests. A lazy first-request load makes the first session hit Core's
  `decode_timeout_sec`.
- Catch `ImportError` on the ML runtime and re-raise with an install hint. A bare
  `ModuleNotFoundError` inside a plugin process is opaque to the operator.
- Convert PCM S16LE to float32 in `[-1, 1]`: `samples.astype(np.float32) / 32768.0`.
- Batch engines: `transcribe()` is **stateless** across requests — Core owns all session
  state. Return a `TranscribeResult`.
- Streaming engines: `stream()` is a generator that must **return**, not raise, when the
  request iterator is exhausted, and must honour `session_config.endpointing_source` —
  in `CORE` mode finalize **only** on `KIND_FINALIZE_UTTERANCE`, never on the engine's own
  endpoint detection (`../docs/decisions/0014-endpointing-source.md`).

## Capabilities are load-bearing

`streaming_mode` and `endpointing_capability` are not descriptive metadata. Core reads them
at runtime (`../docs/decisions/0007-runtime-capability-discovery.md`) and they decide:

- whether Core opens a `TranscribeStream` or sends unary `Transcribe` calls to this endpoint,
- whether `endpointing_source: engine` is legal for it (needs `NATIVE` + `AUTO_FINALIZE`,
  else **ERR1020**).

Misreporting them breaks routing in a way configuration cannot override. Batch engines do
not set them — the host reports `STREAMING_MODE_BATCH_ONLY` for every `InferenceEngine`.

## Build, test, lint

```bash
make install     # uv pip install -e ".[dev]" into ../.venv
make test        # pytest tests/ -v
make lint        # ruff check src/
make typecheck   # mypy src/
```

`Makefile` is identical across engine repos and defaults to `PYTHON ?= ../.venv/bin/python3`.
Do not diverge from it.

## Testing rules

- **Tests never load a model or import the real ML runtime.** Mock it before importing the
  engine — `sys.modules["<runtime>"] = MagicMock()` in a fixture that also purges cached
  plugin modules before and after each test, so the mock wins even when the real package is
  installed.
- Every test that touches engine code takes that fixture, including tests of module-level
  helpers that live in a module importing the runtime.
- **Set every mock attribute the production code reads in a loop condition.** A bare
  `MagicMock()` call result is truthy; `while recognizer.is_ready(stream):` against one never
  terminates and the suite hangs instead of failing.
- The suite must pass on a machine with none of the ML dependencies installed.

## Dependencies

`speechmux-plugin-stt` plus this engine's ML runtime (and `numpy` if needed). Nothing else.
Never depend on Core, another engine, or `grpc` server machinery.

## Do not

- Add a `main.py`, a gRPC server, or CLI flags — the host framework owns those.
- Read a config file directly; only `from_config` receives configuration.
- Make model loading lazy.
- Misreport `streaming_mode` or `endpointing_capability`.
- Keep per-request state in a batch engine.
- Let a streaming `stream()` raise on iterator exhaustion.
- Write a test that imports the real runtime, or build a mock without setting the
  attributes used in loop conditions.
- Diverge the `Makefile` from the other engine repos.
- Edit this file. Edit `plugin-stt/templates/AGENTS.md` and re-copy.

## Where the engine-specific rules are

`ENGINE.md` in this repo: which Protocol, declared capabilities, config keys the engine
reads, its concurrency characteristics, runtime/API quirks, the pitfalls found while
building it, and its test-suite status. Read it before changing `engine.py`.

## Related

- Host framework rules: `../plugin-stt/AGENTS.md`
- Engine contracts and lifecycle: `../docs/architecture/plugin-system.md`
- Wire contract: `../docs/api/plugin-protocol.md`
- Batch and streaming paths in Core: `../docs/architecture/core-pipeline.md`
- Adding another engine: `../.codex/skills/add-engine-plugin/SKILL.md`
