# ENGINE.md — <engine display name>

Engine-specific rules for `plugin-stt-<impl>`. The common rules for every STT engine
adapter are in `AGENTS.md` beside this file; do not repeat them here. Write only what is
true of **this** engine and would surprise someone who has read `AGENTS.md`.

Delete this instruction block and every `<placeholder>` before committing.

---

## Role

<One paragraph: what runtime this wraps, which hardware it targets, and whether it is a
batch (`InferenceEngine`) or streaming (`StreamingInferenceEngine`) engine. Say why someone
would pick it over the other engines.>

## Entry point and names

| Name | Value |
|------|-------|
| Entry-point name (`server.engine:`) | `<engine_name>` |
| Class | `speechmux_plugin_stt_<impl>.engine:<ClassName>` |
| Core endpoint `id` (`engine_hint`) | `<id in core/config/plugins.yaml>` |
| Config file | `<path to the inference-*.yaml that configures it>` |

## Declared capabilities

```python
engine_name = "<engine_name>"
device = "<cpu|cuda|mlx|...>"
max_concurrent_requests = <n>      # batch engines — say WHY this number
supports_partial_decode = <bool>
# streaming engines instead:
# streaming_mode = STREAMING_MODE_NATIVE
# endpointing_capability = ENDPOINTING_CAPABILITY_<...>
```

<Explain what each non-obvious value means for Core: e.g. why concurrency is 1, what
`fair_dispatch_max_concurrent` should be set to alongside it.>

## Configuration

<Table of the keys `from_config` reads from `engine.<engine_name>:`, with defaults. Note
any key present in the YAML that is accepted but not consumed.>

## Runtime specifics

<The things that make this engine different: how it loads the model, threading model,
what it does with `decode_options` (and which ones it silently ignores), anti-hallucination
or endpointing behaviour, API version quirks of the underlying library.>

## Pitfalls

<Bugs found while building it and the invariant each one taught. Each entry should be
something a reasonable engineer would otherwise do.>

## Docker

<Only if this repo ships a Dockerfile: build context, base image, GPU requirements,
lockfile reuse, model mounting. Delete the section otherwise.>

## Test suite status

<Current state of `make test`: passing count, anything that needs a real dependency, any
known failure with its cause and a link to the plan that tracks it.>

## Do not

<Engine-specific prohibitions only. The common ones are in AGENTS.md.>
