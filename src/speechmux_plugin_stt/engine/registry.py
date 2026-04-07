"""Engine discovery via ``speechmux.stt_engine`` entry-points."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, cast

from speechmux_plugin_stt.engine.base import InferenceEngine


def get_engine(
    name: str, config: dict[str, Any] | None = None
) -> InferenceEngine:
    """Return an instantiated InferenceEngine registered under *name*.

    If *config* is provided and the engine class exposes a ``from_config``
    classmethod, the engine is constructed via ``cls.from_config(config)``
    so that YAML settings (model size, device, compute type, etc.) are
    applied. Otherwise ``cls()`` is called with no arguments.

    Args:
        name: Entry-point name of the engine to load (e.g. ``"mlx_whisper"``).
        config: Engine-specific config dict parsed from YAML. Passed to
            ``cls.from_config(config)`` if the engine supports it.

    Returns:
        An instantiated InferenceEngine ready to call ``load()`` and
        ``transcribe()``.

    Raises:
        KeyError: If no entry-point named *name* is found in the
            ``speechmux.stt_engine`` group.
    """
    eps = entry_points(group="speechmux.stt_engine")
    for ep in eps:
        if ep.name == name:
            cls = ep.load()
            if config is not None and hasattr(cls, "from_config"):
                return cast(InferenceEngine, cls.from_config(config))
            return cast(InferenceEngine, cls())
    available = [ep.name for ep in eps]
    raise KeyError(
        f"STT engine {name!r} not found. "
        f"Available engines: {sorted(available)}"
    )


def list_engines() -> list[str]:
    """Return the names of all registered STT engines."""
    return sorted(ep.name for ep in entry_points(group="speechmux.stt_engine"))
