"""STT Inference Plugin gRPC server entry point."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from concurrent import futures
from typing import Any

import grpc

from stt_proto.inference.v1 import inference_pb2_grpc

from speechmux_plugin_stt.engine.base import InferenceEngine
from speechmux_plugin_stt.engine.registry import get_engine, list_engines
from speechmux_plugin_stt.service.inference_servicer import InferencePluginServicer

logger = logging.getLogger(__name__)

_REQUIRED_SERVER_KEYS = ("socket", "engine")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpeechMux STT Inference Plugin")
    p.add_argument(
        "--config",
        required=True,
        help="Path to inference.yaml config file (required)",
    )
    return p.parse_args(argv)


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load and validate the YAML config file at *config_path*.

    Args:
        config_path: Filesystem path to the inference YAML config.

    Returns:
        Parsed config dict.

    Raises:
        SystemExit: When the file cannot be read or required keys are missing.
    """
    try:
        import yaml  # pyyaml — listed in pyproject.toml dependencies

        with open(config_path) as f:
            result = yaml.safe_load(f)
        if not isinstance(result, dict):
            print(f"ERROR: config file {config_path!r} is not a YAML mapping", file=sys.stderr)
            sys.exit(1)
        return result
    except FileNotFoundError:
        print(f"ERROR: config file not found: {config_path!r}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: failed to load config {config_path!r}: {exc}", file=sys.stderr)
        sys.exit(1)


def _require_server_field(config: dict[str, Any], key: str) -> str:
    """Return config.server.<key> as a string, or exit with a clear error.

    Args:
        config: Top-level parsed YAML dict.
        key: Key name under the ``server`` section.

    Returns:
        The string value for the requested key.

    Raises:
        SystemExit: When the key is absent or empty.
    """
    value = (config.get("server") or {}).get(key)
    if not value:
        print(f"ERROR: missing required config key: server.{key}", file=sys.stderr)
        sys.exit(1)
    return str(value)


def _load_engine(engine_name: str, config: dict[str, Any]) -> InferenceEngine:
    """Instantiate the engine registered under *engine_name*.

    Args:
        engine_name: Entry-point name from the ``speechmux.stt_engine`` group.
        config: Top-level parsed YAML dict (engine section is read from here).

    Returns:
        An initialised ``InferenceEngine`` instance.

    Raises:
        SystemExit: When the engine name is not found in registered entry-points.
    """
    if engine_name == "dummy":
        dummy_cfg: dict[str, Any] = (config.get("engine") or {}).get("dummy") or {}
        from speechmux_plugin_stt.engine.dummy import DummySTTEngine

        return DummySTTEngine(
            dummy_text=str(dummy_cfg.get("text") or "hello world"),
            latency_ms=float(dummy_cfg.get("latency_ms") or 0.0),
        )
    engine_config: dict[str, Any] = (config.get("engine") or {}).get(engine_name) or {}
    try:
        return get_engine(engine_name, engine_config if engine_config else None)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Available engines: {list_engines()}", file=sys.stderr)
        sys.exit(1)


def serve(argv: list[str] | None = None) -> None:
    """Start the STT Inference Plugin gRPC server and block until stopped."""
    args = _parse_args(argv)
    config = _load_yaml_config(args.config)

    server_cfg: dict[str, Any] = config.get("server") or {}
    socket_path = _require_server_field(config, "socket")
    engine_name = _require_server_field(config, "engine")
    log_level = str(server_cfg.get("log_level") or "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    engine = _load_engine(engine_name, config)
    logger.info(
        "Loaded STT engine: %s (engine_name=%s)", engine_name, engine.engine_name
    )
    engine.load()

    # Effective concurrency: YAML server.max_concurrent_sessions overrides the
    # engine class default so operators can tune throughput without changing code.
    yaml_max = server_cfg.get("max_concurrent_sessions")
    effective_max = int(yaml_max) if yaml_max is not None else engine.max_concurrent_requests
    logger.info(
        "STT concurrency limit: %d (source: %s)",
        effective_max,
        "yaml" if yaml_max is not None else "engine default",
    )

    log_text: bool = bool(server_cfg.get("log_transcription_text", True))
    servicer = InferencePluginServicer(engine, max_concurrent=effective_max, log_text=log_text)

    # max_workers = effective_max + 4 to prevent HealthCheck/GetCapabilities starvation.
    max_workers = effective_max + 4
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        maximum_concurrent_rpcs=max_workers,
    )
    inference_pb2_grpc.add_InferencePluginServicer_to_server(servicer, server)

    addr = f"unix://{socket_path}"
    server.add_insecure_port(addr)
    server.start()
    logger.info(
        "STT Inference Plugin listening on %s (engine=%s)", addr, engine_name
    )

    def _handle_stop(signum: int, frame: object) -> None:
        logger.info("Received signal %d, stopping…", signum)
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    server.wait_for_termination()
    logger.info("STT Inference Plugin stopped")


if __name__ == "__main__":
    serve()
