"""Tests for _decode_options_to_dict and _task_name helper functions."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

import grpc
import pytest

from speechmux_plugin_stt.engine.dummy import DummySTTEngine
from speechmux_plugin_stt.service.inference_servicer import (
    InferencePluginServicer,
    _decode_options_to_dict,
    _task_name,
)
from google.protobuf import empty_pb2
from stt_proto.inference.v1 import inference_pb2


# ── helpers ──────────────────────────────────────────────────────────────────


def _pcm_bytes(n_samples: int = 160, value: int = 1000) -> bytes:
    """Create PCM S16LE bytes filled with a constant value."""
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


def _make_context(active: bool = True) -> MagicMock:
    """Build a minimal mock gRPC servicer context.

    Args:
        active: Value returned by ``is_active()``.

    Returns:
        Configured MagicMock.
    """
    ctx = MagicMock()
    ctx.is_active.return_value = active
    ctx.abort.side_effect = RuntimeError("abort called")
    return ctx


def _make_request(
    audio: bytes | None = None,
    decode_options: inference_pb2.DecodeOptions | None = None,
    request_id: str = "req1",
    session_id: str = "s1",
) -> inference_pb2.TranscribeRequest:
    """Build a TranscribeRequest for testing.

    Args:
        audio: PCM bytes; defaults to a short constant-value buffer.
        decode_options: Optional DecodeOptions proto message.
        request_id: RPC request identifier.
        session_id: Session identifier.

    Returns:
        Populated TranscribeRequest.
    """
    kwargs: dict = dict(
        request_id=request_id,
        session_id=session_id,
        audio_data=audio or _pcm_bytes(),
        sample_rate=16000,
        language_code="en",
        is_final=True,
    )
    if decode_options is not None:
        kwargs["decode_options"] = decode_options
    return inference_pb2.TranscribeRequest(**kwargs)


# ── _decode_options_to_dict ───────────────────────────────────────────────────


def test_decode_options_none_returns_empty_dict():
    """_decode_options_to_dict(None) must return an empty dict."""
    assert _decode_options_to_dict(None) == {}


def test_decode_options_default_message_returns_empty_dict():
    """DecodeOptions with all-zero/false fields must yield an empty dict."""
    assert _decode_options_to_dict(inference_pb2.DecodeOptions()) == {}


def test_decode_options_beam_size_only():
    """beam_size=4 must appear as the sole key."""
    result = _decode_options_to_dict(inference_pb2.DecodeOptions(beam_size=4))
    assert result == {"beam_size": 4}


def test_decode_options_temperature_and_no_speech_threshold():
    """Both temperature and no_speech_threshold must be present."""
    result = _decode_options_to_dict(
        inference_pb2.DecodeOptions(temperature=0.8, no_speech_threshold=0.5)
    )
    assert result["temperature"] == pytest.approx(0.8)
    assert result["no_speech_threshold"] == pytest.approx(0.5)
    assert len(result) == 2


def test_decode_options_without_timestamps_true():
    """without_timestamps=True must appear in the dict."""
    result = _decode_options_to_dict(
        inference_pb2.DecodeOptions(without_timestamps=True)
    )
    assert result == {"without_timestamps": True}


def test_decode_options_without_timestamps_false_excluded():
    """without_timestamps=False (default) must not appear in the dict."""
    result = _decode_options_to_dict(
        inference_pb2.DecodeOptions(without_timestamps=False)
    )
    assert "without_timestamps" not in result


def test_decode_options_multiple_fields():
    """Multiple non-default fields must all be included."""
    opts = inference_pb2.DecodeOptions(
        beam_size=5,
        best_of=3,
        temperature=0.2,
        length_penalty=1.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        without_timestamps=True,
    )
    result = _decode_options_to_dict(opts)
    assert result["beam_size"] == 5
    assert result["best_of"] == 3
    assert result["temperature"] == pytest.approx(0.2)
    assert result["length_penalty"] == pytest.approx(1.0)
    assert result["compression_ratio_threshold"] == pytest.approx(2.4)
    assert result["no_speech_threshold"] == pytest.approx(0.6)
    assert result["log_prob_threshold"] == pytest.approx(-1.0)
    assert result["without_timestamps"] is True


# ── _task_name ────────────────────────────────────────────────────────────────


def test_task_name_unspecified_is_transcribe():
    """TASK_UNSPECIFIED (0) must map to 'transcribe'."""
    assert _task_name(0) == "transcribe"


def test_task_name_transcribe_is_transcribe():
    """TASK_TRANSCRIBE (1) must map to 'transcribe'."""
    assert _task_name(1) == "transcribe"


def test_task_name_translate_is_translate():
    """TASK_TRANSLATE (2) must map to 'translate'."""
    assert _task_name(2) == "translate"


def test_task_name_unknown_value_is_transcribe():
    """Any value other than 2 must fall back to 'transcribe'."""
    assert _task_name(99) == "transcribe"


# ── error mapping ─────────────────────────────────────────────────────────────


def test_memory_error_maps_to_model_oom():
    """MemoryError raised by the engine must abort with RESOURCE_EXHAUSTED."""
    engine = DummySTTEngine()

    def raise_oom(*args: object, **kwargs: object) -> None:
        raise MemoryError("GPU out of memory")

    engine.transcribe = raise_oom  # type: ignore[method-assign]
    svc = InferencePluginServicer(engine)
    ctx = _make_context()

    with pytest.raises(RuntimeError, match="abort called"):
        svc.Transcribe(_make_request(), ctx)

    ctx.abort.assert_called_once_with(grpc.StatusCode.RESOURCE_EXHAUSTED, "model out of memory")


def test_value_error_maps_to_invalid_audio():
    """ValueError raised by the engine must abort with INVALID_ARGUMENT."""
    engine = DummySTTEngine()
    error_message = "bad rate: 8000"

    def raise_value_error(*args: object, **kwargs: object) -> None:
        raise ValueError(error_message)

    engine.transcribe = raise_value_error  # type: ignore[method-assign]
    svc = InferencePluginServicer(engine)
    ctx = _make_context()

    with pytest.raises(RuntimeError, match="abort called"):
        svc.Transcribe(_make_request(), ctx)

    code, details = ctx.abort.call_args[0]
    assert code == grpc.StatusCode.INVALID_ARGUMENT
    assert error_message in details


def test_runtime_error_maps_to_inference_failed():
    """RuntimeError raised by the engine must abort with INTERNAL."""
    engine = DummySTTEngine()
    error_message = "crash"

    def raise_runtime(*args: object, **kwargs: object) -> None:
        raise RuntimeError(error_message)

    engine.transcribe = raise_runtime  # type: ignore[method-assign]
    svc = InferencePluginServicer(engine)
    ctx = _make_context()

    with pytest.raises(RuntimeError, match="abort called"):
        svc.Transcribe(_make_request(), ctx)

    code, details = ctx.abort.call_args[0]
    assert code == grpc.StatusCode.INTERNAL
    assert error_message in details


def test_memory_error_releases_semaphore():
    """Semaphore must be released even when engine raises MemoryError."""
    engine = DummySTTEngine()

    def raise_oom(*args: object, **kwargs: object) -> None:
        raise MemoryError

    engine.transcribe = raise_oom  # type: ignore[method-assign]
    svc = InferencePluginServicer(engine)
    ctx = _make_context()

    with pytest.raises(RuntimeError, match="abort called"):
        svc.Transcribe(_make_request(request_id="r99", session_id="sess-7"), ctx)

    # After abort, the semaphore slot must have been released (active == 0).
    health_ctx = MagicMock()
    health = svc.HealthCheck(empty_pb2.Empty(), health_ctx)
    assert health.active == 0
