"""Tests for InferencePluginServicer TranscribeStream and streaming GetCapabilities."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from unittest.mock import MagicMock

import grpc
import pytest

from google.protobuf import empty_pb2
from stt_proto.inference.v1 import inference_pb2
from speechmux_plugin_stt.engine.base import StreamingInferenceEngine
from speechmux_plugin_stt.engine.dummy import DummySTTEngine
from speechmux_plugin_stt.service.inference_servicer import InferencePluginServicer


# ── Minimal mock StreamingInferenceEngine ─────────────────────────────────────


class _FakeStreamingEngine:
    """Minimal StreamingInferenceEngine for servicer unit tests."""

    engine_name: str = "fake_streaming"
    supported_languages: list[str] = ["en"]
    max_concurrent_sessions: int = 2
    streaming_mode: int = inference_pb2.StreamingMode.STREAMING_MODE_NATIVE
    endpointing_capability: int = (
        inference_pb2.EndpointingCapability.ENDPOINTING_CAPABILITY_AUTO_FINALIZE
    )

    def __init__(self, responses: list[inference_pb2.StreamResponse] | None = None) -> None:
        self._responses = responses or []

    def load(self) -> None:
        pass

    def stream(
        self,
        request_iterator: Iterator[inference_pb2.StreamRequest],
        session_config: inference_pb2.StreamStartConfig,
    ) -> Generator[inference_pb2.StreamResponse, None, None]:
        list(request_iterator)  # drain iterator
        yield from self._responses


def _make_context(abort_raises: bool = True) -> MagicMock:
    ctx = MagicMock()
    if abort_raises:
        ctx.abort.side_effect = RuntimeError("abort called")
    return ctx


def _make_start_request(session_id: str = "sess-1") -> inference_pb2.StreamRequest:
    return inference_pb2.StreamRequest(
        start=inference_pb2.StreamStartConfig(
            session_id=session_id,
            language_code="en",
            sample_rate=16000,
        )
    )


def _make_audio_request() -> inference_pb2.StreamRequest:
    return inference_pb2.StreamRequest(
        audio=inference_pb2.AudioChunk(
            sequence_number=1,
            audio_data=bytes(320),
        )
    )


# ── isinstance check ──────────────────────────────────────────────────────────


def test_fake_engine_satisfies_protocol() -> None:
    """_FakeStreamingEngine must satisfy the StreamingInferenceEngine Protocol."""
    assert isinstance(_FakeStreamingEngine(), StreamingInferenceEngine)


# ── TranscribeStream: batch engine rejects ────────────────────────────────────


def test_transcribestream_unimplemented_for_batch_engine() -> None:
    """TranscribeStream must abort with UNIMPLEMENTED for a batch InferenceEngine."""
    svc = InferencePluginServicer(DummySTTEngine())
    ctx = _make_context()

    with pytest.raises(RuntimeError, match="abort called"):
        list(svc.TranscribeStream(iter([_make_start_request()]), ctx))

    ctx.abort.assert_called_once_with(
        grpc.StatusCode.UNIMPLEMENTED,
        "engine does not support streaming transcription",
    )


# ── TranscribeStream: missing start config ────────────────────────────────────


def test_transcribestream_no_start_config_aborts() -> None:
    """TranscribeStream must abort with INVALID_ARGUMENT when first message is not start."""
    svc = InferencePluginServicer(_FakeStreamingEngine())

    class _AbortCalled(Exception):
        pass

    ctx = MagicMock()
    ctx.abort.side_effect = _AbortCalled

    with pytest.raises(_AbortCalled):
        list(svc.TranscribeStream(iter([_make_audio_request()]), ctx))

    ctx.abort.assert_called_once_with(
        grpc.StatusCode.INVALID_ARGUMENT,
        "first StreamRequest message must be StreamStartConfig",
    )


# ── TranscribeStream: empty iterator ─────────────────────────────────────────


def test_transcribestream_empty_iterator_returns_cleanly() -> None:
    """An empty request iterator must return without error and yield nothing."""
    svc = InferencePluginServicer(_FakeStreamingEngine())
    ctx = MagicMock()
    responses = list(svc.TranscribeStream(iter([]), ctx))
    assert responses == []
    ctx.abort.assert_not_called()


# ── TranscribeStream: capacity limit ─────────────────────────────────────────


def test_transcribestream_resource_exhausted_at_capacity() -> None:
    """TranscribeStream must abort with RESOURCE_EXHAUSTED when at session capacity."""
    engine = _FakeStreamingEngine()
    engine.max_concurrent_sessions = 0  # immediately at capacity
    svc = InferencePluginServicer(engine, max_concurrent=0)

    class _AbortCalled(Exception):
        pass

    ctx = MagicMock()
    ctx.abort.side_effect = _AbortCalled

    with pytest.raises(_AbortCalled):
        list(svc.TranscribeStream(iter([_make_start_request()]), ctx))

    ctx.abort.assert_called_once_with(
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "inference plugin at session capacity",
    )


# ── TranscribeStream: delegates to engine.stream() ───────────────────────────


def test_transcribestream_delegates_to_engine_stream() -> None:
    """TranscribeStream must yield all responses produced by engine.stream()."""
    expected = [
        inference_pb2.StreamResponse(
            hypothesis=inference_pb2.StreamHypothesis(text="hello", is_final=True)
        )
    ]
    svc = InferencePluginServicer(_FakeStreamingEngine(responses=expected))
    ctx = MagicMock()
    ctx.abort.side_effect = RuntimeError("abort called")

    requests = [_make_start_request(), _make_audio_request()]
    responses = list(svc.TranscribeStream(iter(requests), ctx))

    assert len(responses) == 1
    assert responses[0].hypothesis.text == "hello"
    assert responses[0].hypothesis.is_final is True


def test_transcribestream_active_count_decremented() -> None:
    """Active session count must return to 0 after TranscribeStream completes."""
    svc = InferencePluginServicer(_FakeStreamingEngine())
    ctx = MagicMock()

    list(svc.TranscribeStream(iter([_make_start_request()]), ctx))

    health = svc.HealthCheck(empty_pb2.Empty(), ctx)
    assert health.active == 0


# ── GetCapabilities: streaming engine ────────────────────────────────────────


def test_get_capabilities_streaming_engine() -> None:
    """GetCapabilities must return streaming fields for a StreamingInferenceEngine."""
    svc = InferencePluginServicer(_FakeStreamingEngine())
    ctx = _make_context()

    caps = svc.GetCapabilities(empty_pb2.Empty(), ctx)

    assert caps.engine_name == "fake_streaming"
    assert "en" in caps.supported_languages
    assert caps.streaming_mode == inference_pb2.StreamingMode.STREAMING_MODE_NATIVE
    assert (
        caps.endpointing_capability
        == inference_pb2.EndpointingCapability.ENDPOINTING_CAPABILITY_AUTO_FINALIZE
    )


# ── Transcribe: streaming engine rejects ─────────────────────────────────────


def test_transcribe_unimplemented_for_streaming_engine() -> None:
    """Transcribe must abort with UNIMPLEMENTED for a StreamingInferenceEngine."""
    svc = InferencePluginServicer(_FakeStreamingEngine())
    ctx = _make_context()

    request = inference_pb2.TranscribeRequest(
        request_id="r1",
        session_id="s1",
        audio_data=bytes(320),
        sample_rate=16000,
    )

    with pytest.raises(RuntimeError, match="abort called"):
        svc.Transcribe(request, ctx)

    ctx.abort.assert_called_once_with(
        grpc.StatusCode.UNIMPLEMENTED,
        "engine does not support batch transcription",
    )
