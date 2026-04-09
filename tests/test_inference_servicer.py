"""Tests for InferencePluginServicer protocol with the DummySTTEngine."""

from __future__ import annotations

import struct
import threading
from unittest.mock import MagicMock

import pytest

from speechmux_plugin_stt.engine.dummy import DummySTTEngine
from speechmux_plugin_stt.service.inference_servicer import InferencePluginServicer
from google.protobuf import empty_pb2
from stt_proto.common.v1 import common_pb2
from stt_proto.inference.v1 import inference_pb2


def _pcm_bytes(n_samples: int = 160, value: int = 1000) -> bytes:
    """Create PCM S16LE bytes filled with a constant value."""
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


def _silence_bytes(n_samples: int = 160) -> bytes:
    return bytes(n_samples * 2)


def _make_context(active: bool = True) -> MagicMock:
    ctx = MagicMock()
    ctx.is_active.return_value = active
    ctx.abort.side_effect = RuntimeError("abort called")
    return ctx


def _make_request(
    audio: bytes | None = None,
    sample_rate: int = 16000,
    language_code: str = "en",
    is_final: bool = True,
    is_partial: bool = False,
    request_id: str = "req1",
    session_id: str = "s1",
) -> inference_pb2.TranscribeRequest:
    return inference_pb2.TranscribeRequest(
        request_id=request_id,
        session_id=session_id,
        audio_data=audio or _pcm_bytes(),
        sample_rate=sample_rate,
        language_code=language_code,
        is_final=is_final,
        is_partial=is_partial,
    )


# ── basic protocol ──────────────────────────────────────────────────────────

def test_transcribe_returns_text():
    engine = DummySTTEngine(dummy_text="test output")
    svc = InferencePluginServicer(engine)
    ctx = _make_context()

    resp = svc.Transcribe(_make_request(), ctx)

    assert resp.text == "test output"
    assert resp.request_id == "req1"
    assert resp.session_id == "s1"


def test_transcribe_echoes_request_and_session_id():
    svc = InferencePluginServicer(DummySTTEngine())
    ctx = _make_context()

    req = _make_request(request_id="r42", session_id="sess-99")
    resp = svc.Transcribe(req, ctx)

    assert resp.request_id == "r42"
    assert resp.session_id == "sess-99"


def test_transcribe_language_code_echoed():
    svc = InferencePluginServicer(DummySTTEngine())
    ctx = _make_context()

    resp = svc.Transcribe(_make_request(language_code="ko"), ctx)
    assert resp.language_code == "ko"


def test_transcribe_silence_returns_empty_text():
    svc = InferencePluginServicer(DummySTTEngine(dummy_text="hello", no_speech_on_silence=True))
    ctx = _make_context()

    resp = svc.Transcribe(_make_request(audio=_silence_bytes()), ctx)
    assert resp.text == ""
    assert resp.no_speech_detected is True


def test_transcribe_has_audio_duration():
    svc = InferencePluginServicer(DummySTTEngine())
    ctx = _make_context()

    # 320 samples @ 16kHz = 0.02 s
    resp = svc.Transcribe(_make_request(audio=_pcm_bytes(320), sample_rate=16000), ctx)
    assert resp.audio_duration_sec == pytest.approx(0.02)


def test_transcribe_segments_populated():
    svc = InferencePluginServicer(DummySTTEngine(dummy_text="hi"))
    ctx = _make_context()

    resp = svc.Transcribe(_make_request(), ctx)
    assert len(resp.segments) == 1
    assert resp.segments[0].text == "hi"


# ── cancellation guard ───────────────────────────────────────────────────────

def test_cancelled_context_returns_empty_response():
    svc = InferencePluginServicer(DummySTTEngine(dummy_text="should not appear"))
    ctx = _make_context(active=False)

    resp = svc.Transcribe(_make_request(), ctx)
    assert resp.text == ""


# ── capacity limit ───────────────────────────────────────────────────────────

def test_capacity_limit_aborts():
    engine = DummySTTEngine(latency_ms=100)
    engine.max_concurrent_requests = 1
    svc = InferencePluginServicer(engine)

    barrier = threading.Barrier(2)
    results: list = []

    def run_first():
        ctx = _make_context()
        # Patch to simulate slow inference
        original = engine.transcribe
        def slow_transcribe(*a, **kw):
            barrier.wait()  # signal second request to try
            return original(*a, **kw)
        engine.transcribe = slow_transcribe
        results.append(svc.Transcribe(_make_request(), ctx))
        engine.transcribe = original

    t = threading.Thread(target=run_first)
    t.start()

    # Wait until first is inside the semaphore, then try second.
    barrier.wait()

    ctx2 = MagicMock()
    ctx2.is_active.return_value = True
    ctx2.abort.side_effect = RuntimeError("aborted")
    with pytest.raises(RuntimeError, match="aborted"):
        svc.Transcribe(_make_request(request_id="req2"), ctx2)

    import grpc as grpc_mod
    ctx2.abort.assert_called_once_with(
        grpc_mod.StatusCode.RESOURCE_EXHAUSTED,
        "inference plugin at request capacity",
    )
    t.join()


# ── active counter ────────────────────────────────────────────────────────────

def test_active_count_decremented_after_transcribe():
    svc = InferencePluginServicer(DummySTTEngine())
    ctx = _make_context()

    svc.Transcribe(_make_request(), ctx)

    health = svc.HealthCheck(empty_pb2.Empty(), ctx)
    assert health.active == 0


# ── error handling ────────────────────────────────────────────────────────────

def test_engine_exception_aborts_with_internal():
    """RuntimeError from the engine must abort the RPC with INTERNAL status."""
    import grpc as _grpc

    engine = DummySTTEngine()

    def bad_transcribe(*a, **kw):
        raise RuntimeError("model crash")

    engine.transcribe = bad_transcribe  # type: ignore[method-assign]
    svc = InferencePluginServicer(engine)
    ctx = _make_context()

    with pytest.raises(RuntimeError, match="abort called"):
        svc.Transcribe(_make_request(), ctx)

    code, details = ctx.abort.call_args[0]
    assert code == _grpc.StatusCode.INTERNAL
    assert "model crash" in details


# ── GetCapabilities ───────────────────────────────────────────────────────────

def test_get_capabilities():
    engine = DummySTTEngine()
    svc = InferencePluginServicer(engine)
    ctx = _make_context()

    caps = svc.GetCapabilities(empty_pb2.Empty(), ctx)
    assert caps.engine_name == "dummy_stt"
    assert caps.max_concurrent_requests == engine.max_concurrent_requests
    assert len(caps.supported_languages) > 0


# ── HealthCheck ───────────────────────────────────────────────────────────────

def test_health_check_initial():
    svc = InferencePluginServicer(DummySTTEngine())
    ctx = _make_context()

    health = svc.HealthCheck(empty_pb2.Empty(), ctx)
    assert health.state == common_pb2.PLUGIN_STATE_READY
    assert health.active == 0
    assert health.message == "ok"
