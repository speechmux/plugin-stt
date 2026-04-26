"""InferencePluginServicer: Transcribe + TranscribeStream with concurrency limiting."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterator

import grpc

from google.protobuf import empty_pb2
from stt_proto.common.v1 import common_pb2
from stt_proto.inference.v1 import inference_pb2, inference_pb2_grpc

from speechmux_plugin_stt.engine.base import InferenceEngine, StreamingInferenceEngine

logger = logging.getLogger(__name__)


def _task_name(task_enum: int) -> str:
    """Convert inference_pb2.Task enum value to Whisper task name string.

    Args:
        task_enum: Numeric value from inference_pb2.Task enum.

    Returns:
        ``"translate"`` when *task_enum* is ``TASK_TRANSLATE`` (2),
        ``"transcribe"`` otherwise.
    """
    # TASK_TRANSLATE = 2
    return "translate" if task_enum == 2 else "transcribe"


def _decode_options_to_dict(
    opts: inference_pb2.DecodeOptions | None,
) -> dict[str, float | int | bool]:
    """Convert a DecodeOptions proto message to a plain dict.

    Args:
        opts: Proto ``DecodeOptions`` message, or *None*.

    Returns:
        Dictionary with only the non-default fields set.
    """
    if opts is None:
        return {}
    result: dict[str, float | int | bool] = {}
    if opts.beam_size:
        result["beam_size"] = opts.beam_size
    if opts.best_of:
        result["best_of"] = opts.best_of
    if opts.temperature:
        result["temperature"] = opts.temperature
    if opts.length_penalty:
        result["length_penalty"] = opts.length_penalty
    if opts.without_timestamps:
        result["without_timestamps"] = opts.without_timestamps
    if opts.compression_ratio_threshold:
        result["compression_ratio_threshold"] = opts.compression_ratio_threshold
    if opts.no_speech_threshold:
        result["no_speech_threshold"] = opts.no_speech_threshold
    if opts.log_prob_threshold:
        result["log_prob_threshold"] = opts.log_prob_threshold
    return result


class InferencePluginServicer(inference_pb2_grpc.InferencePluginServicer):  # type: ignore[misc]
    """gRPC servicer for the InferencePlugin service.

    Supports both batch (InferenceEngine) and native-streaming
    (StreamingInferenceEngine) engines. Concurrency limiting, HealthCheck, and
    GetCapabilities are handled here so individual engine implementations stay
    free of gRPC concerns.
    """

    def __init__(
        self,
        engine: InferenceEngine | StreamingInferenceEngine,
        max_concurrent: int | None = None,
        log_text: bool = True,
    ) -> None:
        self._engine = engine
        self._is_streaming = isinstance(engine, StreamingInferenceEngine)

        if max_concurrent is not None:
            limit = max_concurrent
        elif self._is_streaming:
            limit = engine.max_concurrent_sessions  # type: ignore[union-attr]
        else:
            limit = engine.max_concurrent_requests  # type: ignore[union-attr]

        self._semaphore = threading.Semaphore(limit)
        self._lock = threading.Lock()
        self._active = 0
        self._state: int = common_pb2.PLUGIN_STATE_READY
        self._last_error_code: int = common_pb2.PLUGIN_ERROR_UNSPECIFIED
        self._last_error_msg: str = ""
        self._log_text = log_text

    # ── RPC handlers ───────────────────────────────────────────────────────

    def Transcribe(
        self,
        request: inference_pb2.TranscribeRequest,
        context: grpc.ServicerContext,
    ) -> inference_pb2.TranscribeResponse:
        if self._is_streaming:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "engine does not support batch transcription",
            )
            raise RuntimeError("abort called")  # unreachable

        # 1. Capacity check — reject immediately if all slots are occupied.
        if not self._semaphore.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "inference plugin at request capacity",
            )
            raise RuntimeError("abort called")  # unreachable; context.abort raises

        with self._lock:
            self._active += 1
        try:
            # 2. Early-exit if the RPC was already cancelled by the caller.
            if not context.is_active():
                logger.debug(
                    "request cancelled before inference session=%s", request.session_id
                )
                return inference_pb2.TranscribeResponse(
                    request_id=request.request_id,
                    session_id=request.session_id,
                )

            # 3. Run inference.
            opts = _decode_options_to_dict(request.decode_options)
            try:
                result = self._engine.transcribe(  # type: ignore[union-attr]
                    audio_data=request.audio_data,
                    sample_rate=request.sample_rate or 16000,
                    language_code=request.language_code,
                    task=_task_name(request.task),
                    decode_options=opts,
                    is_final=request.is_final,
                    is_partial=request.is_partial,
                )
            except MemoryError:
                logger.exception(
                    "OOM during inference request=%s session=%s",
                    request.request_id,
                    request.session_id,
                )
                with self._lock:
                    self._state = common_pb2.PLUGIN_STATE_ERROR
                    self._last_error_code = common_pb2.PLUGIN_ERROR_MODEL_OOM
                    self._last_error_msg = "model out of memory"
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "model out of memory")
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "invalid audio request=%s session=%s: %s",
                    request.request_id,
                    request.session_id,
                    exc,
                )
                with self._lock:
                    self._last_error_code = common_pb2.PLUGIN_ERROR_INVALID_AUDIO
                    self._last_error_msg = f"invalid audio: {exc}"
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception as exc:
                logger.exception(
                    "inference failed request=%s session=%s",
                    request.request_id,
                    request.session_id,
                )
                with self._lock:
                    self._last_error_code = common_pb2.PLUGIN_ERROR_INFERENCE_FAILED
                    self._last_error_msg = (
                        f"inference engine error: {exc}" if str(exc) else "inference engine error"
                    )
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"inference engine error: {exc}" if str(exc) else "inference engine error",
                )

            # 4. Log result and build response.
            text_repr = repr(result.text[:120]) if self._log_text else f"[{len(result.text)}c]"
            decode_type = "final" if request.is_final else "partial"
            log = logger.info if request.is_final else logger.debug
            log(
                "transcribed session=%s request=%s type=%s audio_sec=%.2f rtf=%.2f text=%s",
                request.session_id,
                request.request_id,
                decode_type,
                result.audio_duration_sec,
                result.real_time_factor,
                text_repr,
            )
            segments = [
                inference_pb2.Segment(
                    text=seg.get("text", ""),
                    start_sec=seg.get("start_sec", 0.0),
                    end_sec=seg.get("end_sec", 0.0),
                    avg_log_prob=seg.get("avg_log_prob", 0.0),
                    no_speech_prob=seg.get("no_speech_prob", 0.0),
                )
                for seg in result.segments
            ]
            return inference_pb2.TranscribeResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                text=result.text,
                language_code=result.language_code,
                inference_sec=result.inference_sec,
                audio_duration_sec=result.audio_duration_sec,
                real_time_factor=result.real_time_factor,
                segments=segments,
                no_speech_detected=result.no_speech_detected,
            )
        finally:
            self._semaphore.release()
            with self._lock:
                self._active -= 1

    def TranscribeStream(
        self,
        request_iterator: Iterator[inference_pb2.StreamRequest],
        context: grpc.ServicerContext,
    ) -> Generator[inference_pb2.StreamResponse, None, None]:
        if not self._is_streaming:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "engine does not support streaming transcription",
            )
            return

        # Read and validate first message (must be StreamStartConfig).
        try:
            first = next(request_iterator, None)
        except Exception:
            return
        if first is None:
            return
        if not first.HasField("start"):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "first StreamRequest message must be StreamStartConfig",
            )
            return
        session_config = first.start

        # Capacity check.
        if not self._semaphore.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "inference plugin at session capacity",
            )
            return

        with self._lock:
            self._active += 1
        try:
            yield from self._engine.stream(request_iterator, session_config)  # type: ignore[union-attr]
        except Exception as exc:
            logger.exception(
                "error in TranscribeStream session_id=%s", session_config.session_id
            )
            with self._lock:
                self._last_error = str(exc)
                self._state = common_pb2.PLUGIN_STATE_ERROR
            context.abort(grpc.StatusCode.INTERNAL, f"streaming engine error: {exc}")
        finally:
            self._semaphore.release()
            with self._lock:
                self._active -= 1

    def GetCapabilities(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> inference_pb2.InferenceCapabilities:
        if self._is_streaming:
            engine = self._engine
            return inference_pb2.InferenceCapabilities(
                engine_name=engine.engine_name,
                supported_languages=list(engine.supported_languages),
                streaming_mode=engine.streaming_mode,  # type: ignore[union-attr]
                endpointing_capability=engine.endpointing_capability,  # type: ignore[union-attr]
            )
        return inference_pb2.InferenceCapabilities(
            engine_name=self._engine.engine_name,
            model_size=self._engine.model_size,  # type: ignore[union-attr]
            device=self._engine.device,  # type: ignore[union-attr]
            supported_languages=list(self._engine.supported_languages),
            max_concurrent_requests=self._engine.max_concurrent_requests,  # type: ignore[union-attr]
            supports_partial_decode=self._engine.supports_partial_decode,  # type: ignore[union-attr]
        )

    def HealthCheck(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.PluginHealthStatus:
        with self._lock:
            active = self._active
            state = self._state
            last_error_code = self._last_error_code
            last_error_msg = self._last_error_msg
        return common_pb2.PluginHealthStatus(
            state=state,
            active=active,
            message=last_error_msg if last_error_msg else "ok",
            last_error=last_error_code,
        )
