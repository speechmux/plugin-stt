"""InferenceEngine and StreamingInferenceEngine Protocols for STT engines."""

from __future__ import annotations

import struct
from collections.abc import Generator, Iterator
from typing import NamedTuple, Protocol, runtime_checkable

from stt_proto.inference.v1 import inference_pb2


class TranscribeResult(NamedTuple):
    """Normalised result returned by every InferenceEngine.

    Attributes:
        text: Transcribed text, stripped of leading/trailing whitespace.
        language_code: Detected or requested BCP-47 language code.
        inference_sec: Wall-clock time spent in the engine.
        audio_duration_sec: Duration of the input audio in seconds.
        real_time_factor: ``inference_sec / audio_duration_sec``.
        segments: Per-segment details; each dict has keys ``text``,
            ``start_sec``, ``end_sec``, ``avg_log_prob``, ``no_speech_prob``.
        no_speech_detected: True when the engine found no speech in the audio.
    """

    text: str
    language_code: str
    inference_sec: float
    audio_duration_sec: float
    real_time_factor: float
    segments: list[dict[str, str | float]]
    no_speech_detected: bool


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol that every STT engine adapter must satisfy.

    Engines are discovered via the ``speechmux.stt_engine`` entry-point group.
    """

    engine_name: str
    model_size: str
    device: str
    supported_languages: list[str]
    max_concurrent_requests: int
    supports_partial_decode: bool

    def load(self) -> None:
        """Eagerly load model weights before the gRPC server starts accepting requests.

        Called once at startup, before ``grpc.server.start()``. Engines that
        download or initialise large model files should override this method.
        Engines with no startup cost may leave this as a no-op.

        Returns:
            None
        """
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
        """Decode a PCM S16LE audio segment and return the transcript.

        Args:
            audio_data: Raw PCM S16LE bytes.
            sample_rate: Sample rate of the audio in Hz.
            language_code: BCP-47 language code (e.g. ``"en"``, ``"ko"``).
            task: ``"transcribe"`` or ``"translate"``.
            decode_options: Engine-specific decoding parameters.
            is_final: Whether this is the final audio chunk for the utterance.
            is_partial: Whether this is a partial/interim decode request.

        Returns:
            A ``TranscribeResult`` with text, timing, and segment details.
        """
        ...


@runtime_checkable
class StreamingInferenceEngine(Protocol):
    """Protocol for native-streaming STT engines (STREAMING_MODE_NATIVE).

    Streaming engines manage their own utterance boundary detection and emit
    is_final hypotheses autonomously. The plugin-stt servicer handles gRPC
    framing, semaphore, HealthCheck, and GetCapabilities.

    Engines are discovered via the same ``speechmux.stt_engine`` entry-point
    group as batch engines; the servicer uses isinstance to distinguish them
    at runtime.
    """

    engine_name: str
    supported_languages: list[str]
    max_concurrent_sessions: int
    streaming_mode: int
    endpointing_capability: int

    def load(self) -> None:
        """Eagerly load model weights before the gRPC server starts."""
        ...

    def stream(
        self,
        request_iterator: Iterator[inference_pb2.StreamRequest],
        session_config: inference_pb2.StreamStartConfig,
    ) -> Generator[inference_pb2.StreamResponse, None, None]:
        """Handle one TranscribeStream bidi session.

        The servicer has already consumed and validated the first StreamStartConfig
        message. *request_iterator* yields the remaining AudioChunk / StreamControl
        messages. The generator must be exhaustible — it should return (not raise)
        when the iterator is exhausted.

        Args:
            request_iterator: Remaining request messages after StreamStartConfig.
            session_config: Parsed StreamStartConfig from the first message.

        Yields:
            StreamResponse messages (partial and final hypotheses).
        """
        ...


def pcm16_rms(pcm_data: bytes) -> float:
    """Return the RMS amplitude of a PCM S16LE byte buffer, normalised to [0, 1].

    Args:
        pcm_data: Raw PCM S16LE byte buffer to measure.

    Returns:
        RMS value in [0.0, 1.0] where 1.0 is full-scale (32767).
    """
    num_samples = len(pcm_data) // 2
    if num_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{num_samples}h", pcm_data[: num_samples * 2])
    return float((sum(s * s for s in samples) / num_samples) ** 0.5) / 32768.0
