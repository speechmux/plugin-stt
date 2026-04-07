"""Dummy STT engine — returns a fixed text response with optional artificial delay."""

from __future__ import annotations

import time

from speechmux_plugin_stt.engine.base import InferenceEngine, TranscribeResult, pcm16_rms


class DummySTTEngine(InferenceEngine):
    """A no-op inference engine for testing and load benchmarks.

    Always returns ``dummy_text`` after ``latency_ms`` milliseconds of sleep.
    Returns an empty transcript for silent audio (RMS == 0) when
    ``no_speech_on_silence`` is True.

    Args:
        dummy_text: Fixed text returned for every non-silent audio segment.
        latency_ms: Artificial inference delay in milliseconds. Set to 0 to
            disable.
        no_speech_on_silence: When True, returns an empty transcript for
            silent audio (RMS == 0).
    """

    engine_name: str = "dummy_stt"
    model_size: str = "dummy"
    device: str = "cpu"
    supported_languages: list[str] = ["en", "ko", "zh", "ja"]
    max_concurrent_requests: int = 10
    supports_partial_decode: bool = True

    def __init__(
        self,
        dummy_text: str = "hello world",
        latency_ms: float = 0.0,
        no_speech_on_silence: bool = True,
    ) -> None:
        self._dummy_text = dummy_text
        self._latency_ms = latency_ms
        self._no_speech_on_silence = no_speech_on_silence

    def load(self) -> None:
        """No-op: dummy engine has no model weights to load."""

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
        """Return a fixed dummy transcript.

        Args:
            audio_data: Raw PCM S16LE byte buffer.
            sample_rate: Sample rate in Hz.
            language_code: BCP-47 language code.
            task: ``"transcribe"`` or ``"translate"`` (ignored).
            decode_options: Decoding parameters (ignored).
            is_final: Whether this is the final chunk (ignored).
            is_partial: Whether this is a partial request (ignored).

        Returns:
            A ``TranscribeResult`` with dummy or empty text.
        """
        start = time.perf_counter()
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)
        inference_sec = time.perf_counter() - start

        n_samples = len(audio_data) // 2
        audio_duration_sec = n_samples / max(sample_rate, 1)
        real_time_factor = inference_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0

        silent = self._no_speech_on_silence and pcm16_rms(audio_data) == 0.0
        text = "" if silent else self._dummy_text
        effective_language_code = language_code or "en"

        segments: list[dict[str, str | float]] = []
        if text:
            segments = [
                {
                    "text": text,
                    "start_sec": 0.0,
                    "end_sec": audio_duration_sec,
                    "avg_log_prob": -0.1,
                    "no_speech_prob": 0.0,
                }
            ]

        return TranscribeResult(
            text=text,
            language_code=effective_language_code,
            inference_sec=inference_sec,
            audio_duration_sec=audio_duration_sec,
            real_time_factor=real_time_factor,
            segments=segments,
            no_speech_detected=silent,
        )
