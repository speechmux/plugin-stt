"""Minimal streaming STT engine for Phase D integration tests.

Behaviour:
  - Accepts any audio (ignores content).
  - Emits one partial hypothesis every 500 ms while audio is flowing.
  - On KIND_FINALIZE_UTTERANCE: emits a final hypothesis then resets.
  - On KIND_CANCEL: resets without emitting a final.
  - Partial text cycles through ["hello", "hello world", "hello world test"].
"""

import asyncio


from speechmux_plugin_stt.proto.inference.v1 import inference_pb2, inference_pb2_grpc

PARTIALS = ["hello", "hello world", "hello world test"]


class DummyStreamingServicer(inference_pb2_grpc.InferencePluginServicer):
    def HealthCheck(self, request, context):
        return inference_pb2.InferenceCapabilities()  # reuse for health stub; unused here

    def GetCapabilities(self, request, context):
        return inference_pb2.InferenceCapabilities(
            engine_name="dummy_streaming",
            model_size="test",
            device="cpu",
            streaming_mode=inference_pb2.StreamingMode.STREAMING_MODE_NATIVE,
            endpointing_capability=inference_pb2.EndpointingCapability.ENDPOINTING_CAPABILITY_AUTO_FINALIZE,
        )

    def Transcribe(self, request, context):
        return inference_pb2.TranscribeResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            text="",
        )

    async def TranscribeStream(self, request_iterator, context):
        partial_idx = 0
        committed = ""
        partial_task = None

        async def emit_partials():
            nonlocal partial_idx
            while True:
                await asyncio.sleep(0.5)
                text = PARTIALS[partial_idx % len(PARTIALS)]
                partial_idx += 1
                yield inference_pb2.StreamResponse(
                    hypothesis=inference_pb2.StreamHypothesis(
                        text=committed + text,
                        is_final=False,
                    )
                )

        async for req in request_iterator:
            if req.HasField("start"):
                if partial_task is None:
                    partial_task = asyncio.ensure_future(emit_partials().__anext__())
            elif req.HasField("audio"):
                pass  # audio content ignored; partials emit on timer
            elif req.HasField("control"):
                kind = req.control.kind
                if kind == inference_pb2.StreamControl.KIND_FINALIZE_UTTERANCE:
                    if partial_task is not None:
                        partial_task.cancel()
                        partial_task = None
                    final_text = PARTIALS[(partial_idx - 1) % len(PARTIALS)] if partial_idx > 0 else ""
                    committed += final_text + " "
                    yield inference_pb2.StreamResponse(
                        hypothesis=inference_pb2.StreamHypothesis(
                            text=committed.strip(),
                            is_final=True,
                        )
                    )
                    partial_idx = 0
                elif kind == inference_pb2.StreamControl.KIND_CANCEL:
                    if partial_task is not None:
                        partial_task.cancel()
                        partial_task = None
                    partial_idx = 0

        if partial_task is not None:
            partial_task.cancel()
