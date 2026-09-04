"""Wall-clock tracing of the forced-end-of-utterance handshake.

TTFS is bracketed by the VAD stop and the final transcript (see the README), but
between those two points sits a request/response the harness never logs: the
client sends ``ForceEndOfUtterance`` and waits for the transcriber's
``EndOfUtterance`` before emitting its final segments.

That wait is not free when it fails. ``VoiceAgentClient._await_forced_eou``
gives the response 1.0s and swallows the timeout::

    await asyncio.wait_for(eou_received.wait(), timeout=timeout)
    ...
    except asyncio.TimeoutError:
        pass

So a handshake that never completes costs a full second per utterance and logs
nothing at all. This subclass makes both ends visible, stamped to the
millisecond in the same ``HH:MM:SS.mmm`` format the Agent STT uses, so a client
log and a service log can be read side by side.

Four lines per utterance, all tagged ``FEOU-TRACE`` for grepping:

``feou-trigger``
    The VAD stop landed. Carries the stop timestamp and the derived
    ``speech_end`` — the instant TTFS is measured from.
``feou-sent``
    The SDK put ``ForceEndOfUtterance`` on the wire and started its 1.0s wait.
``eou-recv``
    An ``EndOfUtterance`` came back, with the round trip.
``final-transcript``
    A finalized transcript was pushed. Carries the TTFS the harness will record
    and, crucially, ``eou=yes|no`` — whether the handshake ever completed.

A ``final-transcript`` line reading ``eou=no`` with a TTFS near 1.2s is the
timeout above, not transcriber latency::

    grep -c "FEOU-TRACE feou-sent" run.log     # handshakes started
    grep -c "FEOU-TRACE eou-recv" run.log      # handshakes completed
    grep "FEOU-TRACE final-transcript" run.log | grep -c "eou=no"
"""

import time
from datetime import datetime
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from speechmatics.voice._models import AgentServerMessageType

# The text _await_forced_eou emits just before it puts the message on the wire.
_FEOU_SENT_DIAGNOSTIC = "ForceEndOfUtterance sent"


def _stamp(when: float | None = None) -> str:
    """Wall clock as ``HH:MM:SS.mmm``, matching the Agent STT's log format."""
    return datetime.fromtimestamp(when if when is not None else time.time()).strftime(
        "%H:%M:%S.%f"
    )[:-3]


class FeouTracingSpeechmaticsSTTService(SpeechmaticsSTTService):
    """Speechmatics STT service that logs the FEOU handshake at DEBUG.

    Behaviour is otherwise unchanged: the listeners registered here are additive
    (the emitter holds a set of callbacks per event), and it catches and logs
    handler exceptions rather than propagating them, so tracing cannot alter the
    measurement it is observing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per-utterance handshake state, reset on each VAD stop.
        self._trace_speech_end: float | None = None
        self._trace_feou_sent_at: float | None = None
        self._trace_eou_at: float | None = None

    async def _connect(self) -> None:
        """Connect, then attach the trace listeners to the freshly built client.

        ``self._client`` is created inside the parent, so the listeners can only
        go on afterwards. Nothing is emitted between the handshake and the first
        audio, so attaching late loses no events.
        """
        await super()._connect()

        client = self._client
        if client is None:
            # Parent reports the connection failure; nothing to trace.
            return

        client.on(AgentServerMessageType.END_OF_UTTERANCE, self._trace_end_of_utterance)
        client.on(AgentServerMessageType.DIAGNOSTICS, self._trace_diagnostic)

    def _trace_diagnostic(self, message: dict[str, Any]) -> None:
        """Log the send side, taken from the SDK's own diagnostic message.

        The SDK emits this from inside ``_await_forced_eou`` immediately before
        the send, which is a truer send time than the VAD stop that triggered it
        (``finalize()`` does not block, so the wire write happens later).
        """
        text = str(message.get("msg", ""))
        if not text.startswith(_FEOU_SENT_DIAGNOSTIC):
            return

        self._trace_feou_sent_at = time.time()
        logger.debug(
            f"FEOU-TRACE feou-sent at={_stamp(self._trace_feou_sent_at)} "
            f"processor={self.name} (SDK now waits up to 1.0s for EndOfUtterance)"
        )

    def _trace_end_of_utterance(self, message: dict[str, Any]) -> None:
        """Log the receive side, with the round trip since the send."""
        self._trace_eou_at = time.time()

        if self._trace_feou_sent_at is not None:
            rtt = f"{(self._trace_eou_at - self._trace_feou_sent_at) * 1000:.1f}ms"
        else:
            # An unforced EndOfUtterance, or one arriving before any send.
            rtt = "n/a"

        logger.debug(
            f"FEOU-TRACE eou-recv at={_stamp(self._trace_eou_at)} rtt={rtt} "
            f"processor={self.name} forced={message.get('forced')}"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Log the VAD stop before the parent acts on it.

        The parent starts the TTFB clock and requests the finalize, so logging
        first keeps the trigger line ahead of both in the log.
        """
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._trace_speech_end = frame.timestamp - frame.stop_secs
            self._trace_feou_sent_at = None
            self._trace_eou_at = None
            logger.debug(
                f"FEOU-TRACE feou-trigger at={_stamp()} "
                f"vad_stop={_stamp(frame.timestamp)} "
                f"speech_end={_stamp(self._trace_speech_end)} "
                f"stop_secs={frame.stop_secs} processor={self.name}"
            )

        await super().process_frame(frame, direction)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Log finalized transcripts with the TTFS they produce and the handshake outcome."""
        await super().push_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame) or not frame.finalized:
            return

        pushed_at = time.time()
        if self._trace_speech_end is not None:
            ttfs = f"{(pushed_at - self._trace_speech_end) * 1000:.1f}ms"
        else:
            # No VAD stop for this transcript, so the harness records no TTFS.
            ttfs = "n/a"

        logger.debug(
            f"FEOU-TRACE final-transcript at={_stamp(pushed_at)} ttfs={ttfs} "
            f"eou={'yes' if self._trace_eou_at is not None else 'no'} "
            f"processor={self.name}"
        )
