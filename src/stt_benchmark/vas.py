"""Route Speechmatics services through a local Voice Agent Service (VAS).

VAS sits between the RT SaaS Proxy and a transcriber. In production the proxy
stamps an ``x-request-metadata`` header on the WebSocket handshake telling VAS
which transcriber instance to route the session to; VAS has no default and
rejects a handshake without it. Benchmarking against a local VAS means standing
in for the proxy, so we stamp that header ourselves.

Set both env vars to use it:

    SPEECHMATICS_RT_URL=ws://localhost:8000/v2/agent   # where VAS listens
    VAS_TRANSCRIBER_URL=localhost:9000                 # transcriber VAS routes to

``VAS_TRANSCRIBER_URL`` is a bare ``host:port`` with no scheme - VAS builds
``ws://<host:port>/v2`` from it, so a scheme here becomes an invalid
``ws://ws://host:port/v2``. Neither hop checks credentials in this setup (VAS
forwards only ``verified-*``/``x-*`` headers downstream, so ``Authorization``
never reaches the transcriber), and ``SPEECHMATICS_API_KEY`` can be any
non-empty placeholder.

The Speechmatics SDK does accept handshake headers - ``AsyncClient.start_session``
takes ``ws_headers`` - but ``VoiceAgentClient.connect`` calls it without them and
pipecat exposes no hook, so we fill them in on the client class.
"""

import json
import os

TRANSCRIBER_URL_ENV = "VAS_TRANSCRIBER_URL"

# Mirrors the routing metadata SaaS Proxy stamps; VAS reads
# transcriber_metadata.transcriber_url and connects to ws://<host:port>/v2.
_REQUEST_METADATA_HEADER = "x-request-metadata"

_patched = False


def transcriber_url() -> str | None:
    """Return the downstream transcriber ``host:port``, or None when unset."""
    return os.getenv(TRANSCRIBER_URL_ENV) or None


def apply_proxy_headers() -> None:
    """Stamp the proxy routing header on every VoiceAgentClient handshake.

    A no-op unless VAS_TRANSCRIBER_URL is set, so cloud runs are untouched.
    Safe to call repeatedly - only the first call patches.
    """
    global _patched

    host_port = transcriber_url()
    if host_port is None or _patched:
        return

    if "://" in host_port:
        raise ValueError(
            f"{TRANSCRIBER_URL_ENV} must be a bare host:port with no scheme, got {host_port!r}"
        )

    from speechmatics.voice import VoiceAgentClient

    metadata = {"transcriber_metadata": {"transcriber_url": host_port}}
    headers = {_REQUEST_METADATA_HEADER: json.dumps(metadata)}
    original = VoiceAgentClient.start_session

    async def start_session(self, *, ws_headers: dict | None = None, **kwargs):
        return await original(self, ws_headers={**headers, **(ws_headers or {})}, **kwargs)

    VoiceAgentClient.start_session = start_session
    _patched = True
