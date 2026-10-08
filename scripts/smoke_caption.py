"""Post-deploy smoke test: one real caption from the live Space (TASK-023, ADR-028).

Usage (from ``.github/workflows/deploy-backend.yml``, after the health gate):
    python3 -m scripts.smoke_caption --url https://<space-domain> --origin <frontend origin>

``/healthz`` reporting ``model_loaded: true`` shows the predictor loaded. It doesn't
show that the deployed container can caption an image. This sends one real
``POST /v1/captions`` with a small generated PNG and checks:

* HTTP 200 and a ``CaptionResponse``-shaped JSON body;
* a non-empty caption. The text itself isn't asserted, because it depends on the weights;
* the ``model_version`` that ``/healthz`` reports;
* the ``x-request-id`` sent, echoed in the header and in the body;
* the frontend origin allowed by ``Access-Control-Allow-Origin``.

Connection errors, timeouts and 502/503/504 (a Space still waking) are retried until a
bounded deadline. Any other response fails at once. The API is public, so no token is
used. The log carries statuses, sizes and an error's ``detail``, never image bytes or
the caption. The script uses only the standard library, so the deploy job runs it on
the runner's interpreter.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

FIELD = "image"
FILENAME = "deploy-smoke.png"
CONTENT_TYPE = "image/png"
BOUNDARY = "deploy-smoke-7b3c9e1f"
TRANSIENT_STATUSES = frozenset({502, 503, 504})
REQUEST_TIMEOUT = 120  # s for one request; the first inference after a restart is slow
POLL_SECONDS = 10
MAX_BODY = 64 * 1024  # bytes read from any response
RESPONSE_FIELDS = ("caption", "model_version", "decode_strategy", "latency_ms", "request_id")

# A DNS host name. The deploy workflow checks the Space domain against the same pattern.
HOST_PATTERN = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+"
_URL = re.compile(rf"https://{HOST_PATTERN}")
_ORIGIN = re.compile(r"https?://[A-Za-z0-9.-]+(?::[0-9]{1,5})?")


@dataclass(frozen=True)
class Reply:
    status: int
    headers: dict[str, str]  # lower-case names
    body: bytes


Send = Callable[[str, str, dict[str, str], bytes | None, float], Reply]
# Failures worth another try while a Space wakes: refused or reset connections, timeouts,
# TLS errors (all OSError) and a body cut off mid-read (http.client.HTTPException).
_TRANSIENT_ERRORS = (OSError, http.client.HTTPException)


class SmokeFailure(Exception):
    """The live Space didn't caption the image as the contract says."""


def make_png(width: int = 64, height: int = 64) -> bytes:
    """A small RGB gradient PNG, built in code so no binary fixture is committed."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter type 0 (none) for each scanline
        for x in range(width):
            rows += bytes((x * 255 // (width - 1), y * 255 // (height - 1), 128))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


def multipart(data: bytes) -> tuple[bytes, str]:
    """The upload as the SPA sends it: one ``image`` file part."""
    if BOUNDARY.encode() in data:
        raise ValueError("the multipart boundary occurs in the image bytes")
    head = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="{FIELD}"; filename="{FILENAME}"\r\n'
        f"Content-Type: {CONTENT_TYPE}\r\n\r\n"
    ).encode()
    return (
        head + data + f"\r\n--{BOUNDARY}--\r\n".encode(),
        f"multipart/form-data; boundary={BOUNDARY}",
    )


def _detail(body: bytes) -> str:
    """An error response's ``detail``, or a short excerpt if it isn't the API's JSON."""
    parsed = _json_object(body)
    if parsed is not None and isinstance(parsed.get("detail"), str):
        return str(parsed["detail"])[:300]
    return repr(body[:200])


def check_caption(reply: Reply, *, request_id: str, model_version: str, origin: str) -> list[str]:
    """Everything wrong with a caption reply; empty when it meets the contract."""
    if reply.status != 200:
        return [f"POST /v1/captions returned HTTP {reply.status}: {_detail(reply.body)}"]
    if not reply.headers.get("content-type", "").startswith("application/json"):
        return [f"the response isn't JSON (content-type {reply.headers.get('content-type')!r})"]
    try:
        body = json.loads(reply.body)
    except ValueError:
        return ["the response body isn't valid JSON"]
    if not isinstance(body, dict):
        return ["the response body isn't a JSON object"]
    problems = [f"the response has no {name!r}" for name in RESPONSE_FIELDS if name not in body]
    caption = body.get("caption")
    if "caption" in body and (not isinstance(caption, str) or not caption.strip()):
        problems.append("the caption is empty")
    if "model_version" in body and body["model_version"] != model_version:
        problems.append(
            f"model_version is {body['model_version']!r}, but /healthz reports {model_version!r}"
        )
    strategy = body.get("decode_strategy")
    if "decode_strategy" in body and (not isinstance(strategy, str) or not strategy):
        problems.append("decode_strategy is empty")
    latency = body.get("latency_ms")
    if "latency_ms" in body and (
        isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0
    ):
        problems.append(f"latency_ms is {latency!r}, not a positive number")
    if "request_id" in body and body["request_id"] != request_id:
        problems.append("the body's request_id isn't the x-request-id sent")
    if reply.headers.get("x-request-id") != request_id:
        problems.append("the x-request-id header doesn't echo the one sent")
    allowed = reply.headers.get("access-control-allow-origin")
    if allowed != origin:
        problems.append(f"Access-Control-Allow-Origin is {allowed!r}, not {origin!r}")
    return problems


def _headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Lower-case names; a repeated header's values joined with ", ", as HTTP combines them.

    Two ``Access-Control-Allow-Origin`` values, which browsers reject, then fail the check.
    """
    merged: dict[str, list[str]] = {}
    for name, value in pairs:
        merged.setdefault(name.lower(), []).append(value)
    return {name: ", ".join(values) for name, values in merged.items()}


def _send(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> Reply:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Reply(
                response.status, _headers(response.headers.items()), response.read(MAX_BODY)
            )
    except urllib.error.HTTPError as exc:
        return Reply(exc.code, _headers(exc.headers.items()), exc.read(MAX_BODY))


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _transient(reply: Reply) -> str | None:
    """Why a reply should be retried, or ``None`` if it's final."""
    return f"HTTP {reply.status}" if reply.status in TRANSIENT_STATUSES else None


def _health_transient(reply: Reply) -> str | None:
    """A restart between the health gate and this step looks like a model not loaded yet."""
    if reply.status != 200:
        return _transient(reply)
    status = _json_object(reply.body)
    if status is not None and status.get("model_loaded") is False:
        return "model not loaded yet"
    return None


def _until_settled(
    send: Send,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    *,
    transient: Callable[[Reply], str | None],
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> Reply:
    """The first final reply, retrying transient ones until ``deadline``.

    No attempt starts after the deadline, and each gets at most the time left, so the
    whole check is bounded by ``--timeout``.
    """
    why = "no time to try"
    while (remaining := deadline - clock()) > 0:
        try:
            reply = send(method, url, headers, body, min(REQUEST_TIMEOUT, remaining))
        except _TRANSIENT_ERRORS as exc:
            why = type(exc).__name__
        else:
            pending = transient(reply)
            if pending is None:
                return reply
            why = pending
        print(_escape(f"  {method} {url}: {why}; retrying in {POLL_SECONDS}s"))
        sleep(POLL_SECONDS)
    raise SmokeFailure(f"{method} {url} still failing ({why}) at the deadline")


def run(
    send: Send,
    base_url: str,
    origin: str,
    request_id: str,
    *,
    timeout: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Caption one image on the live Space. Returns a summary; raises ``SmokeFailure``."""
    deadline = clock() + timeout

    def settle(
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None,
        transient: Callable[[Reply], str | None],
    ) -> Reply:
        return _until_settled(
            send,
            method,
            f"{base_url}{path}",
            headers,
            body,
            transient=transient,
            deadline=deadline,
            clock=clock,
            sleep=sleep,
        )

    health = settle("GET", "/healthz", {}, None, _health_transient)
    status = _json_object(health.body) if health.status == 200 else None
    if status is None or status.get("model_loaded") is not True:
        raise SmokeFailure(f"/healthz returned HTTP {health.status} without model_loaded: true")
    model_version = status.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        raise SmokeFailure("/healthz reports no model_version")

    body, content_type = multipart(make_png())
    headers = {"Content-Type": content_type, "Origin": origin, "x-request-id": request_id}
    start = clock()
    reply = settle("POST", "/v1/captions", headers, body, _transient)
    problems = check_caption(
        reply, request_id=request_id, model_version=model_version, origin=origin
    )
    if problems:
        raise SmokeFailure("; ".join(problems))
    result = json.loads(reply.body)
    return (
        f"Captioned a {len(body)}-byte PNG in {clock() - start:.1f}s: HTTP 200, "
        f"{len(result['caption'].split())}-word caption from model {model_version} "
        f"({result['decode_strategy']}, {result['latency_ms']:.0f} ms inference), "
        f"x-request-id echoed, Access-Control-Allow-Origin matched {origin}."
    )


def _escape(text: str) -> str:
    """One log line that can't start a workflow command (GitHub's command escaping)."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _url(value: str) -> str:
    if not _URL.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not an https://host URL without a path: {value!r}")
    return value


def _origin(value: str) -> str:
    if not _ORIGIN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not a scheme://host origin: {value!r}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caption one image on the live Space.")
    parser.add_argument("--url", type=_url, required=True, help="the Space, https://<domain>")
    parser.add_argument("--origin", type=_origin, required=True, help="the frontend's origin")
    parser.add_argument("--timeout", type=float, default=300, help="s to wait out a waking Space")
    args = parser.parse_args(argv)
    try:
        summary = run(_send, args.url, args.origin, uuid.uuid4().hex, timeout=args.timeout)
    except SmokeFailure as exc:
        print(f"::error title=Post-deploy smoke test failed::{_escape(str(exc))}")
        return 1
    print(_escape(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
