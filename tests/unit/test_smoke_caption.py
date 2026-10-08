"""Tests for the post-deploy caption smoke test (TASK-023, ADR-028).

The checks run against hand-built replies and against the real FastAPI app, through its
real middleware (CORS, request id, body cap) with a stand-in predictor, so nothing goes
over the network. One test decodes the generated image through the serving decoder,
which imports TensorFlow. The last tests read the deploy workflow.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import scripts.smoke_caption as smoke
import yaml
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-backend.yml"
ORIGIN = "https://image-captioning-system.vercel.app"
BASE = "https://space.example.hf.space"
REQUEST_ID = "4f9d2c1ab3e84d5f9a7b6c5d4e3f2a1b"


# ---------------------------------------------------------------- the image


def test_image_is_a_small_deterministic_rgb_png() -> None:
    png = smoke.make_png()
    assert png == smoke.make_png()
    assert len(png) < 16 * 1024  # about 8 KB: one small upload per deploy
    image = Image.open(io.BytesIO(png))
    assert (image.format, image.mode, image.size) == ("PNG", "RGB", (64, 64))
    assert image.getpixel((0, 0)) == (0, 0, 128)
    assert image.getpixel((63, 63)) == (255, 255, 128)


def test_image_decodes_through_the_serving_decoder() -> None:
    """The same decode and preprocessing every upload goes through in production."""
    from app.utils.image import bytes_to_tensor

    tensor = bytes_to_tensor(smoke.make_png())
    assert tuple(tensor.shape) == (299, 299, 3)
    assert tensor.dtype.name == "float32"


def test_multipart_refuses_bytes_containing_its_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        smoke.multipart(b"x" + smoke.BOUNDARY.encode() + b"y")


# ------------------------------------------------------------ the contract


def _reply(
    status: int = 200,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> smoke.Reply:
    if body is None:
        body = {
            "caption": "a blurry photo of a blue wall",
            "model_version": "v2.0.0",
            "decode_strategy": "greedy",
            "latency_ms": 812.4,
            "request_id": REQUEST_ID,
        }
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    base = {
        "content-type": "application/json",
        "x-request-id": REQUEST_ID,
        "access-control-allow-origin": ORIGIN,
    }
    return smoke.Reply(status, {**base, **(headers or {})}, raw)


def _check(reply: smoke.Reply) -> list[str]:
    return smoke.check_caption(reply, request_id=REQUEST_ID, model_version="v2.0.0", origin=ORIGIN)


def _with(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(_reply().body)
    for key, value in changes.items():
        if value is ...:
            del body[key]
        else:
            body[key] = value
    return body


def test_a_reply_that_meets_the_contract_passes() -> None:
    assert _check(_reply()) == []


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        (_reply(500, {"detail": "boom", "request_id": REQUEST_ID}), "HTTP 500: boom"),
        (_reply(422, {"detail": "Could not decode image bytes"}), "HTTP 422: Could not decode"),
        (_reply(200, b"<html>proxy error</html>"), "isn't valid JSON"),
        (_reply(headers={"content-type": "text/html"}), "isn't JSON"),
        (_reply(200, ["a", "list"]), "isn't a JSON object"),
        (_reply(200, _with(caption="")), "the caption is empty"),
        (_reply(200, _with(caption="   ")), "the caption is empty"),
        (_reply(200, _with(caption=None)), "the caption is empty"),
        (_reply(200, _with(caption=...)), "no 'caption'"),
        (_reply(200, _with(model_version="v1.0.0")), "but /healthz reports 'v2.0.0'"),
        (_reply(200, _with(decode_strategy="")), "decode_strategy is empty"),
        (_reply(200, _with(latency_ms=0)), "latency_ms is 0"),
        (_reply(200, _with(latency_ms=True)), "latency_ms is True"),
        (_reply(200, _with(latency_ms="812")), "latency_ms is '812'"),
        (_reply(200, _with(request_id="someone-else")), "the body's request_id"),
        (_reply(headers={"x-request-id": "proxy-id"}), "x-request-id header"),
        (_reply(headers={"access-control-allow-origin": "https://evil.example"}), "evil.example"),
    ],
)
def test_a_reply_that_breaks_the_contract_fails(reply: smoke.Reply, problem: str) -> None:
    assert any(problem in found for found in _check(reply)), _check(reply)


def test_a_missing_cors_header_fails() -> None:
    reply = _reply()
    del reply.headers["access-control-allow-origin"]
    assert _check(reply) == [f"Access-Control-Allow-Origin is None, not {ORIGIN!r}"]


def test_a_repeated_cors_or_request_id_header_fails() -> None:
    """Browsers reject two Access-Control-Allow-Origin values, e.g. one from the proxy and
    one from the app, so the repeated header has to fail rather than read as one."""
    headers = smoke._headers(
        [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", ORIGIN),
            ("access-control-allow-origin", ORIGIN),
            ("X-Request-Id", REQUEST_ID),
            ("x-request-id", "from-the-proxy"),
        ]
    )
    problems = _check(smoke.Reply(200, headers, _reply().body))
    assert any("Access-Control-Allow-Origin is" in p for p in problems), problems
    assert any("x-request-id header" in p for p in problems), problems


# ----------------------------------------------------- waiting out a waking Space


class FakeSpace:
    """Replays replies in order; an exception in the list is raised instead."""

    def __init__(self, *posts: smoke.Reply | Exception, loaded: tuple[bool, ...] = (True,)) -> None:
        self.posts = list(posts)
        self.loaded = list(loaded)  # /healthz model_loaded, one per GET; the last repeats
        self.sent: list[tuple[str, str, dict[str, str]]] = []
        self.timeouts: list[float] = []
        self.now = 0.0
        self.sleeps: list[float] = []

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> Any:
        self.sent.append((method, url, headers))
        self.timeouts.append(timeout)
        if method == "GET":
            loaded = self.loaded.pop(0) if len(self.loaded) > 1 else self.loaded[0]
            payload = {"model_loaded": loaded, "model_version": "v2.0.0"}
            return smoke.Reply(
                200, {"content-type": "application/json"}, json.dumps(payload).encode()
            )
        nxt = self.posts.pop(0) if len(self.posts) > 1 else self.posts[0]
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def run(self, timeout: float = 60) -> str:
        return smoke.run(
            self.send,
            BASE,
            ORIGIN,
            REQUEST_ID,
            timeout=timeout,
            clock=lambda: self.now,
            sleep=self.sleep,
        )


def test_a_waking_space_is_retried_until_it_captions() -> None:
    space = FakeSpace(_reply(503, b""), urllib.error.URLError("reset"), _reply(502, b""), _reply())
    summary = space.run()
    assert "HTTP 200" in summary and "x-request-id echoed" in summary
    assert len(space.sleeps) == 3
    assert "a blurry photo" not in summary  # the caption itself isn't logged


def test_a_space_that_never_wakes_fails_at_the_deadline() -> None:
    space = FakeSpace(_reply(504, b""))
    with pytest.raises(smoke.SmokeFailure, match="HTTP 504"):
        space.run(timeout=60)
    assert sum(space.sleeps) == 60


def test_an_error_from_a_running_space_fails_at_once() -> None:
    space = FakeSpace(_reply(500, {"detail": "predictor crashed", "request_id": REQUEST_ID}))
    with pytest.raises(smoke.SmokeFailure, match="HTTP 500: predictor crashed"):
        space.run()
    assert space.sleeps == []


def test_a_body_cut_off_mid_read_is_retried() -> None:
    space = FakeSpace(http.client.IncompleteRead(b"partial"), ConnectionResetError(), _reply())
    assert "HTTP 200" in space.run()
    assert len(space.sleeps) == 2


def test_a_model_still_loading_after_a_restart_is_waited_for() -> None:
    space = FakeSpace(_reply(), loaded=(False, False, True))
    assert "HTTP 200" in space.run()
    assert [method for method, _, _ in space.sent] == ["GET", "GET", "GET", "POST"]


def test_a_model_that_never_loads_fails_at_the_deadline_before_any_upload() -> None:
    space = FakeSpace(_reply(), loaded=(False,))
    with pytest.raises(smoke.SmokeFailure, match="model not loaded yet"):
        space.run(timeout=30)
    assert {method for method, _, _ in space.sent} == {"GET"}


def test_no_attempt_outlives_the_deadline() -> None:
    """Each request gets at most the time left, so --timeout bounds the whole check."""
    space = FakeSpace(_reply(503, b""))
    with pytest.raises(smoke.SmokeFailure):
        space.run(timeout=25)
    assert space.timeouts[0] == 25
    assert all(0 < timeout <= smoke.REQUEST_TIMEOUT for timeout in space.timeouts)
    assert space.now == 30  # attempts at 0, 10 and 20 s; none at or after the deadline


def test_the_upload_carries_the_origin_and_request_id() -> None:
    space = FakeSpace(_reply())
    space.run()
    method, url, headers = space.sent[-1]
    assert (method, url) == ("POST", f"{BASE}/v1/captions")
    assert headers["Origin"] == ORIGIN
    assert headers["x-request-id"] == REQUEST_ID
    assert headers["Content-Type"] == f"multipart/form-data; boundary={smoke.BOUNDARY}"


# ------------------------------------------------ against the real app stack


class StandInPredictor:
    """Duck-typed ``PredictorService``: the real model only runs on the Space."""

    model_version = "v-smoke"
    decode_strategy = "greedy"
    max_upload_bytes = 10 * 1024 * 1024

    def __init__(self) -> None:
        self.images: list[bytes] = []

    async def caption_image_bytes(self, image_bytes: bytes) -> tuple[str, float]:
        self.images.append(image_bytes)
        return "a picture of a gradient", 3.5


@pytest.fixture
def real_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """``create_app()`` (the production middleware stack), with its CORS origins set."""
    from app.core.config import get_backend_settings

    def build(origins: list[str]) -> tuple[TestClient, StandInPredictor]:
        config = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
        config["serve"]["cors_allowed_origins"] = origins
        path = tmp_path / f"config-{len(origins)}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv("BACKEND_CONFIG_PATH", str(path))
        get_backend_settings.cache_clear()
        from app.main import create_app

        app = create_app()
        predictor = StandInPredictor()
        app.state.predictor_service = predictor
        return TestClient(app), predictor  # no `with`: the lifespan would load real weights

    monkeypatch.chdir(REPO_ROOT)
    # The env variable outranks the YAML, so a value exported in the shell would leak in.
    monkeypatch.delenv("CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS", raising=False)
    yield build
    get_backend_settings.cache_clear()


def _through(client: TestClient) -> smoke.Send:
    def send(
        method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> smoke.Reply:
        response = client.request(method, url.removeprefix(BASE), headers=headers, content=body)
        return smoke.Reply(
            response.status_code,
            smoke._headers(response.headers.multi_items()),
            response.content,
        )

    return send


def test_the_smoke_test_passes_against_the_real_app(real_app: Any) -> None:
    client, predictor = real_app([ORIGIN])
    summary = smoke.run(_through(client), BASE, ORIGIN, REQUEST_ID, timeout=5)
    assert "HTTP 200" in summary and "model v-smoke" in summary
    assert predictor.images == [smoke.make_png()]  # the multipart upload arrived byte for byte


def test_the_smoke_test_fails_when_the_app_does_not_allow_the_origin(real_app: Any) -> None:
    client, _ = real_app(["http://localhost:5173"])
    with pytest.raises(smoke.SmokeFailure, match="Access-Control-Allow-Origin is None"):
        smoke.run(_through(client), BASE, ORIGIN, REQUEST_ID, timeout=5)


def test_the_app_takes_its_origins_from_the_environment_over_the_yaml(
    real_app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production's wiring: the Space's ``CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS`` over a YAML
    that doesn't list the SPA's origin (runbook § 4, TASK-024)."""
    monkeypatch.setenv("CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS", json.dumps([ORIGIN]))
    client, _ = real_app(["http://localhost:5173"])
    smoke.run(_through(client), BASE, ORIGIN, REQUEST_ID, timeout=5)
    # The variable replaces the YAML list, so the YAML-only origin is no longer allowed.
    refused = client.get("/healthz", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in refused.headers


# ------------------------------------------------------------------- the CLI


@pytest.mark.parametrize(
    "argv",
    [
        ["--url", "http://space.hf.space", "--origin", ORIGIN],
        ["--url", f"{BASE}/healthz", "--origin", ORIGIN],
        ["--url", "https://user:pw@space.hf.space", "--origin", ORIGIN],
        ["--url", "https://space.hf.space\nx: y", "--origin", ORIGIN],
        ["--url", BASE, "--origin", f"{ORIGIN}\r\nx-injected: 1"],
        ["--url", BASE, "--origin", "javascript:alert(1)"],
    ],
)
def test_cli_accepts_only_a_bare_https_host_and_a_plain_origin(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        smoke.main(argv)
    assert exc.value.code == 2


def test_cli_failure_is_an_error_annotation_and_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refused(*args: Any) -> smoke.Reply:
        return smoke.Reply(500, {}, b'{"detail": "boom"}')

    monkeypatch.setattr(smoke, "_send", refused)
    assert smoke.main(["--url", BASE, "--origin", ORIGIN, "--timeout", "5"]) == 1
    assert "::error title=Post-deploy smoke test failed::" in capsys.readouterr().out


def test_cli_output_cannot_inject_a_workflow_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A server-supplied detail with a newline mustn't start a line the runner obeys."""

    space = FakeSpace(_reply(500, {"detail": "boom\n::add-mask::x\r%"}))
    monkeypatch.setattr(smoke, "_send", space.send)
    assert smoke.main(["--url", BASE, "--origin", ORIGIN, "--timeout", "5"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith("::")] == [
        "::error title=Post-deploy smoke test failed::"
        "POST /v1/captions returned HTTP 500: boom%0A::add-mask::x%0D%25"
    ]


# -------------------------------------------------------- the deploy workflow


def _steps() -> list[dict[str, Any]]:
    workflow: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["push-to-space"]["steps"]
    return steps


def test_the_smoke_test_gates_the_deploy_between_health_and_record() -> None:
    steps = _steps()
    names = [step.get("id") or step.get("run", "")[:40] for step in steps]
    health = names.index("health")
    smoke_step = next(i for i, s in enumerate(steps) if "scripts.smoke_caption" in s.get("run", ""))
    record = next(
        i for i, s in enumerate(steps) if "scripts.deploy_scope record" in s.get("run", "")
    )
    assert health < smoke_step < record
    step = steps[smoke_step]
    assert step["if"] == "steps.scope.outputs.deploy == 'true'"
    assert "continue-on-error" not in step
    assert step["env"]["SPACE_URL"] == "${{ steps.health.outputs.space_url }}"
    assert step["env"]["FRONTEND_ORIGIN"] == ORIGIN
    assert step["run"] == (
        'python3 -m scripts.smoke_caption --url "$SPACE_URL" --origin "$FRONTEND_ORIGIN"'
    )
    assert "secrets." not in json.dumps(step)  # the API is public
    assert "space_url=https://{domain}" in steps[health]["run"]
    # The health gate checks the domain with the same pattern the script accepts.
    assert f'r"{smoke.HOST_PATTERN}"' in steps[health]["run"]
