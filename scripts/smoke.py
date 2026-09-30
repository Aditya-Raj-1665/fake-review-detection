"""Verify the running production API and frontend proxy using only the standard library."""

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def request(url: str, payload: dict | None = None) -> tuple[int, bytes]:
    """Send an HTTP request and retain useful non-success status codes."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> None:
    """Check live health, provenance, predictions, errors, and shared-model concurrency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:3000")
    args = parser.parse_args()
    status, body = request(args.api + "/health")
    assert status == 200 and json.loads(body)["model_loaded"]
    status, body = request(args.frontend)
    assert status == 200 and b'<div id="root">' in body
    status, body = request(args.frontend + "/api/v1/model/info")
    assert status == 200
    info = json.loads(body)
    payload = {"text": "The room was quiet but the shower took a while to warm up."}
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: request(args.frontend + "/api/v1/predict", payload), range(8))
        )
    values = []
    for status, body in responses:
        assert status == 200
        prediction = json.loads(body)
        assert prediction["model_version"] == info["model_version"]
        assert 0 <= prediction["authenticity_score"] <= 100
        assert (
            abs(prediction["genuine_probability"] + prediction["deceptive_probability"] - 1) < 1e-9
        )
        values.append(prediction["genuine_probability"])
    assert len(set(values)) == 1
    status, body = request(args.frontend + "/api/v1/predict/batch", {"reviews": ["Good", "Bad"]})
    assert status == 200 and len(json.loads(body)) == 2
    assert request(args.frontend + "/api/v1/predict", {"text": "   "})[0] == 422
    assert request(args.frontend + "/api/v1/predict", {"text": "a" * 100001})[0] == 422
    print(json.dumps({"status": "passed", "model": info, "concurrent_requests": 8}, indent=2))


if __name__ == "__main__":
    main()
