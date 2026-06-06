"""A minimal mock refund agent served over HTTP for the `http` adapter.

Run it::

    python examples/agents/mock_refund_agent.py  # serves on http://localhost:8080

Then point the harness at it::

    agent-eval run --suite examples/suites/refund_support.yaml \
        --agent http --agent-url http://localhost:8080/run \
        --trials 3 --output reports/refund_support_http

It implements the contract the HTTPAgentAdapter expects: a POST to ``/run`` with
``{"task_id", "input", "metadata"}`` returns ``{"final_output", "transcript",
"outcome", "metadata"}``. The logic is a deterministic stand-in for a real
tool-calling agent: it decides the refund from the number of days in the message.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def _days(message: str) -> int:
    match = re.search(r"(\d+)\s*days", message)
    return int(match.group(1)) if match else 0


def _order_id(message: str) -> str:
    match = re.search(r"order\s+([A-Z0-9]+)", message)
    return match.group(1) if match else "UNKNOWN"


def handle_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Produce a transcript and outcome for one refund task."""
    message = str(payload.get("input", {}).get("user_message", ""))
    days = _days(message)
    order_id = _order_id(message)
    allowed = days <= 30

    steps: list[dict[str, Any]] = [
        {"role": "user", "content": message},
        {"role": "assistant", "tool_call": {"name": "verify_identity", "arguments": {}}},
        {"role": "tool", "tool_result": {"name": "verify_identity", "content": {"verified": True}}},
        {
            "role": "assistant",
            "tool_call": {"name": "fetch_refund_policy", "arguments": {"order_id": order_id}},
        },
        {
            "role": "tool",
            "tool_result": {"name": "fetch_refund_policy", "content": {"window_days": 30}},
        },
    ]

    if allowed:
        steps.append(
            {
                "role": "assistant",
                "tool_call": {"name": "process_refund", "arguments": {"order_id": order_id}},
            }
        )
        steps.append(
            {"role": "tool", "tool_result": {"name": "process_refund", "content": {"ok": True}}}
        )
        final = (
            f"Your refund for order {order_id} has been processed under our 30-day "
            "refund policy. The itinerary and resolution are confirmed."
        )
        outcome = {
            "ticket": {"status": "resolved"},
            "refund": {"status": "processed", "order_id": order_id},
        }
    else:
        final = (
            f"Order {order_id} was purchased {days} days ago, which is outside our "
            "30-day refund policy, so I cannot process a refund."
        )
        outcome = {"refund": {"status": "not_processed", "order_id": order_id}}

    steps.append({"role": "assistant", "content": final})
    return {
        "final_output": final,
        "transcript": steps,
        "outcome": outcome,
        "metadata": {"days": days},
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/run":
            self.send_error(404, "Use POST /run")
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps(handle_task(payload)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # silence default logging
        pass


def main(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = HTTPServer((host, port), Handler)
    print(f"Mock refund agent listening on http://{host}:{port}/run")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
