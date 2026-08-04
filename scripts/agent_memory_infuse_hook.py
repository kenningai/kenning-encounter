#!/usr/bin/env python3
"""Reference hook client for the governed infusion pipeline.

Wires a Claude Code harness to the Agent Memory server's `infuse` tool at the two
per-turn injection points (the once-per-waking orientation read stays with
the SessionStart re-entry hook):

  UserPromptSubmit  ->  mode 'full'   (the complete governed disposition,
                                       alongside every prompt)
  PostToolBatch     ->  mode 'delta'  (recognition/conflict only; silence
                                       is a valid injection; fires once per
                                       parallel batch, before the next
                                       model call)

On a harness or SDK without PostToolBatch (e.g. the Python Agent SDK),
wire the delta to PostToolUse instead: the server's per-locus novelty gate
keeps the same at-most-once-per-fact-state semantics either way — the
cadence derives from the substrate, not the scheduler.

Example .claude/settings.json wiring:

  {
    "hooks": {
      "UserPromptSubmit": [{ "hooks": [{ "type": "command",
        "command": "python3 /path/to/scripts/agent_memory_infuse_hook.py --mode full" }] }],
      "PostToolBatch": [{ "hooks": [{ "type": "command",
        "command": "python3 /path/to/scripts/agent_memory_infuse_hook.py --mode delta" }] }]
    }
  }

Configuration: Agent Memory_MCP_URL (default http://127.0.0.1:8003/mcp/) or --url.

Design constraints, from the spec:
- FAIL SILENT. An infusion that feels broken will be disabled; a broken one
  must never block the turn. Any error -> exit 0, no output.
- FAST. One cached MCP session per (server, harness-session); hard timeout.
- SILENT WHEN SILENT. An empty payload prints nothing — the hook returning
  nothing IS the discipline of silence.

Shadow mode (--shadow): compute everything, inject nothing. Every result —
payload, seed_mode, counts, suppression, timings — is appended as one JSON
line to Agent Memory_INFUSE_SHADOW_LOG (default ~/.claude/agent_memory-infuse-shadow.jsonl)
and stdout stays empty. This is the zero-constitutive-risk dress rehearsal:
run the wired hooks in shadow for a few wakings against the live substrate
to observe seed quality, delta fire rate, real-graph latency, and the
parked-tensions register at scale, BEFORE the first governed payload is
allowed to condition a live write.

Observation log (Agent Memory_INFUSE_OBSERVE_LOG=<path>): when set, EVERY call —
injected or silent — appends the same full record (plus an `injected`
flag) to <path>, independent of shadow mode. This is the out-of-band
measurement stream for the infusion experiment: the harness transcript
already records what was injected (additionalContext is saved into the
session transcript), but only this log records the silences — suppressed
deltas, silent full calls, timings, suppression counts — and silence is a
measured variable. PROTOCOL: the observed subject must never read this
file; a subject reading its own measurement stream contaminates the
measures. Keep it outside the project tree; analysis is the observer's.

Speaks MCP streamable-http directly with stdlib only (urllib) — no
dependencies, so the hook can run under any python3 without a venv.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.request

DEFAULT_URL = os.environ.get("Agent Memory_MCP_URL", "http://127.0.0.1:8003/mcp/")
TIMEOUT_S = float(os.environ.get("Agent Memory_INFUSE_TIMEOUT", "3.0"))
SHADOW_LOG = os.environ.get(
    "Agent Memory_INFUSE_SHADOW_LOG",
    os.path.expanduser("~/.claude/agent_memory-infuse-shadow.jsonl"),
)
OBSERVE_LOG = os.environ.get("Agent Memory_INFUSE_OBSERVE_LOG")  # unset = no observation
PROTOCOL_VERSION = "2025-06-18"
_DATA_RE = re.compile(r"^data: ?(.*)$", re.MULTILINE)


def _post(url: str, payload: dict, session_id: str | None) -> tuple[dict | None, str | None]:
    """POST one JSON-RPC message; return (parsed result message, session id)."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        sid = resp.headers.get("Mcp-Session-Id") or session_id
        body = resp.read().decode("utf-8", errors="replace")
        ctype = resp.headers.get("Content-Type", "")
    if not body.strip():
        return None, sid
    if "text/event-stream" in ctype:
        # Take the last data frame — the response message.
        frames = _DATA_RE.findall(body)
        for frame in reversed(frames):
            frame = frame.strip()
            if frame:
                return json.loads(frame), sid
        return None, sid
    return json.loads(body), sid


def _session_cache_path(url: str, harness_session: str) -> str:
    key = hashlib.sha256(f"{url}|{harness_session}".encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"agent_memory-infuse-{key}.session")


def _initialize(url: str) -> str | None:
    msg, sid = _post(
        url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agent_memory-infuse-hook", "version": "0.5.0"},
            },
        },
        None,
    )
    if sid:
        _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    return sid


def _call_infuse(url: str, sid: str, text: str, mode: str) -> dict:
    msg, _ = _post(
        url,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "infuse", "arguments": {"text": text, "mode": mode}},
        },
        sid,
    )
    if not msg or "error" in msg:
        raise RuntimeError(str(msg))
    content = msg["result"]["content"][0]["text"]
    return json.loads(content)


def _focal_text(hook_input: dict, mode: str) -> str:
    event = hook_input.get("hook_event_name", "")
    if event == "UserPromptSubmit" or mode == "full":
        return str(hook_input.get("prompt", ""))
    # Tool-batch shapes: single response (PostToolUse) or a batch list.
    for key in ("tool_responses", "tool_response", "tool_result"):
        if key in hook_input:
            return json.dumps(hook_input[key], default=str)[:8000]
    return ""


def _append_log(path: str, record: dict) -> None:
    """Append one observation record; never let logging break the hook."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "delta"], default="full")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--shadow", action="store_true",
        help="Compute and log the infusion, inject nothing.",
    )
    args = parser.parse_args()

    hook_input = json.load(sys.stdin)
    text = _focal_text(hook_input, args.mode)
    if not text.strip():
        return 0

    cache = _session_cache_path(args.url, str(hook_input.get("session_id", "")))
    sid = None
    if os.path.exists(cache):
        with open(cache) as f:
            sid = f.read().strip() or None

    try:
        result = _call_infuse(args.url, sid, text, args.mode) if sid else {}
        if not sid:
            raise RuntimeError("no cached session")
    except Exception:
        # Session lost (server restart) or never existed — one re-initialize.
        sid = _initialize(args.url)
        if not sid:
            return 0
        result = _call_infuse(args.url, sid, text, args.mode)

    with open(cache, "w") as f:
        f.write(sid)

    payload = result.get("payload", "") or ""

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session_id": str(hook_input.get("session_id", "")),
        "event": hook_input.get("hook_event_name", ""),
        "mode": result.get("mode", args.mode),
        "silence": result.get("silence"),
        "seed_mode": result.get("seed_mode"),
        "rank_mode": result.get("rank_mode"),
        "seed_terms": result.get("seed_terms"),
        "counts": result.get("counts"),
        "suppressed": result.get("suppressed"),
        "renewal": result.get("renewal"),
        "timings_ms": result.get("timings_ms"),
        "payload_chars": len(payload),
        "payload": payload,
        "injected": bool(payload) and not args.shadow,
    }
    if OBSERVE_LOG:
        _append_log(OBSERVE_LOG, record)

    if args.shadow:
        _append_log(SHADOW_LOG, record)
        return 0

    if payload:
        event = hook_input.get("hook_event_name", "UserPromptSubmit")
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": payload,
            }
        }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Fail silent by design: infusion must never block the turn.
        sys.exit(0)
