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

Example .claude/settings.json wiring (use an ABSOLUTE interpreter path: hook
processes do not inherit an interactive shell PATH, and a bare `python3` can
resolve to an ancient system interpreter):

  {
    "hooks": {
      "UserPromptSubmit": [{ "hooks": [{ "type": "command",
        "command": "/absolute/path/to/python3 /path/to/scripts/agent_memory_infuse_hook.py --mode full" }] }],
      "PostToolBatch": [{ "hooks": [{ "type": "command",
        "command": "/absolute/path/to/python3 /path/to/scripts/agent_memory_infuse_hook.py --mode delta" }] }]
    }
  }

Configuration: AGENT_MEMORY_MCP_URL (default http://127.0.0.1:8003/mcp/) or --url;
AGENT_MEMORY_INFUSE_TIMEOUT seconds (default 10.0) or --timeout. The default budget
covers the COLD first call of a session: rank against a cold Neo4j page cache
costs roughly 10x the warm call, and the first prompt is exactly when
re-entry matters — a budget sized to the warm call fails silently at the one
moment the hook exists for.

Design constraints, from the spec:
- FAIL SILENT. An infusion that feels broken will be disabled; a broken one
  must never block the turn. Any error -> exit 0, no output.
- FAST. One cached MCP session per (server, harness-session); hard timeout.
- SILENT WHEN SILENT. An empty payload prints nothing — the hook returning
  nothing IS the discipline of silence.

Shadow mode (--shadow): compute everything, inject nothing. Every result —
payload, seed_mode, counts, suppression, timings — is appended as one JSON
line to AGENT_MEMORY_INFUSE_SHADOW_LOG (default ~/.claude/agent_memory-infuse-shadow.jsonl)
and stdout stays empty. This is the zero-constitutive-risk dress rehearsal:
run the wired hooks in shadow for a few wakings against the live substrate
to observe seed quality, delta fire rate, real-graph latency, and the
parked-tensions register at scale, BEFORE the first governed payload is
allowed to condition a live write.

Observation log (AGENT_MEMORY_INFUSE_OBSERVE_LOG=<path>): when set, EVERY call —
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
dependencies, no venv. Requires Python 3.12+: the floor is the oldest
interpreter this hook is actually operated and tested under, not the oldest
that could parse it — an interpreter we never run is a behavior surface we
never verified. Older interpreters are refused cleanly at startup (exit 0,
one line on stderr) instead of crashing mid-import.
"""

import sys

# Interpreter floor — enforced BEFORE the rest of the module executes. The
# PEP 604 annotations below raise at def-time on older interpreters, which
# would crash the script before main()'s fail-silent guard exists. The
# refusal IS the fail-silent path: exit 0 toward the harness, one stderr
# line for the operator running the hook by hand.
MIN_PYTHON = (3, 12)
if sys.version_info < MIN_PYTHON:
    sys.stderr.write(
        f"agent_memory_infuse_hook: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
        f"running {sys.version.split()[0]} ({sys.executable}) — wire the hook "
        "to an absolute path of a supported interpreter.\n")
    sys.exit(0)

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.request

DEFAULT_URL = os.environ.get("AGENT_MEMORY_MCP_URL", "http://127.0.0.1:8003/mcp/")
TIMEOUT_S = float(os.environ.get("AGENT_MEMORY_INFUSE_TIMEOUT", "10.0"))
SHADOW_LOG = os.environ.get(
    "AGENT_MEMORY_INFUSE_SHADOW_LOG",
    os.path.expanduser("~/.claude/agent_memory-infuse-shadow.jsonl"),
)
OBSERVE_LOG = os.environ.get("AGENT_MEMORY_INFUSE_OBSERVE_LOG")  # unset = no observation

# ---------------------------------------------------------------------------
# Full-mode gating: a prompt injection belongs to a prompt somebody WROTE.
#
# THE PARADIGM SPLIT, measured on a live headless agent (2026-08-07). In an
# interactive session each prompt carries new intentionality from a second
# party, which is exactly what the per-prompt injection point is for. A
# scheduled run is a MONOLOGUE: one fixed trigger string, then autonomous
# work. There is no second party, so the per-prompt point has nothing to
# carry — the only place new content enters is tool results, which is the
# delta channel's job.
#
# Left ungated, full mode in a scheduled run is not merely noisy, it builds
# a gravity well. The trigger is byte-identical every waking, so Extract
# yields identical seeds forever; the renewal ledger is session-scoped and a
# waking IS a session, so the ledger is cold every time and the same bodies
# arrive at FULL WEIGHT at the top of every cycle, in perpetuity. The
# renewal economy — the mechanism whose whole purpose is to stop the channel
# droning a permanent core — cannot reach across the session boundary.
# Measured: 16 bodies, 0 handles, 0 standing on every scheduled full call.
#
# THE DISCRIMINATOR is the harness's own, not a heuristic on prompt text:
# Claude Code sets CLAUDE_CODE_ENTRYPOINT=cli for an interactive session and
# sdk-cli under `claude -p`. Detecting INTERACTIVE positively (rather than
# enumerating headless runners we cannot know) means an unrecognised harness
# degrades toward SILENCE, never toward drone — the safe direction, and the
# design's own "silence is a valid injection" principle applied one level
# earlier. Operators whose harness we misclassify have an explicit override.
INTERACTIVE_ENTRYPOINTS = frozenset({"cli"})
FULL_MODE_POLICY = os.environ.get("AGENT_MEMORY_INFUSE_FULL", "interactive").lower()


def full_mode_allowed(env: dict[str, str] | None = None) -> tuple[bool, str]:
    """(allowed, reason) for a full-mode call under the current harness.

    Returns the reason either way so the observation stream can record WHY a
    call was skipped. A guard that suppresses silently is indistinguishable
    from a guard that is broken — this project has shipped that mistake
    three times, and the observe log is where it gets caught.
    """
    env = os.environ if env is None else env
    policy = (env.get("AGENT_MEMORY_INFUSE_FULL", FULL_MODE_POLICY) or "interactive").lower()
    if policy == "always":
        return True, "policy=always"
    if policy == "never":
        return False, "policy=never"
    entrypoint = env.get("CLAUDE_CODE_ENTRYPOINT", "")
    if not entrypoint:
        return False, "no CLAUDE_CODE_ENTRYPOINT — harness unrecognised, degrading to silence"
    if entrypoint in INTERACTIVE_ENTRYPOINTS:
        return True, f"interactive entrypoint ({entrypoint})"
    return False, f"non-interactive entrypoint ({entrypoint}) — scheduled run is a monologue"
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
                "clientInfo": {"name": "agent_memory-infuse-hook", "version": "0.8.0"},
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
    global TIMEOUT_S
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "delta"], default="full")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Request timeout in seconds (overrides AGENT_MEMORY_INFUSE_TIMEOUT; "
             "default 10 — sized to the cold first call, not the warm repeat).",
    )
    parser.add_argument(
        "--shadow", action="store_true",
        help="Compute and log the infusion, inject nothing.",
    )
    args = parser.parse_args()
    if args.timeout is not None:
        TIMEOUT_S = args.timeout

    hook_input = json.load(sys.stdin)

    # Gate full mode BEFORE any network call: a scheduled run's trigger is a
    # ritual string, and infusing on it drones a permanent core (see
    # full_mode_allowed). The skip is RECORDED, never silent — an invisible
    # guard is the failure mode this whole apparatus keeps rediscovering.
    if args.mode == "full":
        allowed, why = full_mode_allowed()
        if not allowed:
            if OBSERVE_LOG:
                _append_log(OBSERVE_LOG, {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "session_id": str(hook_input.get("session_id", "")),
                    "event": hook_input.get("hook_event_name", ""),
                    "mode": args.mode,
                    "injected": False,
                    "skipped": True,
                    "skip_reason": why,
                })
            return 0

    text = _focal_text(hook_input, args.mode)
    if not text.strip():
        return 0

    # The join anchor for the observation stream: sha256 of the stripped
    # focal text, first 16 hex. The invocation scorer hashes each transcript
    # prompt identically and joins on it — a missed call then reads as a
    # counted gap instead of shearing every index-pair after it.
    psha = hashlib.sha256(text.strip().encode()).hexdigest()[:16]

    cache = _session_cache_path(args.url, str(hook_input.get("session_id", "")))
    sid = None
    if os.path.exists(cache):
        with open(cache) as f:
            sid = f.read().strip() or None

    try:
        try:
            result = _call_infuse(args.url, sid, text, args.mode) if sid else {}
            if not sid:
                raise RuntimeError("no cached session")
        except Exception:
            # Session lost (server restart) or never existed — one re-initialize.
            sid = _initialize(args.url)
            if not sid:
                raise RuntimeError("initialize failed")
            result = _call_infuse(args.url, sid, text, args.mode)
    except Exception as exc:
        # Fail silent toward the harness — but the observation stream records
        # the attempt. An unlogged failure is what made the positional join
        # unrepairable: the payload sequence sheared with no visible gap.
        if OBSERVE_LOG:
            _append_log(OBSERVE_LOG, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "session_id": str(hook_input.get("session_id", "")),
                "event": hook_input.get("hook_event_name", ""),
                "mode": args.mode,
                "prompt_sha": psha,
                "error": type(exc).__name__,
            })
        return 0

    with open(cache, "w") as f:
        f.write(sid)

    payload = result.get("payload", "") or ""

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session_id": str(hook_input.get("session_id", "")),
        "event": hook_input.get("hook_event_name", ""),
        "prompt_sha": psha,
        "mode": result.get("mode", args.mode),
        "silence": result.get("silence"),
        "seed_mode": result.get("seed_mode"),
        "rank_mode": result.get("rank_mode"),
        "seed_terms": result.get("seed_terms"),
        "counts": result.get("counts"),
        "suppressed": result.get("suppressed"),
        "renewal": result.get("renewal"),
        # v0.8.0 (EXPERIMENT-BARLOW A9): the assembly accounting — selected /
        # assembled / delivered plus the blind-spot log — and the server
        # version, the seam marker B1 stratifies on across the progression
        # cutover. Absent on pre-0.8.0 servers; .get keeps old servers clean.
        "assembly": result.get("assembly"),
        "server_version": result.get("server_version"),
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
