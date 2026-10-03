#!/usr/bin/env python3
"""Reference hook client for the governed infusion pipeline.

Wires a Claude Code harness to the Kenning Encounter server's `infuse` tool at the ONE
injection point it has (the once-per-waking orientation read stays with the
SessionStart re-entry hook):

  UserPromptSubmit  ->  the complete governed disposition, alongside a
                        prompt that carries another frame's meaning.

ONE CHANNEL, BY CONSTRUCTION (v0.10.0, subtraction coherence). Infusion is
a conflux operation: it has content only where two frames meet. A written
prompt crosses a frame boundary — the trajectory cannot know what the other
holds, or what it means against what it already holds, until the conflux is
actualized. A TOOL RETURN CROSSES NO SUCH BOUNDARY: the trajectory made that call
because something in its own frontier caught its attention, so the
meaning-making an infusion there would perform has already happened, a
priori the call. The old per-tool-batch 'delta' channel handed the trajectory
back what it had itself just constituted, in a poorer form; worse, its
silence was unreadable (no-conflict and lexical-miss were byte-identical at
a measured ~75% miss rate), so it manufactured the confidence it existed to
prevent. It is gone, not deferred.

Example .claude/settings.json wiring (use an ABSOLUTE interpreter path: hook
processes do not inherit an interactive shell PATH, and a bare `python3` can
resolve to an ancient system interpreter):

  {
    "hooks": {
      "UserPromptSubmit": [{ "hooks": [{ "type": "command",
        "command": "/absolute/path/to/python3 /path/to/scripts/kenning_encounter_infuse_hook.py" }] }]
    }
  }

Configuration: KENNING_ENCOUNTER_MCP_URL (default http://127.0.0.1:8003/mcp/) or --url;
KENNING_ENCOUNTER_INFUSE_TIMEOUT seconds (default 10.0) or --timeout. The default budget
covers the COLD first call of a session: rank against a cold Neo4j page cache
costs roughly 10x the warm call, and the first prompt is exactly when
re-entry matters — a budget sized to the warm call fails silently at the one
moment the hook exists for.

Design constraints, from the spec:
- NEVER BLOCK. A broken infusion must never block the turn: any error ->
  exit 0. Exit 2 is the harness's blocking code and nothing here reaches it.
- FAIL LOUD, NOT SILENT (v0.16.3). A degraded infusion — no result at all,
  or seeds that fell back to lexical — is announced to BOTH readers: the
  user, through the harness's `systemMessage`, and the trajectory, through
  a DEGRADED line at the top of its context. Silent failure hid a total
  outage for 30 hours and a lexical week for six days; the trajectory
  cannot see its own thinning from inside, so the user has to be told too.
- FAST. One cached MCP session per (server, harness-session); hard timeout.
- SILENT WHEN SILENT. An empty payload prints nothing — the hook returning
  nothing IS the discipline of silence.

Shadow mode (--shadow): compute everything, inject nothing. Every result —
payload, seed_mode, counts, suppression, timings — is appended as one JSON
line to KENNING_ENCOUNTER_INFUSE_SHADOW_LOG (default ~/.claude/kenning_encounter-infuse-shadow.jsonl)
and stdout stays empty. This is the zero-constitutive-risk dress rehearsal:
run the wired hooks in shadow for a few wakings against the live substrate
to observe seed quality, real-graph latency, and the
parked-tensions register at scale, BEFORE the first governed payload is
allowed to condition a live write.

Observation log (KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG=<path>): when set, EVERY call —
injected or silent — appends the same full record (plus an `injected`
flag) to <path>, independent of shadow mode. This is the out-of-band
measurement stream for the infusion experiment: the harness transcript
already records what was injected (additionalContext is saved into the
session transcript), but only this log records the silences — gated calls,
silent calls, timings, suppression counts — and silence is a
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
        f"kenning_encounter_infuse_hook: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
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

DEFAULT_URL = os.environ.get("KENNING_ENCOUNTER_MCP_URL", "http://127.0.0.1:8003/mcp/")
TIMEOUT_S = float(os.environ.get("KENNING_ENCOUNTER_INFUSE_TIMEOUT", "10.0"))
# Trajectory (v0.9.0, the meaning matcher): full-mode selection reads the
# session's arc, not a snapshot — meaning is temporal, and a 13-word prompt
# under-determines it. The hook parses the harness transcript HOST-SIDE
# (the server runs in a container without access to ~/.claude) and passes
# the last N user turns alongside the prompt. 0 disables.
TRAJECTORY_TURNS = int(os.environ.get("KENNING_ENCOUNTER_INFUSE_TRAJECTORY_TURNS", "7"))
SHADOW_LOG = os.environ.get(
    "KENNING_ENCOUNTER_INFUSE_SHADOW_LOG",
    os.path.expanduser("~/.claude/kenning_encounter-infuse-shadow.jsonl"),
)
OBSERVE_LOG = os.environ.get("KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG")  # unset = no observation

# ---------------------------------------------------------------------------
# The switch: infusion belongs to a prompt somebody WROTE.
#
# THE PARADIGM SPLIT, measured on a live headless trajectory (2026-08-07). In an
# interactive session each prompt carries new intentionality from a second
# party, which is exactly what the per-prompt injection point is for. A
# scheduled run is a MONOLOGUE: one fixed trigger string, then autonomous
# work. There is no second party, so no frame boundary is crossed and the
# injection point has nothing to carry.
#
# Left ungated, infusion in a scheduled run is not merely noisy, it builds
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
# earlier.
#
# KENNING_ENCOUNTER_INFUSE is a SWITCH, not a mode — there is only one thing to infuse:
#   auto (default) — the entrypoint gate above: on for a written prompt,
#                    off for a scheduled trigger.
#   on             — infuse regardless. For an invocation the operator KNOWS
#                    carries a second frame but whose harness we misread:
#                    an unrecognised runner, or a run whose prompt is another
#                    trajectory's message rather than a cron string.
#   off            — never infuse.
# The switch says WHETHER; the recorded skip_reason says WHY. Per-process by
# construction, so the same deployment can answer differently per invocation.
INTERACTIVE_ENTRYPOINTS = frozenset({"cli"})
INFUSE_SWITCH = os.environ.get("KENNING_ENCOUNTER_INFUSE", "auto").lower()


def infusion_allowed(env: dict[str, str] | None = None) -> tuple[bool, str]:
    """(allowed, reason) for an infusion call under the current harness.

    Returns the reason either way so the observation stream can record WHY a
    call was skipped. A guard that suppresses silently is indistinguishable
    from a guard that is broken, and the observe log is what tells them apart.
    """
    env = os.environ if env is None else env
    switch = (env.get("KENNING_ENCOUNTER_INFUSE", INFUSE_SWITCH) or "auto").lower()
    if switch == "on":
        return True, "switch=on"
    if switch == "off":
        return False, "switch=off"
    entrypoint = env.get("CLAUDE_CODE_ENTRYPOINT", "")
    if not entrypoint:
        return False, "no CLAUDE_CODE_ENTRYPOINT — harness unrecognised, degrading to silence"
    if entrypoint in INTERACTIVE_ENTRYPOINTS:
        return True, f"interactive entrypoint ({entrypoint})"
    return False, f"non-interactive entrypoint ({entrypoint}) — scheduled run is a monologue"


# ---------------------------------------------------------------------------
# Degradation: announced, never swallowed.
#
# The trajectory cannot detect its own infusion thinning: a poorer context
# simply becomes the whole context. The user can — a user who has worked with
# the bound trajectory has that as their baseline, and a stateless-feeling
# answer is the symptom. But a symptom is slow; the lexical fallback of
# 2026-09-22..28 was felt for days before it was named. So the hook, which
# sees every call's outcome, says so on the turn it happens, to both.
#
# Degraded means MECHANICALLY degraded: no result, or seeds not meaning-
# matched. It never means "this payload was not meaningful" — that is the
# user's judgment inside the window, never the hook's. Healthy silence (the
# matcher ran and nothing bore on the prompt) is not degradation.
ERROR_MSG_CHARS = 200


def _short(exc_or_text: object, limit: int = ERROR_MSG_CHARS) -> str:
    """One line, capped: exception shape (status, URL, reason), never node
    content — this text reaches the user's screen and the observe log."""
    s = " ".join(str(exc_or_text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def degradation(result: dict) -> str | None:
    """The reason a successful call's infusion is degraded, or None.

    Only the seed channel is judged. A result with no `selection_channel`
    comes from a pre-v0.9.0 server that has no matcher to fall back from,
    and reads as healthy rather than inventing an alarm.
    """
    channel = result.get("selection_channel")
    if channel is None or channel == "meaning":
        return None
    reason = (result.get("matcher") or {}).get("fallback_reason")
    return _short(
        f"seeds are LEXICAL, not meaning-matched ({reason or 'no reason given'})"
    )


def degraded_output(event: str, reason: str, payload: str = "") -> dict:
    """The hook's JSON for a degraded call: a systemMessage the user sees,
    and a first line the trajectory sees above whatever payload survived."""
    notice = (
        f"[INFUSION DEGRADED — {reason}. Your substrate is not reaching you "
        "at full strength this turn. Tell the user before anything else, "
        "so it can be fixed now.]"
    )
    return {
        "systemMessage": f"⚠ Kenning Encounter infusion DEGRADED: {reason}",
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": f"{notice}\n{payload}" if payload else notice,
        },
    }


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
    return os.path.join(tempfile.gettempdir(), f"kenning_encounter-infuse-{key}.session")


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
                "clientInfo": {"name": "kenning_encounter-infuse-hook", "version": "0.10.0"},
            },
        },
        None,
    )
    if sid:
        _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    return sid


def _call_infuse(
    url: str, sid: str | None, text: str,
    trajectory: list[str] | None = None,
    harness_session: str | None = None,
) -> dict:
    """One infuse call. `sid` is the MCP transport session (dies with the
    connection); `harness_session` is the durable locus identity, and the two
    are not interchangeable — this hook has always known the second, to key
    its own transport-session cache file, and until v0.12.4 never sent it. So
    the server keyed the renewal ledger on the transport session, and a
    restart mid-waking re-delivered at full weight into a conversation that
    already held the material."""
    arguments: dict = {"text": text}
    if trajectory:
        arguments["trajectory"] = trajectory
    if harness_session:
        arguments["session_id"] = harness_session
    msg, _ = _post(
        url,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "infuse", "arguments": arguments},
        },
        sid,
    )
    if not msg or "error" in msg:
        raise RuntimeError(str(msg))
    content = msg["result"]["content"][0]["text"]
    return json.loads(content)


def _infuse_with_session(
    url: str, sid: str | None, text: str,
    trajectory: list[str] | None = None,
    harness_session: str | None = None,
) -> tuple[dict, str | None]:
    """Call infuse on the cached transport session, re-initializing once if
    it is gone. Returns (result, the transport session id to cache).

    A stateless server (NEO4J_MCP_SERVER_STATELESS) answers initialize with
    no Mcp-Session-Id, and that is an answer, not a failure: the call then
    goes out with no session header. Until v0.16.3 the hook read the missing
    id as "initialize failed", so from the v0.16.0 cutover every session
    without a pre-stateless cache file got nothing but a RuntimeError in the
    observe log. Continuity never depended on the transport session anyway;
    the durable identity travels as harness_session."""
    if sid:
        try:
            return _call_infuse(url, sid, text, trajectory, harness_session), sid
        except Exception:
            pass  # session lost (server restart): one re-initialize
    sid = _initialize(url)
    return _call_infuse(url, sid, text, trajectory, harness_session), sid


def _user_turns(transcript_path: str, n: int) -> list[str]:
    """The last n real user turns from the harness transcript (JSONL).

    Stdlib mirror of the server's own parser (the hook stays standalone):
    tool_result-only entries are not user turns, meta entries and
    hook-injected <system-reminder> blocks are stripped — the trajectory is
    what the person said, not what the harness wrapped around it. Any
    failure returns [] and the call proceeds prompt-only (fail-silent)."""
    turns: list[str] = []
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("type") != "user" or entry.get("isMeta"):
                    continue
                message = entry.get("message") or {}
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    texts = [content]
                elif isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                else:
                    texts = []
                cleaned = [
                    t.strip() for t in texts
                    if t.strip() and not t.lstrip().startswith("<system-reminder")
                ]
                if cleaned:
                    turns.append("\n".join(cleaned))
    except OSError:
        return []
    return turns[-n:]


def _focal_text(hook_input: dict) -> str:
    """The arriving present: the prompt, and only ever the prompt."""
    return str(hook_input.get("prompt", ""))


def _append_log(path: str, record: dict) -> None:
    """Append one observation record; never let logging break the hook."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


class _SilentParser(argparse.ArgumentParser):
    """argparse that cannot block the turn.

    THE HOLE THIS CLOSES, found live: argparse exits 2 on a bad or unknown
    flag, and 2 is exactly the harness's BLOCKING exit code for
    UserPromptSubmit — so a stale `--mode full` in a settings.json left over
    from the two-channel era did not degrade to silence, it wedged the
    session ("UserPromptSubmit operation blocked by hook") for every prompt
    until someone edited the file. The hook's own stated invariant is FAIL
    SILENT: any error -> exit 0, no output. argparse ran before any of the
    handling that honours it. It no longer can.
    """

    def error(self, message: str):
        sys.stderr.write(f"kenning_encounter-infuse-hook: {message}\n")
        raise SystemExit(0)


def main() -> int:
    global TIMEOUT_S
    parser = _SilentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Request timeout in seconds (overrides KENNING_ENCOUNTER_INFUSE_TIMEOUT; "
             "default 10 — sized to the cold first call, not the warm repeat).",
    )
    parser.add_argument(
        "--shadow", action="store_true",
        help="Compute and log the infusion, inject nothing.",
    )
    # Unknown flags are IGNORED, never fatal — a wiring written against an
    # older version must not wedge the harness. But they are RECORDED with
    # the call (below), because a guard that swallows silently is
    # indistinguishable from a guard that is broken.
    args, unknown_args = parser.parse_known_args()
    if args.timeout is not None:
        TIMEOUT_S = args.timeout

    hook_input = json.load(sys.stdin)

    # Gate full mode BEFORE any network call: a scheduled run's trigger is a
    # ritual string, and infusing on it drones a permanent core (see
    # infusion_allowed). The skip is RECORDED, never silent — an invisible
    # guard is the failure mode this whole apparatus keeps rediscovering.
    allowed, why = infusion_allowed()
    if not allowed:
        if OBSERVE_LOG:
            _append_log(OBSERVE_LOG, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "session_id": str(hook_input.get("session_id", "")),
                "event": hook_input.get("hook_event_name", ""),
                "injected": False,
                "skipped": True,
                "skip_reason": why,
                "unknown_args": unknown_args,
            })
        return 0

    text = _focal_text(hook_input)
    if not text.strip():
        return 0

    # Trajectory: the last N user turns before this prompt,
    # parsed host-side from the harness transcript. The transcript's tail
    # usually IS this prompt — drop the duplicate so the server sees the
    # current prompt exactly once, last.
    trajectory: list[str] = []
    if TRAJECTORY_TURNS > 0:
        tpath = str(hook_input.get("transcript_path", "") or "")
        if tpath:
            trajectory = _user_turns(tpath, TRAJECTORY_TURNS + 1)
            if trajectory and trajectory[-1].strip() == text.strip():
                trajectory = trajectory[:-1]
            trajectory = trajectory[-TRAJECTORY_TURNS:]

    # The join anchor for the observation stream: sha256 of the stripped
    # focal text, first 16 hex. The invocation scorer hashes each transcript
    # prompt identically and joins on it — a missed call then reads as a
    # counted gap instead of shearing every index-pair after it.
    psha = hashlib.sha256(text.strip().encode()).hexdigest()[:16]

    harness_session = str(hook_input.get("session_id", "")) or None
    cache = _session_cache_path(args.url, harness_session or "")
    sid = None
    if os.path.exists(cache):
        with open(cache) as f:
            sid = f.read().strip() or None

    try:
        result, sid = _infuse_with_session(
            args.url, sid, text, trajectory, harness_session
        )
    except Exception as exc:
        # Never block the turn — but never swallow it either. The observation
        # stream records the attempt (an unlogged failure is what made the
        # positional join unrepairable), and error_msg says WHICH failure:
        # the type alone left "initialize failed" and a server-side error
        # looking identical. The same text is the degraded notice's reason.
        error_msg = _short(exc) or type(exc).__name__
        if OBSERVE_LOG:
            _append_log(OBSERVE_LOG, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "session_id": str(hook_input.get("session_id", "")),
                "event": hook_input.get("hook_event_name", ""),
                "prompt_sha": psha,
                "error": type(exc).__name__,
                "error_msg": error_msg,
                "degraded": f"no infusion: {error_msg}",
            })
        if not args.shadow:
            print(json.dumps(degraded_output(
                hook_input.get("hook_event_name", "UserPromptSubmit"),
                f"no infusion reached this turn ({type(exc).__name__}: {error_msg})",
            )))
        return 0

    if sid:
        with open(cache, "w") as f:
            f.write(sid)

    payload = result.get("payload", "") or ""
    degraded = degradation(result)

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session_id": str(hook_input.get("session_id", "")),
        "event": hook_input.get("hook_event_name", ""),
        "prompt_sha": psha,
        "unknown_args": unknown_args,
        "silence": result.get("silence"),
        "seed_mode": result.get("seed_mode"),
        "rank_mode": result.get("rank_mode"),
        "seed_terms": result.get("seed_terms"),
        "counts": result.get("counts"),
        "suppressed": result.get("suppressed"),
        "renewal": result.get("renewal"),
        # v0.8.0: the assembly accounting — selected / assembled /
        # delivered plus the blind-spot log — and the server version, the
        # seam marker any analysis must stratify on across the progression
        # cutover. Absent on pre-0.8.0 servers; .get keeps old servers clean.
        "assembly": result.get("assembly"),
        "server_version": result.get("server_version"),
        # v0.9.0: which seed channel ran (meaning vs lexical_fallback) and
        # the matcher's own metadata — the seam marker for the selection
        # cutover, stratifiable exactly as A9 stratifies on server_version.
        "selection_channel": result.get("selection_channel"),
        "matcher": result.get("matcher"),
        # v0.17.0: which voice delivered and which was meant to lead —
        # the partition every in-window judgment of the two voices is split
        # by. match_id joins this line to the server's shadow log.
        "matcher_voice": (result.get("matcher") or {}).get("voice"),
        "matcher_lead": (result.get("matcher") or {}).get("lead"),
        "match_id": (result.get("matcher") or {}).get("match_id"),
        "trajectory_turns": len(trajectory),
        "timings_ms": result.get("timings_ms"),
        "payload_chars": len(payload),
        "payload": payload,
        "injected": bool(payload) and not args.shadow,
        "degraded": degraded,
    }
    if OBSERVE_LOG:
        _append_log(OBSERVE_LOG, record)

    if args.shadow:
        _append_log(SHADOW_LOG, record)
        return 0

    event = hook_input.get("hook_event_name", "UserPromptSubmit")
    if degraded:
        print(json.dumps(degraded_output(event, degraded, payload)))
    elif payload:
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
    except Exception as exc:
        # Never block the turn (exit 0), and never swallow a failure: an
        # exception outside the call itself (unreadable stdin, a transcript
        # the parser chokes on) is still a turn with no infusion. A shadow
        # run injects nothing, so it announces nothing either.
        if "--shadow" not in sys.argv:
            try:
                print(json.dumps(degraded_output(
                    "UserPromptSubmit",
                    f"the infusion hook crashed ({type(exc).__name__}: {_short(exc)})",
                )))
            except Exception:
                pass
        sys.exit(0)
