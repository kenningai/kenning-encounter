"""Unit tests for scripts/agent_memory_infuse_hook.py — the harness-side hook client.

The property under test is the FULL-MODE GATE, and it exists because of a
measured pathology rather than a preference: in a scheduled (headless) run
the trigger prompt is a fixed ritual string, so Extract yields identical
seeds every waking; the renewal ledger is session-scoped and a waking IS a
session, so the ledger is cold every time and the same bodies arrive at full
weight forever. The renewal economy cannot reach across the session
boundary, and a permanent payload core is the gravity well mechanized.

The gate's two load-bearing properties:
  1. It detects INTERACTIVE positively, so an unrecognised harness degrades
     toward SILENCE rather than toward drone.
  2. Every suppression is RECORDED. A guard that suppresses silently is
     indistinguishable from a guard that is broken — this project has
     shipped that failure three times (assembler ladder, phantom ledger,
     permutation-invariant null), and the observation stream is where the
     fourth gets caught.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "scripts" / "agent_memory_infuse_hook.py"
_spec = importlib.util.spec_from_file_location("agent_memory_infuse_hook", _SCRIPT)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


class TestFullModeGate:
    """Pure decision function — no network, no harness."""

    def test_interactive_cli_is_allowed(self):
        allowed, why = hook.full_mode_allowed({"CLAUDE_CODE_ENTRYPOINT": "cli"})
        assert allowed is True
        assert "interactive" in why

    def test_headless_print_mode_is_blocked(self):
        # `claude -p` — the scheduled/cron shape. This is the case the gate
        # was built for.
        allowed, why = hook.full_mode_allowed({"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"})
        assert allowed is False
        assert "monologue" in why

    def test_unknown_harness_degrades_to_silence_not_drone(self):
        # THE DIRECTION THAT MATTERS: an SDK or runner we cannot enumerate
        # must lose injection, never gain an unbounded permanent core.
        for env in ({}, {"CLAUDE_CODE_ENTRYPOINT": "sdk-py"},
                    {"CLAUDE_CODE_ENTRYPOINT": "some-future-runner"}):
            allowed, _ = hook.full_mode_allowed(env)
            assert allowed is False, f"{env} must degrade to silence"

    def test_operator_override_forces_on(self):
        allowed, why = hook.full_mode_allowed(
            {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli", "AGENT_MEMORY_INFUSE_FULL": "always"})
        assert allowed is True and why == "policy=always"

    def test_operator_override_forces_off(self):
        allowed, why = hook.full_mode_allowed(
            {"CLAUDE_CODE_ENTRYPOINT": "cli", "AGENT_MEMORY_INFUSE_FULL": "never"})
        assert allowed is False and why == "policy=never"

    def test_reason_is_returned_on_both_paths(self):
        # The reason string is not decoration: it is what lands in the
        # observation stream and makes a suppression legible.
        for env in ({"CLAUDE_CODE_ENTRYPOINT": "cli"},
                    {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}):
            _, why = hook.full_mode_allowed(env)
            assert why and isinstance(why, str)


def _run(mode, env_extra, tmp_path, stdin_obj):
    """Invoke the hook as the harness does: JSON on stdin, env, exit code."""
    import os
    log = tmp_path / "observe.jsonl"
    env = dict(os.environ)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.update(env_extra)
    env["AGENT_MEMORY_INFUSE_OBSERVE_LOG"] = str(log)
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--mode", mode,
         "--url", "http://127.0.0.1:59999/api/mcp/", "--timeout", "1"],
        input=json.dumps(stdin_obj), capture_output=True, text=True, env=env,
    )
    records = []
    if log.exists():
        records = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    return proc, records


class TestGateEndToEnd:
    STDIN = {"session_id": "s1", "hook_event_name": "UserPromptSubmit",
             "prompt": "You are waking. Run your EERRS cycle."}

    def test_blocked_call_is_silent_to_harness_but_recorded(self, tmp_path):
        proc, recs = _run("full", {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"},
                          tmp_path, self.STDIN)
        assert proc.returncode == 0
        assert proc.stdout == ""            # nothing reaches the agent
        assert len(recs) == 1, "the suppression MUST be observable"
        r = recs[0]
        assert r["skipped"] is True
        assert r["injected"] is False
        assert "monologue" in r["skip_reason"]

    def test_gate_short_circuits_before_the_network(self, tmp_path):
        # The URL points at a closed port with a 1s timeout. If the gate ran
        # after the call, this test would pay that timeout; the gate must
        # decide first, so a blocked scheduled run costs nothing.
        import time
        t0 = time.monotonic()
        proc, _ = _run("full", {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"},
                       tmp_path, self.STDIN)
        assert proc.returncode == 0
        assert time.monotonic() - t0 < 1.0

    def test_delta_is_never_gated(self, tmp_path):
        # Delta carries the agent's own unfolding and is the ONLY channel a
        # scheduled run has. Gating it would disable infusion entirely there.
        proc, recs = _run("delta", {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}, tmp_path,
                          {"session_id": "s1", "hook_event_name": "PostToolUse",
                           "tool_response": {"text": "the VIP distributes to node-02"}})
        assert proc.returncode == 0
        assert not any(r.get("skipped") for r in recs), \
            "delta must not be gated by entrypoint"

    def test_interactive_full_is_not_skipped(self, tmp_path):
        # It will still fail to reach the closed port and fail silent, but it
        # must NOT be recorded as gate-skipped — the distinction between
        # 'suppressed by policy' and 'server unreachable' has to survive.
        proc, recs = _run("full", {"CLAUDE_CODE_ENTRYPOINT": "cli"},
                          tmp_path, self.STDIN)
        assert proc.returncode == 0
        assert not any(r.get("skipped") for r in recs)
