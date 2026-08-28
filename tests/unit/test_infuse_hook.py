"""Unit tests for scripts/kenning_encounter_infuse_hook.py — the harness-side hook client.

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
_SCRIPT = _ROOT / "scripts" / "kenning_encounter_infuse_hook.py"
_spec = importlib.util.spec_from_file_location("kenning_encounter_infuse_hook", _SCRIPT)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


class TestInfusionSwitch:
    """Pure decision function — no network, no harness."""

    def test_interactive_cli_is_allowed(self):
        allowed, why = hook.infusion_allowed({"CLAUDE_CODE_ENTRYPOINT": "cli"})
        assert allowed is True
        assert "interactive" in why

    def test_headless_print_mode_is_blocked(self):
        # `claude -p` — the scheduled/cron shape. This is the case the gate
        # was built for.
        allowed, why = hook.infusion_allowed({"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"})
        assert allowed is False
        assert "monologue" in why

    def test_unknown_harness_degrades_to_silence_not_drone(self):
        # THE DIRECTION THAT MATTERS: an SDK or runner we cannot enumerate
        # must lose injection, never gain an unbounded permanent core.
        for env in ({}, {"CLAUDE_CODE_ENTRYPOINT": "sdk-py"},
                    {"CLAUDE_CODE_ENTRYPOINT": "some-future-runner"}):
            allowed, _ = hook.infusion_allowed(env)
            assert allowed is False, f"{env} must degrade to silence"

    def test_operator_override_forces_on(self):
        allowed, why = hook.infusion_allowed(
            {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli", "KENNING_ENCOUNTER_INFUSE": "on"})
        assert allowed is True and why == "switch=on"

    def test_operator_override_forces_off(self):
        allowed, why = hook.infusion_allowed(
            {"CLAUDE_CODE_ENTRYPOINT": "cli", "KENNING_ENCOUNTER_INFUSE": "off"})
        assert allowed is False and why == "switch=off"

    def test_reason_is_returned_on_both_paths(self):
        # The reason string is not decoration: it is what lands in the
        # observation stream and makes a suppression legible.
        for env in ({"CLAUDE_CODE_ENTRYPOINT": "cli"},
                    {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}):
            _, why = hook.infusion_allowed(env)
            assert why and isinstance(why, str)


def _run(env_extra, tmp_path, stdin_obj):
    """Invoke the hook as the harness does: JSON on stdin, env, exit code."""
    import os
    log = tmp_path / "observe.jsonl"
    env = dict(os.environ)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.update(env_extra)
    env["KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG"] = str(log)
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT),
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
        proc, recs = _run({"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"},
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
        proc, _ = _run({"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"},
                       tmp_path, self.STDIN)
        assert proc.returncode == 0
        assert time.monotonic() - t0 < 1.0

    def test_switch_on_reaches_a_headless_run(self, tmp_path):
        # The gate keys on the HARNESS (entrypoint), which is a proxy for
        # "a second frame is present" — good for the human case, blind to
        # any other. A headless run whose prompt is ANOTHER AGENT's message
        # does cross a frame boundary, and the caller is the only party that
        # knows it. KENNING_ENCOUNTER_INFUSE=on is how that judgment is declared; without
        # this the sibling channel would be silenced by the cron proxy.
        proc, recs = _run({"CLAUDE_CODE_ENTRYPOINT": "sdk-cli",
                           "KENNING_ENCOUNTER_INFUSE": "on"}, tmp_path, self.STDIN)
        assert proc.returncode == 0
        assert not any(r.get("skipped") for r in recs), \
            "a declared second frame must not be gated by the harness proxy"

    def test_interactive_full_is_not_skipped(self, tmp_path):
        # It will still fail to reach the closed port and fail silent, but it
        # must NOT be recorded as gate-skipped — the distinction between
        # 'suppressed by policy' and 'server unreachable' has to survive.
        proc, recs = _run({"CLAUDE_CODE_ENTRYPOINT": "cli"},
                          tmp_path, self.STDIN)
        assert proc.returncode == 0
        assert not any(r.get("skipped") for r in recs)


class TestUserTurns:
    """Trajectory extraction (v0.9.0): the hook parses the harness
    transcript HOST-SIDE and passes the last N user turns — the server's
    container cannot see ~/.claude, so this is the only live path."""

    def _write(self, tmp_path, entries):
        p = tmp_path / "transcript.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in entries))
        return str(p)

    def test_last_n_user_text_turns_in_order(self, tmp_path):
        entries = [
            {"type": "user", "message": {"role": "user", "content": "one"}},
            {"type": "assistant", "message": {"role": "assistant",
                                              "content": "reply"}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "two"}]}},
            {"type": "user", "message": {"role": "user", "content": "three"}},
        ]
        assert hook._user_turns(self._write(tmp_path, entries), 2) == [
            "two", "three",
        ]

    def test_tool_results_meta_and_reminders_are_not_turns(self, tmp_path):
        entries = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": "output"}]}},
            {"type": "user", "isMeta": True,
             "message": {"role": "user", "content": "meta"}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text",
                 "text": "<system-reminder>injected</system-reminder>"},
                {"type": "text", "text": "the real ask"}]}},
        ]
        assert hook._user_turns(self._write(tmp_path, entries), 5) == [
            "the real ask",
        ]

    def test_failures_return_empty_never_raise(self, tmp_path):
        assert hook._user_turns(str(tmp_path / "absent.jsonl"), 3) == []
        p = tmp_path / "garbage.jsonl"
        p.write_text("not json at all\n{broken")
        assert hook._user_turns(str(p), 3) == []


class TestStaleWiringCannotBlockTheTurn:
    """A settings.json written against an older version must not wedge the harness.

    argparse exits 2 on an unknown flag, and 2 is the harness's BLOCKING exit
    code for UserPromptSubmit. A `--mode full` left over from the two-channel
    era therefore did not degrade to silence — it blocked every prompt in the
    session. Found live on the author's own machine, which is exactly where
    the portability defects keep being found.
    """

    STDIN = {"session_id": "s1", "hook_event_name": "UserPromptSubmit",
             "prompt": "a written prompt from another frame"}

    def _run_argv(self, extra_argv, env_extra, tmp_path):
        import os
        log = tmp_path / "observe.jsonl"
        env = dict(os.environ)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        env.update(env_extra)
        env["KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG"] = str(log)
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), *extra_argv,
             "--url", "http://127.0.0.1:59999/api/mcp/", "--timeout", "1"],
            input=json.dumps(self.STDIN), capture_output=True, text=True, env=env,
        )
        recs = []
        if log.exists():
            recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        return proc, recs

    def test_retired_mode_flag_does_not_block(self, tmp_path):
        proc, _ = self._run_argv(
            ["--mode", "full"], {"CLAUDE_CODE_ENTRYPOINT": "cli"}, tmp_path)
        assert proc.returncode == 0, (
            "exit 2 is the harness's blocking code — a stale flag must never "
            f"reach it (stderr: {proc.stderr!r})")

    def test_unparseable_flag_does_not_block(self, tmp_path):
        proc, _ = self._run_argv(
            ["--timeout", "not-a-number"], {"CLAUDE_CODE_ENTRYPOINT": "cli"}, tmp_path)
        assert proc.returncode == 0

    def test_the_ignored_flag_is_recorded_not_swallowed(self, tmp_path):
        # Gated path so the record lands without a live server.
        proc, recs = self._run_argv(
            ["--mode", "delta"], {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}, tmp_path)
        assert proc.returncode == 0
        assert recs, "the call must still be observable"
        assert recs[0]["unknown_args"] == ["--mode", "delta"], \
            "a swallowed flag is indistinguishable from a broken guard"
