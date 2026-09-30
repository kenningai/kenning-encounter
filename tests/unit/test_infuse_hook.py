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
     indistinguishable from a guard that is broken, and the observation
     stream is what tells them apart.
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


class TestTheDurableIdentityReachesTheLedger:
    """v0.12.4 — the hook has always known the harness session id, and used it
    to key its own transport-session cache file. It never sent it.

    So the server keyed the renewal ledger on the MCP TRANSPORT session, which
    dies with the connection where the locus does not. v0.12.0 made the locus
    survive a restart and a /resume; the ledger tracking what that locus had
    already been told did not follow. The consequence is the pathology this
    module's own docstring describes, arriving one level in: a mid-waking
    restart re-delivers the same bodies at full weight into a conversation
    that still holds them."""

    def _sent(self, harness_session):
        captured = {}

        def fake_post(url, payload, sid):
            if payload.get("method") == "tools/call":
                captured.update(payload["params"]["arguments"])
            return {"result": {"content": [{"text": "{}"}]}}, sid

        original = hook._post
        hook._post = fake_post
        try:
            hook._call_infuse(
                "http://x", "transport-1", "a prompt",
                None, harness_session,
            )
        finally:
            hook._post = original
        return captured

    def test_the_harness_session_is_sent_as_session_id(self):
        assert self._sent("harness-abc")["session_id"] == "harness-abc"

    def test_it_is_not_the_transport_session(self):
        """The two are different identities and the bug was conflating them."""
        assert self._sent("harness-abc")["session_id"] != "transport-1"

    def test_absent_identity_sends_no_key_rather_than_an_empty_one(self):
        """A client with no harness session degrades to the transport-keyed
        ledger — honest, and what v0.12.3 says a non-Claude-Code harness
        gets. An empty string would collapse every such client into one
        shared ledger, which is worse than having none."""
        assert "session_id" not in self._sent(None)
        assert "session_id" not in self._sent("")


class TestAStatelessServerIsNotAFailedOne:
    """v0.16.3 — a stateless server answers initialize with no
    Mcp-Session-Id. The hook read that as "initialize failed" and raised, so
    every session that had no transport id cached from before the v0.16.0
    cutover got a RuntimeError on every prompt and no infusion at all."""

    def _run(self, cached_sid, init_sid, stale_cached=False):
        sent = []

        def fake_post(url, payload, sid):
            method = payload.get("method")
            sent.append((method, sid))
            if method == "initialize":
                return {"result": {}}, init_sid
            if method == "tools/call":
                if stale_cached and sid == cached_sid:
                    return {"error": {"message": "Session not found"}}, sid
                return {"result": {"content": [{"text": '{"payload": "p"}'}]}}, sid
            return None, sid

        original = hook._post
        hook._post = fake_post
        try:
            result, sid = hook._infuse_with_session(
                "http://x", cached_sid, "a prompt", None, "harness-abc",
            )
        finally:
            hook._post = original
        return result, sid, sent

    def test_no_session_id_from_initialize_still_infuses(self):
        result, sid, sent = self._run(cached_sid=None, init_sid=None)
        assert result == {"payload": "p"}
        assert sid is None, "there is nothing to cache"
        assert ("tools/call", None) in sent, "the call goes out headerless"

    def test_a_stateful_server_still_gets_its_session(self):
        result, sid, sent = self._run(cached_sid=None, init_sid="t-1")
        assert result == {"payload": "p"}
        assert sid == "t-1"
        assert ("tools/call", "t-1") in sent

    def test_a_cached_session_is_used_without_initializing(self):
        _, sid, sent = self._run(cached_sid="t-0", init_sid="t-1")
        assert sid == "t-0"
        assert [m for m, _ in sent] == ["tools/call"]

    def test_a_lost_session_reinitializes_once(self):
        result, sid, sent = self._run(
            cached_sid="t-0", init_sid="t-1", stale_cached=True,
        )
        assert result == {"payload": "p"}
        assert sid == "t-1"
        assert [m for m, _ in sent].count("initialize") == 1

    def test_a_real_error_still_raises(self):
        """The gate says no too: an error from the server is still an error,
        and main() logs it rather than injecting."""
        def fake_post(url, payload, sid):
            if payload.get("method") == "tools/call":
                return {"error": {"message": "boom"}}, sid
            return {"result": {}}, None

        original = hook._post
        hook._post = fake_post
        try:
            with pytest.raises(RuntimeError):
                hook._infuse_with_session("http://x", None, "a prompt")
        finally:
            hook._post = original


class TestDegradationReachesBothReaders:
    """v0.16.3 — a degraded infusion is announced to the user (systemMessage)
    and to the trajectory (a first line of context), on the turn it happens.

    The trajectory cannot see its own thinning from inside: a poorer context
    simply becomes the whole context. The lexical fallback of 2026-09-22..28
    was felt by the user for days before it was named, and the stateless
    cutover left 30 hours with no infusion at all and nothing on anyone's
    screen. Two-sided: every healthy shape must stay quiet."""

    MEANING = {"selection_channel": "meaning", "payload": "[p]",
               "matcher": {"channel": "meaning"}}
    LEXICAL = {"selection_channel": "lexical", "payload": "[p]",
               "matcher": {"channel": "lexical_fallback",
                           "fallback_reason": "HTTP 402: credits depleted"}}

    def test_meaning_is_healthy(self):
        assert hook.degradation(self.MEANING) is None

    def test_healthy_silence_is_not_degradation(self):
        assert hook.degradation({**self.MEANING, "payload": ""}) is None

    def test_a_pre_matcher_server_is_not_an_alarm(self):
        assert hook.degradation({"payload": "[p]"}) is None

    def test_lexical_fallback_is_degraded_with_its_reason(self):
        why = hook.degradation(self.LEXICAL)
        assert why and "LEXICAL" in why and "402" in why

    def test_the_output_reaches_both_readers(self):
        out = hook.degraded_output("UserPromptSubmit", "a reason", "[p]")
        assert "DEGRADED" in out["systemMessage"]
        assert "a reason" in out["systemMessage"]
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert ctx.splitlines()[0].startswith("[INFUSION DEGRADED — a reason")
        assert ctx.endswith("[p]"), "the surviving payload still arrives"

    def test_the_reason_is_capped_to_one_line(self):
        s = hook._short("a\nb " + "x" * 500)
        assert "\n" not in s and len(s) <= hook.ERROR_MSG_CHARS

    def test_an_unreachable_server_is_announced_not_swallowed(self, tmp_path):
        # _run points at a closed port: the 30-hour outage in miniature.
        proc, recs = _run({"CLAUDE_CODE_ENTRYPOINT": "cli"}, tmp_path,
                          TestGateEndToEnd.STDIN)
        assert proc.returncode == 0, "degraded never means blocked"
        out = json.loads(proc.stdout)
        assert "DEGRADED" in out["systemMessage"]
        assert "no infusion" in out["hookSpecificOutput"]["additionalContext"]
        assert recs[0]["error_msg"], "the log says WHICH failure"
        assert recs[0]["degraded"].startswith("no infusion")

    def test_a_crash_outside_the_call_is_announced(self, tmp_path):
        import os
        env = dict(os.environ, CLAUDE_CODE_ENTRYPOINT="cli")
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), "--url",
             "http://127.0.0.1:59999/api/mcp/", "--timeout", "1"],
            input="not json", capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 0
        assert "DEGRADED" in json.loads(proc.stdout)["systemMessage"]


class TestTheWholeHookAgainstAStatelessServer:
    """The wiring, not the parts: main() against a fake STATELESS server
    (initialize returns no Mcp-Session-Id), because the orchestration glue is
    exactly where the v0.16.0 cutover broke while every unit held."""

    @pytest.fixture
    def server(self):
        import http.server
        import threading

        state = {"result": {}}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body.get("method") == "tools/call":
                    msg = {"jsonrpc": "2.0", "id": body["id"], "result": {
                        "content": [{"text": json.dumps(state["result"])}]}}
                elif "id" in body:
                    msg = {"jsonrpc": "2.0", "id": body["id"], "result": {}}
                else:
                    msg = None
                data = json.dumps(msg).encode() if msg else b""
                self.send_response(200 if msg else 202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()  # no Mcp-Session-Id: stateless
                self.wfile.write(data)

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{srv.server_port}/api/mcp/", state
        srv.shutdown()

    def _hook(self, url, tmp_path):
        import os
        env = dict(os.environ, CLAUDE_CODE_ENTRYPOINT="cli",
                   KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG=str(tmp_path / "o.jsonl"),
                   TMPDIR=str(tmp_path))
        stdin = {**TestGateEndToEnd.STDIN, "session_id": f"s-{tmp_path.name}"}
        return subprocess.run(
            [sys.executable, str(_SCRIPT), "--url", url, "--timeout", "2"],
            input=json.dumps(stdin), capture_output=True, text=True, env=env,
        )

    def test_meaning_injects_quietly(self, server, tmp_path):
        url, state = server
        state["result"] = {"selection_channel": "meaning", "payload": "[p]"}
        out = json.loads(self._hook(url, tmp_path).stdout)
        assert "systemMessage" not in out
        assert out["hookSpecificOutput"]["additionalContext"] == "[p]"

    def test_lexical_is_announced_and_still_delivered(self, server, tmp_path):
        url, state = server
        state["result"] = {"selection_channel": "lexical", "payload": "[p]",
                           "matcher": {"fallback_reason": "HTTP 402"}}
        out = json.loads(self._hook(url, tmp_path).stdout)
        assert "HTTP 402" in out["systemMessage"]
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert ctx.startswith("[INFUSION DEGRADED") and ctx.endswith("[p]")

    def test_healthy_silence_prints_nothing(self, server, tmp_path):
        url, state = server
        state["result"] = {"selection_channel": "meaning", "payload": ""}
        assert self._hook(url, tmp_path).stdout == ""
