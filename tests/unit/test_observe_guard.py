"""The observe-log guard: two-sided before it is trusted.

It must refuse the shapes the 2026-10-16 breach took (list the directory,
count its lines, read a record) and running an observer script; it must
allow everything else, including editing its own source and reading an
observer script's code. End to end through a subprocess, so the JSON the
harness reads is what is tested.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
GUARD = SCRIPTS / "kenning_encounter_observe_guard.py"

spec = importlib.util.spec_from_file_location("kenning_encounter_observe_guard", GUARD)
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

LOG = "/home/someone/.kenning_encounter-observe/infuse-observe.jsonl"


def _run(event, env=None):
    r = subprocess.run([sys.executable, str(GUARD)], input=json.dumps(event),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, "the guard never blocks by exit code"
    return json.loads(r.stdout) if r.stdout.strip() else None


def _denied(out):
    return bool(out) and out["hookSpecificOutput"]["permissionDecision"] == "deny"


REFUSE = [
    ("Read", {"file_path": LOG}),
    ("Bash", {"command": "ls -la ~/.kenning_encounter-observe/"}),
    ("Bash", {"command": f"wc -l {LOG}"}),
    ("Bash", {"command": "tail -1 $HOME/.kenning_encounter-observe/infuse-observe.jsonl | python3 -m json.tool"}),
    ("Bash", {"command": "python3 -c \"print(open('/x/go-forward-shadow.jsonl').read())\""}),
    ("Grep", {"pattern": "seed", "path": "/home/someone/.kenning_encounter-observe"}),
    ("Glob", {"pattern": "*.jsonl", "path": "/home/someone/.kenning_encounter-observe"}),
    ("Edit", {"file_path": LOG, "old_string": "a", "new_string": "b"}),
    ("Bash", {"command": "uv run python scripts/matcher_voices_report.py"}),
    ("Bash", {"command": "python3 scripts/kenning_encounter_invocation_scorer.py --since today"}),
    ("Bash", {"command": "python3 scripts/kenning_encounter_replay_runbook.py"}),
]

ALLOW = [
    ("Read", {"file_path": "/repo/src/kenning_encounter/infuse.py"}),
    ("Bash", {"command": "uv run pytest -q"}),
    ("Bash", {"command": "ls ~/.kenning_encounter/hooks"}),
    # Editing the guard itself, whose content names the log, stays possible.
    ("Edit", {"file_path": "/repo/scripts/kenning_encounter_observe_guard.py",
              "old_string": "x", "new_string": "PROTECTED_PATHS = ('.kenning_encounter-observe',)"}),
    # Searching code for the string is not reading the stream.
    ("Grep", {"pattern": "kenning_encounter-observe", "path": "/repo/scripts"}),
    # Reading an observer script's source is not running it.
    ("Read", {"file_path": "/repo/scripts/matcher_voices_report.py"}),
    ("mcp__kenning_encounter__read_cypher", {"query": "MATCH (n) RETURN count(n)"}),
]


@pytest.mark.parametrize("tool,inp", REFUSE)
def test_refuses(tool, inp):
    out = _run({"tool_name": tool, "tool_input": inp})
    assert _denied(out), f"{tool} {inp} was allowed"
    assert "observer" in out["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.parametrize("tool,inp", ALLOW)
def test_allows(tool, inp):
    assert _run({"tool_name": tool, "tool_input": inp}) is None


def test_extra_paths_and_scripts_from_env(monkeypatch):
    monkeypatch.setenv("KENNING_ENCOUNTER_GUARD_PATHS", "/srv/observe-host2, other-log")
    monkeypatch.setenv("KENNING_ENCOUNTER_GUARD_SCRIPTS", "my_observer")
    assert guard.verdict({"tool_name": "Read", "tool_input": {"file_path": "/srv/observe-host2/a"}})
    assert guard.verdict({"tool_name": "Bash", "tool_input": {"command": "python3 my_observer.py"}})
    assert guard.verdict({"tool_name": "Read", "tool_input": {"file_path": "/srv/other"}}) is None


def test_fails_open_and_says_so():
    r = subprocess.run([sys.executable, str(GUARD)], input="not json",
                       capture_output=True, text=True)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert "allowed unguarded" in out["systemMessage"]
    assert "hookSpecificOutput" not in out


def test_the_infuse_hooks_log_dirs_are_protected(monkeypatch):
    monkeypatch.setenv("KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG", "/var/obs/run/infuse.jsonl")
    monkeypatch.setenv("KENNING_ENCOUNTER_INFUSE_SHADOW_LOG", "/var/shadow/s.jsonl")
    assert guard.verdict({"tool_name": "Read", "tool_input": {"file_path": "/var/obs/run/infuse.jsonl"}})
    assert guard.verdict({"tool_name": "Bash", "tool_input": {"command": "ls /var/shadow"}})
    assert guard.verdict({"tool_name": "Read", "tool_input": {"file_path": "/var/obs/other.txt"}}) is None


def test_a_log_at_the_root_does_not_protect_everything(monkeypatch):
    monkeypatch.setenv("KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG", "/infuse.jsonl")
    assert guard.verdict({"tool_name": "Read", "tool_input": {"file_path": "/etc/hosts"}}) is None
