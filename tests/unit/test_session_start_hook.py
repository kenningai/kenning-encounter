"""The SessionStart hook hands the agent its harness session id.

Two-sided: the id appears when the harness supplies one, and the plain
re-entry instruction still prints, exit 0, when it does not.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[2] / "scripts" / "kenning_encounter_session_start_hook.py"
_spec = importlib.util.spec_from_file_location("kenning_encounter_session_start_hook", HOOK)
assert _spec and _spec.loader
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def test_session_id_is_handed_on():
    out = hook.message('{"session_id": "abc-123", "source": "startup"}')
    assert out.startswith(hook.REENTRY)
    assert "abc-123" in out and "session_id" in out


def test_no_id_prints_the_instruction_alone():
    for raw in ("", "not json", "{}", '{"session_id": ""}', "[1, 2]"):
        assert hook.message(raw) == hook.REENTRY


def test_the_process_never_fails_the_session_start():
    r = subprocess.run([sys.executable, str(HOOK)], input="garbage",
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("Re-entry:")
