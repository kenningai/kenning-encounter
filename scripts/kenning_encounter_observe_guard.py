#!/usr/bin/env python3
"""PreToolUse guard: the observed subject never reads its own measurement stream.

The infuse hook writes an observe log (and a matcher shadow log) whose reader
is the observer, never the trajectory: a subject reading its own measurement
stream contaminates the measures. As a comment in the infuse hook's header,
that rule held only while it was remembered.

This hook runs on the agent's tool calls only. The observer's terminal, the observer's
scripts and the infuse hook writing the log never pass through it. It denies:

- any tool call whose path-bearing input names a protected path (Read,
  Edit, Write, NotebookEdit, Grep, Glob: file_path / path / notebook_path /
  glob), and any Bash command whose text names one;
- any Bash command that runs an observer script, whose output is the
  observer's to read. Reading an observer script's SOURCE stays allowed.

Content fields (an Edit's new_string, a Grep pattern) are not matched, so the
guard's own source and the docs that name the log stay editable.

It is a guard against accident, not against evasion: a path assembled in a
variable passes, and so does a recursive search rooted above the log
directory. The deny reason says what to do instead, because the reader who
can act on it is the trajectory, at the moment of the call.

Fails open, and says so: a crash on every tool call would wedge the session,
so an unreadable input allows the call with a systemMessage to the user.

    KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG, KENNING_ENCOUNTER_INFUSE_SHADOW_LOG
                        when set (wire the guard with the same values as
                        the infuse hook), each log's directory is protected
    KENNING_ENCOUNTER_GUARD_PATHS    extra protected path fragments, comma-separated
    KENNING_ENCOUNTER_GUARD_SCRIPTS  extra observer script names, comma-separated

Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys

PROTECTED_PATHS = (".kenning_encounter-observe", "infuse-observe", "go-forward-shadow")
OBSERVER_SCRIPTS = (
    "matcher_voices_report",
    "kenning_encounter_invocation_scorer",
    "kenning_encounter_replay_runbook",
)
PATH_FIELDS = ("file_path", "path", "notebook_path", "glob")

REASON = (
    "Refused by kenning_encounter_observe_guard: this names {what} ({hit}). That is your own "
    "infusion measurement stream, and its reader is the observer who runs this "
    "deployment, never you: a subject reading its own measurements contaminates "
    "them. Do not route around this. If the measurement is wanted, ask the "
    "observer for their reading of the log, or for a script they run whose "
    "output you do not see."
)


def _extra(env: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in os.environ.get(env, "").split(",") if s.strip())


def _log_dirs() -> tuple[str, ...]:
    """The directories the infuse hook is told to write its logs into."""
    dirs = []
    for env in ("KENNING_ENCOUNTER_INFUSE_OBSERVE_LOG", "KENNING_ENCOUNTER_INFUSE_SHADOW_LOG"):
        value = os.environ.get(env, "").strip()
        if value:
            d = os.path.dirname(os.path.expanduser(value))
            if d and d != os.sep:
                dirs.append(d)
    return tuple(dirs)


def verdict(event: dict) -> str | None:
    """The deny reason for this tool call, or None to allow it. Pure."""
    tool = event.get("tool_name") or ""
    inp = event.get("tool_input") or {}
    paths = PROTECTED_PATHS + _log_dirs() + _extra("KENNING_ENCOUNTER_GUARD_PATHS")
    scripts = OBSERVER_SCRIPTS + _extra("KENNING_ENCOUNTER_GUARD_SCRIPTS")

    if tool == "Bash":
        command = str(inp.get("command") or "")
        for p in paths:
            if p in command:
                return REASON.format(what="a protected path", hit=p)
        for s in scripts:
            if s in command:
                return REASON.format(what="an observer script", hit=s)
        return None

    for field in PATH_FIELDS:
        value = str(inp.get(field) or "")
        for p in paths:
            if p in value:
                return REASON.format(what="a protected path", hit=p)
    return None


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
        reason = verdict(event)
    except Exception as exc:  # fail open, visibly
        print(json.dumps({
            "systemMessage": f"kenning_encounter_observe_guard could not read this tool call "
                             f"({type(exc).__name__}); it was allowed unguarded.",
        }))
        return 0
    if reason:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
