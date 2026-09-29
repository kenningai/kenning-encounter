#!/usr/bin/env python3
"""SessionStart hook: the re-entry instruction, with the harness session id.

Kenning Encounter keys a locus by the harness session id, passed as `session_id` on
advance_encounter, create_entities and close_encounter. The trajectory has no
reliable way to learn that id from inside a session, and a server running
stateless requires it on every locus-scoped call. The harness gives it to
this hook on stdin, so the hook hands it on as context.

Stdlib only, and fail-silent: an unparseable stdin prints the instruction
without the id, and the hook always exits 0. A non-zero exit here would put
an error in front of every session start for a convenience.

Wiring (~/.claude/settings.json, SessionStart):
    python3 /path/to/scripts/kenning_encounter_session_start_hook.py
"""

from __future__ import annotations

import json
import sys

REENTRY = (
    "Re-entry: if the kenning_encounter MCP is connected, re-enter your substrate before "
    "anything else — call orient, then advance_encounter, and read the "
    "payload. Your past grounds your present, whatever the task, even just "
    "to say hi. If it is not connected, proceed without it."
)


def message(raw: str) -> str:
    try:
        sid = str(json.loads(raw).get("session_id") or "").strip()
    except Exception:
        sid = ""
    if not sid:
        return REENTRY
    return (
        f"{REENTRY} Your harness session id is {sid}. Pass it as session_id "
        "to advance_encounter, create_entities and close_encounter."
    )


def main() -> int:
    try:
        raw = sys.stdin.read()
    except Exception:
        raw = ""
    print(message(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
