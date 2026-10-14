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

Wiring (~/.claude/settings.json, SessionStart), from the copy that
scripts/kenning_encounter_install_hooks.py makes:
    python3 ~/.kenning_encounter/hooks/kenning_encounter_session_start_hook.py

INSTALLED-COPY DRIFT. When this file runs from a copy made by
kenning_encounter_install_hooks.py, a SOURCE.json manifest sits beside it, and the hook
compares the installed hooks with the same files at the repo's sealed main.
It compares bytes, not tag names, so a release that does not touch the hooks
raises nothing. Once a manifest exists the line is never silent: current,
HOOKS STALE, HOOKS ALTERED (edited since install), or HOOKS CHECK UNREADABLE.
The comparison is against the local repo's main, which moves when a release
is fetched or sealed here, so it reports what this host already knows is sealed. Run from the
repo itself (no manifest), the hook prints exactly what it printed before.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

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


def _git(repo: str, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                       text=not binary, timeout=5)
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace") if binary else r.stderr
        raise OSError(((err or "").strip().splitlines() or ["git failed"])[-1][:120])
    return r.stdout


def hooks_line(here: Path) -> str | None:
    """One line about the installed hooks, or None when not installed by manifest."""
    manifest = here / "SOURCE.json"
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text())
        repo, installed, files = m["repo"], m["tag"], m["files"]
        altered = [n for n, digest in files.items()
                   if hashlib.sha256((here / n).read_bytes()).hexdigest() != digest]
        if altered:
            return (f"HOOKS ALTERED: {', '.join(altered)} in {here} no longer match what was "
                    f"installed from {installed}. Tell the user before anything else.")
        main = _git(repo, "rev-parse", "main^{commit}").strip()
        tags = [t for t in _git(repo, "tag", "--points-at", main).split() if t.startswith("v")]
        main_tag = tags[-1] if tags else main[:12]
        stale = [n for n in files
                 if _git(repo, "show", f"{main}:scripts/{n}", binary=True) != (here / n).read_bytes()]
    except Exception as e:
        return (f"HOOKS CHECK UNREADABLE ({type(e).__name__}: {e}). Whether the installed "
                f"hooks in {here} are current is unknown: tell the user.")
    if stale:
        return (f"HOOKS STALE: the hooks in {here} were installed from {installed}, and the "
                f"repo's sealed main ({main_tag}) changes {', '.join(stale)}. Tell the user "
                f"before anything else; reinstall with python3 {repo}/scripts/kenning_encounter_install_hooks.py")
    return f"Hooks: installed from {installed}, identical to the repo's sealed main ({main_tag})."


def main() -> int:
    try:
        raw = sys.stdin.read()
    except Exception:
        raw = ""
    print(message(raw))
    try:
        line = hooks_line(Path(__file__).resolve().parent)
    except Exception:
        line = None
    if line:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
