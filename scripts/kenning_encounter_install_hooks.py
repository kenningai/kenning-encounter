#!/usr/bin/env python3
"""Install the harness hooks from a sealed release, and say when they drift.

The hooks run from a copy outside the repo (default ~/.kenning_encounter/hooks), never
from the working tree: a working tree may hold an unsealed working set, and
a hook runs at every waking. A copy kept current by a note beside it is kept
by vigilance.
So the copy is made by this script, from a ref that carries a
sealed vX.Y.Z tag, read out of git rather than off disk, and it records what
it installed in SOURCE.json. The session-start hook reads that manifest at
every waking and reports when the copy is behind the repo's sealed main.

    python3 scripts/kenning_encounter_install_hooks.py               # install from main
    python3 scripts/kenning_encounter_install_hooks.py --ref vX.Y.Z
    python3 scripts/kenning_encounter_install_hooks.py --check       # exit 1 on drift

Wiring (~/.claude/settings.json) then points at the installed copies:
    SessionStart:     python3 ~/.kenning_encounter/hooks/kenning_encounter_session_start_hook.py
    UserPromptSubmit: python3 ~/.kenning_encounter/hooks/kenning_encounter_infuse_hook.py

Stdlib only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HOOKS = ("kenning_encounter_infuse_hook.py", "kenning_encounter_session_start_hook.py")
MANIFEST = "SOURCE.json"
LEGACY_NOTE = "SOURCE.txt"
SEMVER = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


class Refused(Exception):
    pass


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or '').strip()[:200]}")
    return r.stdout


def sealed_tag(repo: Path, commit: str) -> str | None:
    """The highest vX.Y.Z tag on this exact commit, or None."""
    tags = [t for t in git(repo, "tag", "--points-at", commit).split() if SEMVER.match(t)]
    if not tags:
        return None
    return max(tags, key=lambda t: tuple(int(x) for x in SEMVER.match(t).groups()))


def resolve(repo: Path, ref: str) -> tuple[str, str]:
    """(commit, tag) for a ref, refusing one that is not a sealed release."""
    commit = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    tag = sealed_tag(repo, commit)
    if tag is None:
        raise Refused(f"{ref} ({commit[:12]}) carries no vX.Y.Z tag: install only from a sealed release")
    return commit, tag


def release_bytes(repo: Path, commit: str, name: str) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:scripts/{name}"],
                       capture_output=True)
    if r.returncode != 0:
        raise Refused(f"scripts/{name} is not in {commit[:12]}")
    return r.stdout


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def install(repo: Path, ref: str, dest: Path) -> dict:
    commit, tag = resolve(repo, ref)
    payload = {name: release_bytes(repo, commit, name) for name in HOOKS}
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        tmp = dest / f".{name}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, dest / name)
    manifest = {
        "repo": str(repo.resolve()),
        "ref": ref,
        "tag": tag,
        "commit": commit,
        "installed": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": {name: sha256(data) for name, data in payload.items()},
    }
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    # The hand-written note this script replaces; it would only contradict
    # the manifest the first time the two disagreed.
    (dest / LEGACY_NOTE).unlink(missing_ok=True)
    return manifest


def check(repo: Path, ref: str, dest: Path) -> list[str]:
    """One line per hook; empty list of problems means the copy is current."""
    commit, tag = resolve(repo, ref)
    problems = []
    for name in HOOKS:
        p = dest / name
        if not p.exists():
            problems.append(f"{name}: missing from {dest}")
        elif p.read_bytes() != release_bytes(repo, commit, name):
            problems.append(f"{name}: differs from {tag} ({commit[:12]})")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="main", help="sealed ref to install from (default: main)")
    ap.add_argument("--dest", default="~/.kenning_encounter/hooks", help="install directory (default: ~/.kenning_encounter/hooks)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                    help="repo root (default: this script's repo)")
    ap.add_argument("--check", action="store_true", help="report drift against --ref; exit 1 if any")
    a = ap.parse_args(argv)
    repo, dest = Path(a.repo), Path(os.path.expanduser(a.dest))
    try:
        if a.check:
            problems = check(repo, a.ref, dest)
            for line in problems:
                print(f"DRIFT {line}")
            if not problems:
                _, tag = resolve(repo, a.ref)
                print(f"OK {dest} matches {tag}")
            return 1 if problems else 0
        m = install(repo, a.ref, dest)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"Installed {', '.join(HOOKS)} from {m['tag']} ({m['commit'][:12]}) into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
