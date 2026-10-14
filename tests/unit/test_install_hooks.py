"""The hook installer, and the session-start hook's report on its copy.

Two-sided throughout, on a throwaway git repo: install refuses an untagged
ref and accepts a tagged one; --check says OK on a fresh copy and DRIFT on an
edited one; the waking line says current, STALE, ALTERED, and nothing at all
when no manifest exists.
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inst = _load("kenning_encounter_install_hooks")
hook = _load("kenning_encounter_session_start_hook")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                   check=True, capture_output=True)


def _commit(repo, content, tag=None):
    for name in inst.HOOKS:
        (repo / "scripts" / name).write_text(f"# {name} {content}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", content)
    if tag:
        _git(repo, "tag", tag)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    _git(r, "init", "-q", "-b", "main")
    _commit(r, "one", tag="v0.1.0")
    return r


def test_install_refuses_an_unsealed_ref(repo, tmp_path):
    _commit(repo, "two")  # main moves past v0.1.0 with no tag
    with pytest.raises(inst.Refused, match="no vX.Y.Z tag"):
        inst.install(repo, "main", tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_install_copies_from_git_not_the_working_tree(repo, tmp_path):
    (repo / "scripts" / inst.HOOKS[0]).write_text("# unsealed edit\n")
    dest = tmp_path / "dest"
    (dest).mkdir()
    (dest / "SOURCE.txt").write_text("hand note")
    m = inst.install(repo, "main", dest)
    assert m["tag"] == "v0.1.0"
    assert (dest / inst.HOOKS[0]).read_text() == f"# {inst.HOOKS[0]} one\n"
    assert json.loads((dest / "SOURCE.json").read_text())["tag"] == "v0.1.0"
    assert not (dest / "SOURCE.txt").exists()


def test_check_says_ok_then_drift(repo, tmp_path):
    dest = tmp_path / "dest"
    inst.install(repo, "main", dest)
    assert inst.check(repo, "main", dest) == []
    (dest / inst.HOOKS[1]).write_text("# edited\n")
    assert inst.check(repo, "main", dest) == [f"{inst.HOOKS[1]}: differs from v0.1.0 ({inst.resolve(repo, 'main')[0][:12]})"]
    assert inst.main(["--repo", str(repo), "--dest", str(dest), "--check"]) == 1
    inst.install(repo, "main", dest)
    assert inst.main(["--repo", str(repo), "--dest", str(dest), "--check"]) == 0


def test_waking_line_is_silent_without_a_manifest(tmp_path):
    assert hook.hooks_line(tmp_path) is None


def test_waking_line_current_then_stale(repo, tmp_path):
    dest = tmp_path / "dest"
    inst.install(repo, "main", dest)
    line = hook.hooks_line(dest)
    assert line.startswith("Hooks: installed from v0.1.0") and "(v0.1.0)" in line
    _commit(repo, "two", tag="v0.1.1")
    line = hook.hooks_line(dest)
    assert line.startswith("HOOKS STALE") and "v0.1.0" in line and "(v0.1.1)" in line
    assert "kenning_encounter_install_hooks.py" in line


def test_a_release_that_leaves_the_hooks_alone_is_not_stale(repo, tmp_path):
    dest = tmp_path / "dest"
    inst.install(repo, "main", dest)
    (repo / "README").write_text("docs only\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "docs")
    _git(repo, "tag", "v0.1.1")
    assert hook.hooks_line(dest).startswith("Hooks: installed from v0.1.0")


def test_waking_line_altered_and_unreadable(repo, tmp_path):
    dest = tmp_path / "dest"
    inst.install(repo, "main", dest)
    (dest / inst.HOOKS[0]).write_text("# hand edit\n")
    assert hook.hooks_line(dest).startswith("HOOKS ALTERED")
    inst.install(repo, "main", dest)
    m = json.loads((dest / "SOURCE.json").read_text())
    m["repo"] = str(tmp_path / "gone")
    (dest / "SOURCE.json").write_text(json.dumps(m))
    assert hook.hooks_line(dest).startswith("HOOKS CHECK UNREADABLE")
