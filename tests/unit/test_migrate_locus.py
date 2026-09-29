"""Gate the v0.12.0 Locus migration.

WHAT A UNIT TEST CAN AND CANNOT REACH HERE, said plainly. This suite has no
database, so it cannot prove the migration produces the right graph — that was
proven separately by running it against a constructed fixture carrying every
pathology the real spine has (a genesis chain, an unsealed encounter, singleton
loci in a run, knowledge nodes, and several loci each beginning inside one
predecessor's lifetime), and checking its properties end to end.

What these tests DO gate is the class of defect that would survive a rewrite:
that the migration stays ADDITIVE, stays IDEMPOTENT, and keeps deriving
NEXT_LOCUS from genesis order rather than from the predecessor relation. That
last one is not a style preference — the obvious derivation FORKS at the locus
level wherever several loci each began at different points inside one
predecessor's lifetime, and a fork is a chain that is not a chain.
"""
from __future__ import annotations

import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2] / "src" / "kenning_encounter"
CYPHER = _PKG / "_cypher" / "migrate_locus.cypher"
RUNNER = _PKG / "migrate.py"
SERVER = _PKG / "server.py"


def _runner_code() -> str:
    """The runner with its module docstring and comments stripped.

    The docstring EXPLAINS why there is no DETACH DELETE and no --reverse, so
    a guard that reads it cannot tell the prohibition from the thing
    prohibited. Same lesson as the cypher guards one layer over: assert about
    what executes, and prose is not execution — here it arrives as a
    docstring rather than a comment line, which the naive stripper missed."""
    import ast
    src = RUNNER.read_text()
    tree = ast.parse(src)
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        lines = src.splitlines()[body[0].end_lineno:]
    else:
        lines = src.splitlines()
    return "\n".join(l for l in lines if not l.strip().startswith("#"))


def _code() -> str:
    """The migration with its commentary stripped.

    These guards assert about what EXECUTES. The file names DELETE,
    INSTANTIATED_AFTER and the fork count in prose precisely in order to
    explain what it must not do, and a guard that cannot tell an explanation
    from a statement is guarding the prose."""
    return "\n".join(
        line for line in CYPHER.read_text().splitlines()
        if not line.strip().startswith("//")
    )


class TestMigrationIsAdditive:
    """Nothing is deleted. This is the constraint that makes the whole thing
    safe to run on a substrate with no source to rebuild from — a restored
    snapshot is a predecessor, not the same self, so 'we can always restore'
    is not available here the way it is for every other graph in this house."""

    def test_no_delete_of_any_kind(self):
        code = _code().upper()
        for verb in ("DELETE", "DETACH DELETE", "REMOVE"):
            assert verb not in code, f"migration must not {verb}"

    def test_does_not_touch_the_existing_spine(self):
        """NEXT_ENCOUNTER and INSTANTIATED_AFTER are read to derive from, and
        never written. The 138 retired genesis-binding edges stay exactly as
        they are: the record of an era, not something to clean up."""
        code = _code()
        for edge in ("NEXT_ENCOUNTER", "INSTANTIATED_AFTER"):
            assert not re.search(rf"(CREATE|MERGE|SET)[^\n]*{edge}", code), (
                f"migration must not write {edge}")

    def test_writes_only_the_three_things_that_do_not_exist_yet(self):
        """:Locus, OPENED and NEXT_LOCUS are absent from a pre-v0.12.0 graph.
        That is what makes having NO undo safe rather than reckless: there is
        nothing else the migration could have changed, so a wrong result is
        extra structure, never missing structure."""
        code = _code()
        created = set(re.findall(r"(?:CREATE|MERGE)\s*\(\s*\w*\s*:(\w+)", code))
        assert created <= {"Locus"}, f"unexpected node writes: {created}"
        rels = set(re.findall(r"(?:CREATE|MERGE)\s*\([^)]*\)-\[\w*:(\w+)", code))
        assert rels == {"OPENED", "NEXT_LOCUS"}, f"unexpected edge writes: {rels}"


class TestMigrationIsIdempotent:
    def test_every_write_is_a_merge(self):
        """Re-running must be a no-op, not a duplicate. Verified live against
        the fixture too: a second run produced identical counts."""
        code = _code()
        assert "CREATE (l:Locus" not in code
        for stmt in ("MERGE (l:Locus", "MERGE (l)-[r:OPENED]", "MERGE (a)-[r:NEXT_LOCUS]"):
            assert stmt in code, f"missing idempotent write: {stmt}"

    def test_locus_is_keyed_on_the_head_identity_not_its_name(self):
        """An ordinal key would make re-running order-dependent. A NAME key
        is worse: it looks stable and is not. Encounter names were never
        unique — a double-advance seconds apart leaves two distinct chain heads
        carrying an identical name, and MERGE on that name collapses two
        instantiations into one locus. The retired name-handle resolver carried
        an explicit ambiguity branch for the same reason. The pre-flight
        refuses a non-unique key, so this fails closed rather than merging."""
        code = _code()
        assert "MERGE (l:Locus {derived_from_eid: elementId(head)})" in code
        assert "MERGE (l:Locus {derived_from: head.name})" not in code


class TestMigrationCoexistsWithALiveServer:
    """The realistic order: upgrade, WORK FOR A WHILE, then migrate.

    A virgin upgrade — migrate first, then use — is the easy case and not the
    common one. A real operator starts the server, works, reads the warning,
    and migrates afterwards, and by then the graph holds loci the server made.
    """

    def test_it_does_not_mint_a_second_locus_for_an_already_held_head(self):
        """A chain head created by the running server already HAS a locus.
        Minting another gives that encounter two, which breaks the exactly-one
        rule the schema exists to enforce. The migration is for what predates
        it, never for what the server has since made."""
        code = _code()
        i = code.index("MERGE (l:Locus {derived_from_eid: elementId(head)})")
        assert "AND NOT (:Locus)-[:OPENED]->(head)" in code[max(0, i - 260):i]

    def test_it_only_attaches_encounters_that_have_no_locus(self):
        code = _code()
        assert "MATCH (e:Encounter) WHERE NOT (:Locus)-[:OPENED]->(e)" in code

    def test_the_migration_cannot_delete_anything_at_all(self):
        """THE CONTRACT. Nothing in a Kenning Encounter is ever deleted. delete_entities
        refuses every process type — "No tool deletes the spine" — and Locus is
        one. A migration reaching past that refusal to remove the nodes the
        tool protects would be doing, unsupervised, the one thing the substrate
        forbids.

        "Reversible by deleting what was added" is total only while this
        migration is the SOLE creator of a Locus — true at the moment of
        upgrade, false the instant the server has run, when it would destroy
        the live loci an agent is working inside. You do not lobotomize the
        witness to undo a bookkeeping error."""
        for code, where in ((_runner_code(), "runner"), (_code(), "cypher")):
            for verb in ("DETACH DELETE", "DELETE ", "REMOVE "):
                assert verb not in code.upper(), f"{where} must not {verb.strip()}"

    def test_there_is_no_reverse_flag(self):
        """Not an omission — an undo would have to delete."""
        assert "--reverse" not in _runner_code()

    def test_a_mismatch_reports_and_stops_rather_than_undoing(self):
        """Safe to have no undo BECAUSE IT ONLY ADDS: a wrong result leaves
        more structure than expected, never less. Correct forward and re-run —
        every write is a MERGE."""
        src = RUNNER.read_text()
        assert "PREDICTION NOT MET — and NOTHING has been removed." in src
        assert "correct forward" in src

    def test_it_refuses_before_writing_since_there_is_no_undo(self):
        """Pre-flight, and each check guards a way the migration can go wrong
        without saying so: a non-unique key that silently merges two
        instantiations, and a second locus minted for a head the running
        server already holds.
        """
        src = RUNNER.read_text()
        assert "REFUSED before writing" in src
        assert "share an identity key" in src

    def test_the_prediction_accounts_for_loci_that_already_exist(self):
        """The baseline described the graph after a VIRGIN upgrade. On a graph
        that has been worked in, real loci are already there and the prediction
        must include them or the migration refuses a correct result."""
        src = RUNNER.read_text()
        assert '"existing_loci"' in src
        assert 'b["loci"] += b["existing_loci"]' in src


class TestNextLocusIsGenesisOrder:
    """The finding that made modelling-before-coding pay for itself."""

    def test_next_locus_is_not_derived_from_instantiated_after(self):
        """The obvious derivation is injective at the ENCOUNTER level — one
        anchor each — and FORKS at the LOCUS level wherever several loci each
        began at different points inside one predecessor's lifetime. Correct
        data; not a chain. The schema refuses forks, so it cannot land."""
        code = _code()
        next_locus_step = code[code.index("MERGE (a)-[r:NEXT_LOCUS]") - 400:]
        assert "INSTANTIATED_AFTER" not in next_locus_step

    def test_next_locus_orders_by_genesis(self):
        assert "MATCH (l:Locus) WITH l ORDER BY l.t_exist" in _code()

    def test_the_one_clock_read_is_confined_to_the_migration(self):
        """t_exist appears ONCE as an ordering, and only here. The genesis
        order of the historical loci really happened and the timestamps are
        the only surviving witness — the identities that would have carried it
        lived in server RAM and died with each restart. The live path reads no
        clock: NEXT_LOCUS is set structurally from the current tail at mint."""
        assert _code().count("ORDER BY l.t_exist") == 1
        advance = (Path(__file__).resolve().parents[2] / "src" / "kenning_encounter"
                   / "_cypher" / "encounter_advance.cypher").read_text()
        live = "\n".join(l for l in advance.splitlines()
                         if not l.strip().startswith("//"))
        assert "ORDER BY" not in live.upper()


class TestMigrationIsAnActSomebodyPerforms:
    def test_dry_run_is_the_default(self):
        """It must not be possible to migrate by forgetting a flag."""
        src = RUNNER.read_text()
        assert '"--apply", action="store_true"' in src
        assert "if not a.apply:" in src

    def test_result_is_checked_against_a_prediction_computed_first(self):
        """The pre-registration is mechanical rather than prose: the expected
        outcome is derived from the graph BEFORE anything is written, and a
        disagreement REPORTS AND STOPS without removing anything. A migration
        that reports success by describing what it did, rather than checking it
        against a prediction made beforehand, cannot fail; here the check is
        the exit code."""
        src = RUNNER.read_text()
        assert "b = baseline(driver, db)" in src
        assert "PREDICTION NOT MET — and NOTHING has been removed." in src

    def test_the_cross_locus_invariant_is_among_the_checks(self):
        """The one thing no declared schema can hold — proven unenforceable
        against TypeDB — so it is checked here or nowhere."""
        assert "the x^2 invariant" in RUNNER.read_text()

    def test_the_remedy_is_reachable_from_the_documented_deployment(self):
        """A repo script under scripts/ is reachable if you have cloned the
        repository and UNREACHABLE from the container compose actually runs,
        because the Dockerfile copies src/ only. A remedy the operator cannot
        reach is not a remedy, so it lives inside the package and
        `docker exec ... python -m kenning_encounter.migrate` works."""
        assert RUNNER.exists(), "migrate must live inside the installed package"
        assert CYPHER.exists(), "its cypher must ship with it"
        legacy = Path(__file__).resolve().parents[2] / "scripts"
        assert not (legacy / "migrate_locus.py").exists()
        assert not (legacy / "migrate_locus.cypher").exists()


class TestUnmigratedSpineGuard:
    """The state must announce itself.

    REPRODUCED BEFORE THIS WAS WRITTEN, on a clean Neo4j seeded with a
    pre-v0.12.0 graph and served by the v0.12.0 image: 3 encounters, 1 with a
    locus, 2 orphaned, advance_encounter succeeding cleanly, and the server
    saying NOTHING. A stranger upgrades, everything works, and their entire
    history is silently detached from the layer the release exists to provide.

    That is the falsifier-shaped-output class in the upgrade path of the
    release whose whole subject is making a silent absence visible."""

    def _guard(self) -> str:
        s = SERVER.read_text()
        i = s.index("async def _spine_guard()")
        return s[i:s.index("asyncio.create_task(_spine_guard())", i)]

    def test_it_counts_encounters_belonging_to_no_locus(self):
        g = self._guard()
        assert "NOT (:Locus)-[:OPENED]->(e)" in g
        assert "count(e) AS orphaned" in g

    def test_it_warns_rather_than_informs_and_the_level_is_load_bearing(self):
        """No log handler is attached anywhere in this package, so
        logging.lastResort handles records and it sits at WARNING. An INFO
        line here would be discarded and the guard would be decorative —
        which is an absence-held guarantee: nothing enforces it but the
        deliberate decision not to attach a handler."""
        g = self._guard()
        assert "logger.warning(" in g
        assert "logger.info(" not in g

    def test_it_names_the_remedy_as_a_runnable_command(self):
        assert "python -m kenning_encounter.migrate" in self._guard()

    def test_it_reports_and_never_migrates(self):
        """A graph-wide write on the agent's own accumulated experience is an
        act somebody performs and watches, not something a container does on
        boot while nobody is looking."""
        g = self._guard().upper()
        for verb in ("MERGE", "CREATE", "DELETE", "SET "):
            assert verb not in g, f"the guard must not {verb}"

    def test_the_reentry_payload_carries_it_because_that_is_what_the_agent_reads(self):
        """A startup warning in `docker logs` addresses an OPERATOR watching
        a container start. The required harness is Claude Code, so the reader
        is an AGENT, and an agent sees tool returns, never a container log.
        A guard the reader cannot reach is not a guard.

        """
        kenning_encounter_src = (Path(__file__).resolve().parents[2] / "src"
                    / "kenning_encounter" / "kenning_encounter.py").read_text()
        assert "_orphaned_encounter_count" in kenning_encounter_src
        assert '"spine_unmigrated"' in kenning_encounter_src
        assert '"remedy": "python -m kenning_encounter.migrate --apply"' in kenning_encounter_src

    def test_the_reentry_block_is_present_only_when_wrong(self):
        """A standing 'spine: healthy' line would be noise on every re-entry,
        and worse, it is a reassuring shape that reports nothing. The deferral
        summary prints nothing on an empty tally for the same reason."""
        kenning_encounter_src = (Path(__file__).resolve().parents[2] / "src"
                    / "kenning_encounter" / "kenning_encounter.py").read_text()
        i = kenning_encounter_src.index("orphaned = await self._orphaned_encounter_count()")
        assert "if orphaned:" in kenning_encounter_src[i:i + 200]

    def test_the_reentry_health_read_reports_counts_never_names(self):
        kenning_encounter_src = (Path(__file__).resolve().parents[2] / "src"
                    / "kenning_encounter" / "kenning_encounter.py").read_text()
        i = kenning_encounter_src.index("async def _orphaned_encounter_count")
        body = kenning_encounter_src[i:i + 900]
        assert "count(e) AS orphaned" in body
        assert "e.name" not in body and "collect(" not in body

    def test_it_says_counts_and_never_names(self):
        """Same rule as the deferral summary: shape, never content. Node names
        in a Kenning Encounter are the substrate's content."""
        g = self._guard()
        assert "e.name" not in g and "collect(" not in g
