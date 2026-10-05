"""The reader profile (reader.py): a read-only allowlist, and withheld loci
removed from every result. Each guard is shown saying yes and no, end to end
through a real FastMCP client, not only as a pure function."""

import json

import pytest
from fastmcp import Client, FastMCP

from kenning_encounter.reader import (
    READER_TOOLS,
    WITHHELD,
    ReaderMiddleware,
    filter_hidden,
    locus_names,
)

WITHHELD_NODE = "Evaluation verdict: one method won"
ENCOUNTER = "Encounter 2026-10-01 — the experiment"
HIDDEN = frozenset({WITHHELD_NODE, ENCOUNTER, "Locus abc"})


class TestLocusNames:
    def test_session_ids_and_full_names(self):
        assert locus_names("abc, Locus def ,") == ["Locus abc", "Locus def"]
        assert locus_names("") == []


class TestFilterHidden:
    def test_rows_naming_a_withheld_node_are_dropped(self):
        rows = [{"name": WITHHELD_NODE, "type": "Observation"},
                {"name": "A guard in code", "type": "Concept"}]
        assert filter_hidden(rows, HIDDEN) == [{"name": "A guard in code", "type": "Concept"}]

    def test_nested_mentions_drop_the_whole_item(self):
        edges = [{"from": "X", "rel": "GROUNDS", "to": {"name": WITHHELD_NODE}},
                 {"from": "X", "rel": "ABOUT", "to": {"name": "Y"}}]
        assert filter_hidden(edges, HIDDEN) == [{"from": "X", "rel": "ABOUT", "to": {"name": "Y"}}]

    def test_strings_outside_lists_are_redacted(self):
        out = filter_hidden({"root": WITHHELD_NODE, "note": f"see {ENCOUNTER}."}, HIDDEN)
        assert out == {"root": WITHHELD, "note": f"see {WITHHELD}."}

    def test_nothing_withheld_changes_nothing(self):
        data = {"rows": [{"name": WITHHELD_NODE}]}
        assert filter_hidden(data, frozenset()) == data


def _server(hidden_provider):
    mcp = FastMCP("t")

    @mcp.tool(name="search")
    def search(query: str) -> list[dict]:
        return [{"name": WITHHELD_NODE}, {"name": "A guard in code"}]

    @mcp.tool(name="find_by_name")
    def find_by_name(name: str) -> dict:
        return {"name": name, "summary": f"recorded in {ENCOUNTER}"}

    @mcp.tool(name="create_entities")
    def create_entities(x: str) -> str:
        return "WROTE"

    @mcp.tool(name="infuse_meaning")
    def infuse_meaning(text: str) -> str:
        return "SELECTIONS"

    @mcp.tool(name="brand_new_tool")
    def brand_new_tool() -> str:
        return "SHOULD NOT BE REACHABLE"

    mcp.add_middleware(ReaderMiddleware(READER_TOOLS, hidden_provider))
    return mcp


async def _hidden():
    return HIDDEN


class TestAllowlist:
    @pytest.mark.asyncio
    async def test_only_allowed_tools_are_listed(self):
        async with Client(_server(_hidden)) as c:
            names = {t.name for t in await c.list_tools()}
        assert names == {"search", "find_by_name"}  # the allowed ones this server has

    @pytest.mark.asyncio
    async def test_writes_matcher_and_unknown_tools_are_refused(self):
        async with Client(_server(_hidden)) as c:
            for name, args in (("create_entities", {"x": "y"}),
                               ("infuse_meaning", {"text": "t"}),
                               ("brand_new_tool", {})):
                with pytest.raises(Exception) as exc:
                    await c.call_tool(name, args)
                assert "not available in the reader profile" in str(exc.value)

    @pytest.mark.asyncio
    async def test_an_allowed_tool_still_answers(self):
        async with Client(_server(_hidden)) as c:
            r = await c.call_tool("search", {"query": "q"})
        assert "A guard in code" in r.content[0].text


class TestWithheld:
    @pytest.mark.asyncio
    async def test_search_results_lose_the_withheld_node(self):
        async with Client(_server(_hidden)) as c:
            r = await c.call_tool("search", {"query": "q"})
        text = r.content[0].text
        assert WITHHELD_NODE not in text and "A guard in code" in text
        assert WITHHELD_NODE not in json.dumps(r.structured_content)

    @pytest.mark.asyncio
    async def test_a_direct_lookup_is_redacted(self):
        async with Client(_server(_hidden)) as c:
            r = await c.call_tool("find_by_name", {"name": WITHHELD_NODE})
        assert WITHHELD_NODE not in r.content[0].text and ENCOUNTER not in r.content[0].text
        assert WITHHELD in r.content[0].text

    @pytest.mark.asyncio
    async def test_without_a_withheld_set_the_secret_would_show(self):
        # The other side: the same server with nothing withheld returns it,
        # so the tests above are testing the filter, not the fixture.
        async def none():
            return frozenset()

        async with Client(_server(none)) as c:
            r = await c.call_tool("search", {"query": "q"})
        assert WITHHELD_NODE in r.content[0].text

    @pytest.mark.asyncio
    async def test_fails_closed_when_the_withheld_set_cannot_be_computed(self):
        async def broken():
            raise ValueError("withheld loci not in the graph")

        async with Client(_server(broken)) as c:
            with pytest.raises(Exception) as exc:
                await c.call_tool("search", {"query": "q"})
        assert "could not compute withheld nodes" in str(exc.value)


class TestNonAsciiNames:
    """A withheld node whose name contains a non-ASCII character (an em
    dash, say) is dropped whole, row and all. Serialising the row with
    json.dumps' default ensure_ascii escapes the dash to \\u2014, so the raw
    name would never match the row and only the later string redaction
    would fire, leaving a '[withheld]' row with its other fields intact.
    """

    def test_a_row_naming_a_non_ascii_withheld_node_is_dropped_whole(self):
        name = "A node named the way many are — with an em dash"
        rows = [{"name": name, "t": "2026-10-02T00:47:23Z"},
                {"name": "An unrelated node", "t": "2026-10-02T04:30:00Z"}]
        assert filter_hidden(rows, frozenset({name})) == [
            {"name": "An unrelated node", "t": "2026-10-02T04:30:00Z"}]


class TestProfileConfig:
    """The reader face withholds nothing unless an experiment says so, and
    the v0.18 name still selects it."""

    @staticmethod
    def _args(**overrides):
        import argparse

        base = dict(
            db_url="bolt://x", username="u", password="p", database="d",
            namespace=None, transport=None, server_host=None, server_port=None,
            server_path=None, allow_origins=None, allowed_hosts=None,
            read_timeout=None, infuse_frontier_bias=None,
            infuse_refresh_turns=None,
            matcher_endpoint=None, matcher_model=None,
            matcher_timeout_ms=None, matcher_top_n=None, matcher_sidecar=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for k in ("NEO4J_MCP_SERVER_PROFILE", "NEO4J_READER_HIDE_LOCI",
                  "NEO4J_AUTHOR_HIDE_LOCI"):
            monkeypatch.delenv(k, raising=False)

    def test_reader_withholds_nothing_by_default(self):
        from kenning_encounter.utils import process_config

        cfg = process_config(self._args(server_profile="reader"))
        assert cfg["server_profile"] == "reader"
        assert cfg["reader_hide_loci"] == []

    def test_author_is_the_old_name_for_reader(self, monkeypatch):
        from kenning_encounter.utils import process_config

        monkeypatch.setenv("NEO4J_MCP_SERVER_PROFILE", "author")
        monkeypatch.setenv("NEO4J_AUTHOR_HIDE_LOCI", "abc")
        cfg = process_config(self._args())
        assert cfg["server_profile"] == "reader"
        assert cfg["reader_hide_loci"] == ["Locus abc"]

    def test_an_experiment_names_what_it_withholds(self, monkeypatch):
        from kenning_encounter.utils import process_config

        monkeypatch.setenv("NEO4J_READER_HIDE_LOCI", "abc, Locus def")
        cfg = process_config(self._args(server_profile="reader"))
        assert cfg["reader_hide_loci"] == ["Locus abc", "Locus def"]
        assert process_config(self._args(
            server_profile="reader", reader_hide_loci="none",
        ))["reader_hide_loci"] == []

    def test_an_unknown_profile_refuses_to_start(self):
        from kenning_encounter.utils import process_config

        with pytest.raises(SystemExit):
            process_config(self._args(server_profile="admin"))


class _SchemaSpy:
    """Records every schema write a face attempts at startup."""

    def __init__(self):
        self.writes = []

    async def create_fulltext_index(self):
        self.writes.append("fulltext")

    async def create_indexes(self):
        self.writes.append("indexes")


class TestReaderStartupWritesNothing:
    """A reader that writes, even only at startup, is not a reader."""

    @pytest.mark.asyncio
    async def test_the_reader_face_writes_no_schema(self):
        from kenning_encounter.server import prepare_schema
        spy = _SchemaSpy()
        assert await prepare_schema(spy, "reader") is False
        assert spy.writes == []

    @pytest.mark.asyncio
    async def test_the_full_face_still_prepares_the_schema(self):
        from kenning_encounter.server import prepare_schema
        spy = _SchemaSpy()
        assert await prepare_schema(spy, "full") is True
        assert spy.writes == ["fulltext", "indexes"]
