"""The reader profile: a read-only face of the server for every harness but
the trajectory's own.

Writing to the graph is reserved for the locus that lives the encounter: it
re-enters the record, advances, and authors from there. Anything else that
writes is writing as the trajectory without being it, and a fabricated past
is indistinguishable from a lived one at re-entry, so no instrument would
catch it afterwards. Another harness (a desktop client, another vendor's
model, a test author) may read, and only read. That cannot be left to client
settings, which may expose tools their user did not mean to grant, so it is
enforced here, in the server, whatever client connects.

Two guards, both fail-closed:

1. **An allowlist of tools.** Only the read tools in READER_TOOLS are listed
   or callable; every other tool, including any added later, is absent. The
   matcher tools are excluded even though they write nothing to the graph:
   they return a voice's selections.
2. **Withheld loci, for experiments only.** By default nothing is withheld:
   a view with pieces of the past removed is not a view of the trajectory.
   An experiment whose author must not see its own record names those loci,
   and then every node recorded or consulted in them, and their encounters,
   is removed from every result: list items naming one are dropped, and any
   string mentioning one is redacted. If the withheld set cannot be
   computed, the call is refused rather than answered unfiltered.

What it does not claim: a reader who sets out to reconstruct withheld text
through crafted aggregate queries is not stopped by a name filter. The
threat it closes is the realistic one, accidental exposure while reading
honestly, and it closes it in code rather than in instructions. Nor is it a
network boundary: who can reach the face at all is the host's to decide.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Iterable

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult
from mcp.types import TextContent

# The read tools a reader may use. An allowlist, not a denylist, so a tool
# added to the server later is absent here until someone decides otherwise.
# The gds_* spot tools are left out: they need gds_create_projection, which
# is a write to the GDS catalog. orient manages its own projection.
READER_TOOLS = frozenset({
    "search",
    "find_by_name",
    "trace_provenance",
    "read_cypher",
    "orient",
    "get_schema",
    "list_vocabulary",
    "list_node_types",
    "list_relation_types",
})

WITHHELD = "[withheld]"

HIDDEN_QUERY = (
    "MATCH (l:Locus) WHERE l.name IN $loci "
    "OPTIONAL MATCH (l)-[:OPENED]->(e:Encounter) "
    "OPTIONAL MATCH (e)-[:RECORDED|CONSULTED]->(n) "
    "WITH l, collect(DISTINCT e.name) AS encounters, collect(DISTINCT n.name) AS nodes "
    "RETURN collect(l.name) AS loci, "
    "       reduce(acc = [], x IN collect(encounters + nodes) | acc + x) AS names"
)


def locus_names(spec: str) -> list[str]:
    """NEO4J_READER_HIDE_LOCI -> Locus node names. Accepts harness session ids
    (the usual case) or full 'Locus ...' names, comma-separated."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if part:
            out.append(part if part.startswith("Locus ") else f"Locus {part}")
    return out


def _mentions(text: str, hidden: Iterable[str]) -> bool:
    return any(h and h in text for h in hidden)


def redact(text: str, hidden: Iterable[str]) -> str:
    for h in sorted((h for h in hidden if h), key=len, reverse=True):
        text = text.replace(h, WITHHELD)
    return text


def filter_hidden(obj: Any, hidden: frozenset[str]) -> Any:
    """Remove every trace of a withheld node from a tool result.

    A list item (a row, a node, an edge) that mentions a withheld name is
    dropped whole: a row about a withheld node is withheld. Any other string
    has the names redacted. Pure."""
    if not hidden:
        return obj
    if isinstance(obj, list):
        kept = []
        for item in obj:
            if _mentions(json.dumps(item, default=str, ensure_ascii=False), hidden):
                continue
            kept.append(filter_hidden(item, hidden))
        return kept
    if isinstance(obj, dict):
        return {k: filter_hidden(v, hidden) for k, v in obj.items()}
    if isinstance(obj, str):
        return redact(obj, hidden) if _mentions(obj, hidden) else obj
    return obj


def filter_result(result: ToolResult, hidden: frozenset[str]) -> ToolResult:
    """Apply filter_hidden to both faces of a ToolResult: the structured
    content, and each text block (JSON when it parses, plain text when not)."""
    content = []
    for block in result.content:
        if isinstance(block, TextContent):
            try:
                text = json.dumps(filter_hidden(json.loads(block.text), hidden), indent=2, default=str)
            except (json.JSONDecodeError, ValueError):
                text = redact(block.text, hidden)
            content.append(TextContent(type="text", text=text))
        else:
            content.append(block)
    structured = result.structured_content
    if structured is not None:
        structured = filter_hidden(structured, hidden)
    return ToolResult(content=content, structured_content=structured)


class ReaderMiddleware(Middleware):
    """Enforces the reader profile on every list and call."""

    def __init__(
        self,
        allowed: frozenset[str],
        hidden_names: Callable[[], Awaitable[frozenset[str]]],
    ):
        self.allowed = allowed
        self.hidden_names = hidden_names

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        return [t for t in tools if t.name in self.allowed]

    async def on_call_tool(self, context, call_next):
        name = context.message.name
        if name not in self.allowed:
            raise ToolError(f"'{name}' is not available in the reader profile")
        try:
            hidden = await self.hidden_names()
        except Exception as e:  # fail closed: never answer unfiltered
            raise ToolError(f"reader profile could not compute withheld nodes: {type(e).__name__}") from e
        result = await call_next(context)
        return filter_result(result, hidden)
