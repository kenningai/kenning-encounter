"""Unit tests for matcher voices (v0.17.0).

A voice is one model that both compresses a sidecar and matches against it.
These pin the three guarantees the release rests on: a sidecar is never read
or extended by a voice that did not write it; the lead is configuration,
with a per-locus alternation that is stable across restarts; and a failing
lead hands the call to the next voice inside one shared budget, cooling the
voice that failed. Every guard is shown saying both yes and no.
"""

import argparse
import json

import pytest

from kenning_encounter.infuse import matcher_fallback_note
from kenning_encounter.meaning import (
    MIN_ATTEMPT_MS,
    MeaningIndex,
    MeaningUnavailable,
    Voice,
    VoiceHealth,
    compress_request,
    match_chain,
    match_request,
    response_text,
    response_usage,
    shadow_record,
    sidecar_path,
    voice_order,
    voice_summary,
)
from kenning_encounter.utils import process_config

GEM = Voice("gemini", "gemini-3.5-flash-lite", "gk", "")
OAI = Voice("openai", "gpt-6-luna", "ok", "")


def _write(path, voice, nodes):
    data = {"prompt_version": "v2", "nodes": nodes}
    if voice is not None:
        data["voice"] = voice
    path.write_text(json.dumps(data))


NODES = {"Alpha": {"type": "Concept", "hash": "h", "meaning": "alpha means a"}}


class TestVoice:
    def test_id_and_key_never_in_repr(self):
        assert GEM.id == "gemini:gemini-3.5-flash-lite"
        assert "gk" not in repr(GEM) and "ok" not in repr(OAI)

    def test_configured_needs_key_and_model(self):
        assert GEM.configured
        assert not Voice("openai", "gpt-6-luna", "").configured

    def test_default_endpoints(self):
        assert GEM.base == "https://generativelanguage.googleapis.com/v1beta"
        assert OAI.base == "https://api.openai.com/v1"
        assert Voice("openai", "m", "k", "http://h/v1/").base == "http://h/v1"

    def test_sidecar_path_is_per_voice_and_never_the_base(self):
        g = sidecar_path("/app/models/meaning_sidecar.json", GEM)
        o = sidecar_path("/app/models/meaning_sidecar.json", OAI)
        assert g.name == "meaning_sidecar.gemini-gemini-3.5-flash-lite.json"
        assert o.name == "meaning_sidecar.openai-gpt-6-luna.json"
        assert g != o and g.parent == o.parent
        assert g.name != "meaning_sidecar.json"


class TestVoiceStamp:
    def test_own_stamp_is_read(self, tmp_path):
        p = tmp_path / "s.json"
        _write(p, GEM.id, NODES)
        prefix, names = MeaningIndex(p, GEM.id).prefix()
        assert names == ["Alpha"] and "alpha means a" in prefix

    def test_another_voices_stamp_is_refused(self, tmp_path):
        p = tmp_path / "s.json"
        _write(p, OAI.id, NODES)
        with pytest.raises(MeaningUnavailable) as exc:
            MeaningIndex(p, GEM.id).prefix()
        assert OAI.id in str(exc.value) and GEM.id in str(exc.value)

    def test_an_unstamped_file_is_refused(self, tmp_path):
        p = tmp_path / "s.json"
        _write(p, None, NODES)
        with pytest.raises(MeaningUnavailable) as exc:
            MeaningIndex(p, GEM.id).prefix()
        assert "unrecorded voice" in str(exc.value)

    def test_unbound_index_still_reads_unstamped(self, tmp_path):
        # The offline builder and older callers bind no voice.
        p = tmp_path / "s.json"
        _write(p, None, NODES)
        assert MeaningIndex(p).prefix()[1] == ["Alpha"]

    def test_upsert_stamps_and_never_carries_a_foreign_voice(self, tmp_path):
        p = tmp_path / "s.json"
        _write(p, OAI.id, NODES)
        idx = MeaningIndex(p, GEM.id)
        idx.upsert({"Beta": {"type": "Note", "hash": "b", "meaning": "beta"}})
        data = json.loads(p.read_text())
        assert data["voice"] == GEM.id
        assert list(data["nodes"]) == ["Beta"]  # Alpha was the other voice's

    def test_upsert_merges_its_own_voice(self, tmp_path):
        p = tmp_path / "s.json"
        _write(p, GEM.id, NODES)
        MeaningIndex(p, GEM.id).upsert(
            {"Beta": {"type": "Note", "hash": "b", "meaning": "beta"}}
        )
        assert list(json.loads(p.read_text())["nodes"]) == ["Alpha", "Beta"]

    def test_foreign_file_reads_as_unbuilt_so_reconcile_rebuilds(self, tmp_path):
        p = tmp_path / "s.json"
        _write(p, OAI.id, NODES)
        idx = MeaningIndex(p, GEM.id)
        assert idx.known_nodes() == {} and idx.file_version() == ""


class TestVoiceOrder:
    def test_named_lead_goes_first(self):
        assert voice_order([GEM, OAI], "openai", "s") == [OAI, GEM]
        assert voice_order([GEM, OAI], "gemini", "s") == [GEM, OAI]

    def test_alternate_is_stable_per_session_and_splits_sessions(self):
        firsts = {voice_order([GEM, OAI], "alternate", f"s{i}")[0] for i in range(40)}
        assert firsts == {GEM, OAI}
        for sid in ("a", "b", "c"):
            assert voice_order([GEM, OAI], "alternate", sid) == voice_order(
                [GEM, OAI], "alternate", sid
            )

    def test_alternate_without_session_takes_the_first(self):
        assert voice_order([GEM, OAI], "alternate", None)[0] == GEM

    def test_single_voice(self):
        assert voice_order([GEM], "openai", "s") == [GEM]


class TestVoiceHealth:
    def test_cools_then_recovers(self):
        now = [0.0]
        h = VoiceHealth(cooldown_s=300, clock=lambda: now[0])
        assert not h.cooling(GEM.id)
        h.failed(GEM.id)
        assert h.cooling(GEM.id)
        now[0] = 301
        assert not h.cooling(GEM.id)

    def test_ok_clears(self):
        h = VoiceHealth()
        h.failed(GEM.id)
        h.ok(GEM.id)
        assert not h.cooling(GEM.id)


class _Idx:
    """A stand-in index whose prefix names its voice, so a test can prove
    each voice matched against its own sidecar."""

    def __init__(self, tag, size=3):
        self.tag = tag
        self.size = size

    def prefix(self):
        return f"PREFIX-{self.tag}", ["n1", "n2", "n3"]


def _matcher(script):
    """script: voice_id -> 'ok' | exception message. Records each call."""
    calls = []

    async def run(prefix, names, traj, top_n, voice, timeout_ms, include_reasons):
        calls.append((voice.id, prefix, timeout_ms))
        out = script[voice.id]
        if out != "ok":
            raise MeaningUnavailable(out)
        return {"selections": [{"name": "n1"}], "ms": 10.0,
                "prompt_tokens": 100, "cached_tokens": 90, "raw": "1"}

    return run, calls


class TestMatchChain:
    @pytest.mark.asyncio
    async def test_lead_succeeds_alone(self):
        run, calls = _matcher({GEM.id: "ok", OAI.id: "ok"})
        out = await match_chain([(GEM, _Idx("g")), (OAI, _Idx("o"))],
                                "t", 5, VoiceHealth(), 5000, matcher=run)
        assert out["voice"] == GEM and [c[0] for c in calls] == [GEM.id]
        assert out["attempts"] == [{"voice": GEM.id, "ms": 10.0}]

    @pytest.mark.asyncio
    async def test_failed_lead_falls_back_on_its_own_sidecar_and_cools(self):
        run, calls = _matcher({GEM.id: "match failed: 503", OAI.id: "ok"})
        h = VoiceHealth()
        out = await match_chain([(GEM, _Idx("g")), (OAI, _Idx("o"))],
                                "t", 5, h, 5000, matcher=run)
        assert out["voice"] == OAI
        assert calls[1][1] == "PREFIX-o"  # the fallback read its own meanings
        assert out["attempts"][0] == {"voice": GEM.id, "error": "match failed: 503"}
        assert h.cooling(GEM.id) and not h.cooling(OAI.id)

    @pytest.mark.asyncio
    async def test_a_cooling_lead_is_tried_after_the_healthy_voice(self):
        run, calls = _matcher({GEM.id: "ok", OAI.id: "ok"})
        h = VoiceHealth()
        h.failed(GEM.id)
        out = await match_chain([(GEM, _Idx("g")), (OAI, _Idx("o"))],
                                "t", 5, h, 5000, matcher=run)
        assert out["voice"] == OAI and [c[0] for c in calls] == [OAI.id]

    @pytest.mark.asyncio
    async def test_all_cooling_are_still_tried(self):
        run, calls = _matcher({GEM.id: "ok", OAI.id: "ok"})
        h = VoiceHealth()
        h.failed(GEM.id)
        h.failed(OAI.id)
        out = await match_chain([(GEM, _Idx("g")), (OAI, _Idx("o"))],
                                "t", 5, h, 5000, matcher=run)
        assert out["voice"] == GEM

    @pytest.mark.asyncio
    async def test_fallback_gets_the_remaining_budget_or_is_skipped(self):
        # The lead burns the budget down to below the attempt floor.
        now = [0.0]
        run, calls = _matcher({GEM.id: "match failed: ReadTimeout", OAI.id: "ok"})

        async def slow(*a, **k):
            now[0] += (5000 - MIN_ATTEMPT_MS + 100) / 1000
            return await run(*a, **k)

        with pytest.raises(MeaningUnavailable) as exc:
            await match_chain([(GEM, _Idx("g")), (OAI, _Idx("o"))], "t", 5,
                              VoiceHealth(), 5000, clock=lambda: now[0], matcher=slow)
        assert [c[0] for c in calls] == [GEM.id]
        assert exc.value.attempts[1]["voice"] == OAI.id
        assert exc.value.attempts[1]["error"].startswith("skipped:")

    @pytest.mark.asyncio
    async def test_a_fast_failure_leaves_the_fallback_most_of_the_budget(self):
        now = [0.0]
        run, calls = _matcher({GEM.id: "match failed: 503", OAI.id: "ok"})

        async def quick(*a, **k):
            now[0] += 0.3
            return await run(*a, **k)

        out = await match_chain([(GEM, _Idx("g")), (OAI, _Idx("o"))], "t", 5,
                                VoiceHealth(), 5000, clock=lambda: now[0], matcher=quick)
        assert out["voice"] == OAI
        assert calls[1][2] == pytest.approx(4700)

    @pytest.mark.asyncio
    async def test_unconfigured_voice_is_reported_not_called(self):
        run, calls = _matcher({GEM.id: "ok"})
        bare = Voice("openai", "gpt-6-luna", "")
        out = await match_chain([(bare, _Idx("o")), (GEM, _Idx("g"))],
                                "t", 5, VoiceHealth(), 5000, matcher=run)
        assert out["voice"] == GEM and [c[0] for c in calls] == [GEM.id]
        assert "not configured (OPENAI_API_KEY)" in out["attempts"][0]["error"]

    @pytest.mark.asyncio
    async def test_every_voice_failing_raises_with_the_account(self):
        run, _ = _matcher({GEM.id: "e1", OAI.id: "e2"})
        with pytest.raises(MeaningUnavailable) as exc:
            await match_chain([(GEM, _Idx("g")), (OAI, _Idx("o"))],
                              "t", 5, VoiceHealth(), 5000, matcher=run)
        assert [a["error"] for a in exc.value.attempts] == ["e1", "e2"]
        assert GEM.id in str(exc.value) and OAI.id in str(exc.value)

    @pytest.mark.asyncio
    async def test_a_refused_sidecar_falls_through(self, tmp_path):
        p = tmp_path / "g.json"
        _write(p, OAI.id, NODES)  # the Gemini voice's file, written by another
        run, calls = _matcher({GEM.id: "ok", OAI.id: "ok"})
        out = await match_chain([(GEM, MeaningIndex(p, GEM.id)), (OAI, _Idx("o"))],
                                "t", 5, VoiceHealth(), 5000, matcher=run)
        assert out["voice"] == OAI and [c[0] for c in calls] == [OAI.id]
        assert "refusing" in out["attempts"][0]["error"]


class TestRequests:
    def test_openai_match_shape(self):
        url, headers, body = match_request(OAI, "PREFIX", "\nTAIL")
        assert url == "https://api.openai.com/v1/responses"
        assert headers == {"Authorization": "Bearer ok"}
        assert body["store"] is False
        assert body["reasoning"] == {"effort": "none"}
        assert body["prompt_cache_key"]
        dev, user = body["input"]
        assert dev["role"] == "developer" and dev["content"][0]["text"] == "PREFIX"
        assert user["content"][0]["text"] == "TAIL"

    def test_gemini_production_shape_is_split(self):
        # v0.18.1: the meanings in systemInstruction, the trajectory as the
        # user turn.
        url, headers, body = match_request(GEM, "PREFIX", "\nTAIL")
        assert url.endswith("/models/gemini-3.5-flash-lite:generateContent")
        assert headers == {"x-goog-api-key": "gk"}
        assert body["systemInstruction"]["parts"][0]["text"] == "PREFIX"
        assert body["contents"][0]["parts"][0]["text"] == "TAIL"
        assert body["generationConfig"]["temperature"] == 0

    def test_gemini_flat_is_still_available_for_comparison(self):
        _, _, body = match_request(GEM, "PREFIX", "\nTAIL", shape="flat")
        assert body["contents"][0]["parts"][0]["text"] == "PREFIX\nTAIL"
        assert "systemInstruction" not in body

    def test_the_other_shapes_exist_for_the_comparison(self):
        _, _, of = match_request(OAI, "PREFIX", "\nTAIL", shape="flat")
        assert [m["role"] for m in of["input"]] == ["user"]
        assert of["input"][0]["content"][0]["text"] == "PREFIX\nTAIL"
        _, _, gs = match_request(GEM, "PREFIX", "\nTAIL", shape="split")
        assert gs["systemInstruction"]["parts"][0]["text"] == "PREFIX"
        assert gs["contents"][0]["parts"][0]["text"] == "TAIL"
        # Native shapes are unchanged by the option existing.
        assert match_request(OAI, "P", "\nT") == match_request(OAI, "P", "\nT", shape="split")
        assert match_request(GEM, "P", "\nT") == match_request(GEM, "P", "\nT", shape="split")

    def test_unknown_shape_is_refused(self):
        with pytest.raises(ValueError):
            match_request(GEM, "P", "T", shape="sideways")

    def test_reasoning_is_read_per_call(self):
        from kenning_encounter.meaning import response_generation

        assert response_generation(OAI, {"usage": {"output_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 12}}}) == (30, 12)
        assert response_generation(GEM, {"usageMetadata": {"candidatesTokenCount": 40,
            "thoughtsTokenCount": 7}}) == (40, 7)
        assert response_generation(GEM, {}) == (0, 0)

    def test_compress_shapes(self):
        _, _, ob = compress_request(OAI, "P")
        assert ob["input"] == "P" and ob["store"] is False
        _, _, gb = compress_request(GEM, "P")
        assert gb["contents"][0]["parts"][0]["text"] == "P"

    def test_openai_response_parsing(self):
        data = {
            "output": [
                {"type": "reasoning", "content": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "3\n"},
                    {"type": "output_text", "text": "1"},
                ]},
            ],
            "usage": {"input_tokens": 75905, "input_tokens_details": {"cached_tokens": 75853}},
        }
        assert response_text(OAI, data) == "3\n1"
        assert response_usage(OAI, data) == (75905, 75853)

    def test_gemini_response_parsing(self):
        data = {"candidates": [{"content": {"parts": [{"text": "2"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "cachedContentTokenCount": 8}}
        assert response_text(GEM, data) == "2"
        assert response_usage(GEM, data) == (10, 8)


class TestShadowRecord:
    def test_mechanics_only(self):
        m = {"selections": [{"name": "n1"}, {"name": "n2"}], "ms": 9.0,
             "prompt_tokens": 5, "cached_tokens": 4}
        rec = shadow_record("abc", "sess", "alternate",
                            voice_summary(GEM, m, _Idx("g")),
                            voice_summary(OAI, None, _Idx("o", 2), error="429"))
        assert rec["match_id"] == "abc" and rec["lead"] == "alternate"
        assert rec["delivered"]["selections"] == ["n1", "n2"]
        assert rec["shadow"] == {"voice": OAI.id, "sidecar_size": 2, "error": "429"}
        assert "trajectory" not in json.dumps(rec)


class TestFallbackVoiceNote:
    def test_note_when_the_fallback_voice_delivered(self):
        note = matcher_fallback_note({
            "channel": "meaning", "voice": OAI.id,
            "attempts": [{"voice": GEM.id, "error": "match failed: 503"},
                         {"voice": OAI.id, "ms": 900}],
        })
        assert note is not None
        assert note.startswith("[matcher lead unavailable")
        assert "503" in note

    def test_note_names_neither_voice(self):
        # The delivering voice is the compared variable; the subject reads
        # the header.
        note = matcher_fallback_note({
            "channel": "meaning",
            "attempts": [{"voice": GEM.id, "error": "boom"}, {"voice": OAI.id, "ms": 1}],
        })
        assert note is not None and GEM.id not in note and OAI.id not in note

    def test_note_while_the_lead_is_cooling(self):
        # The lead is skipped without an attempt, so no error is recorded;
        # the header must still say the lead is not the one matching.
        note = matcher_fallback_note({
            "channel": "meaning", "lead": GEM.id, "voice": OAI.id,
            "attempts": [{"voice": OAI.id, "ms": 900}],
        })
        assert note is not None and "cooling" in note

    def test_no_note_when_the_lead_delivered(self):
        assert matcher_fallback_note({
            "channel": "meaning", "lead": GEM.id, "voice": GEM.id,
            "attempts": [{"voice": GEM.id, "ms": 900}],
        }) is None


class TestVoiceConfig:
    def _args(self, **overrides):
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
    def _clean(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "NEO4J_MATCHER_OPENAI_MODEL",
                    "NEO4J_MATCHER_OPENAI_ENDPOINT", "NEO4J_MATCHER_LEAD",
                    "NEO4J_MATCHER_COOLDOWN_S", "NEO4J_MATCHER_SHADOW",
                    "NEO4J_MATCHER_SHADOW_LOG", "NEO4J_MATCHER_SIDECAR"):
            monkeypatch.delenv(var, raising=False)

    def test_defaults_change_nothing_for_an_existing_deployment(self):
        cfg = process_config(self._args())
        assert cfg["matcher_openai_api_key"] == ""
        assert cfg["matcher_openai_model"] == "gpt-6-luna"
        assert cfg["matcher_lead"] == "gemini"
        assert cfg["matcher_cooldown_s"] == 300
        assert cfg["matcher_shadow"] is False
        assert cfg["matcher_shadow_log"] == "models/matcher-shadow.jsonl"

    def test_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("NEO4J_MATCHER_LEAD", "Alternate")
        monkeypatch.setenv("NEO4J_MATCHER_SHADOW", "on")
        monkeypatch.setenv("NEO4J_MATCHER_COOLDOWN_S", "99999")
        cfg = process_config(self._args())
        assert cfg["matcher_openai_api_key"] == "sk-test"
        assert cfg["matcher_lead"] == "alternate"
        assert cfg["matcher_shadow"] is True
        assert cfg["matcher_cooldown_s"] == 3600

    def test_unknown_lead_falls_back_to_gemini(self, monkeypatch):
        monkeypatch.setenv("NEO4J_MATCHER_LEAD", "claude")
        assert process_config(self._args())["matcher_lead"] == "gemini"

    def test_shadow_log_sits_beside_the_sidecar(self, monkeypatch):
        monkeypatch.setenv("NEO4J_MATCHER_SIDECAR", "/app/models/meaning_sidecar.json")
        assert process_config(self._args())["matcher_shadow_log"] == (
            "/app/models/matcher-shadow.jsonl"
        )


class TestAdoptLegacySidecar:
    def test_adopts_an_unstamped_file_in_order(self, tmp_path):
        from kenning_encounter.meaning import adopt_legacy_sidecar

        base = tmp_path / "meaning_sidecar.json"
        nodes = {"Zed": {"meaning": "z"}, "Alpha": {"meaning": "a"}}
        _write(base, None, nodes)
        dst = adopt_legacy_sidecar(base, GEM)
        assert dst == sidecar_path(base, GEM)
        assert MeaningIndex(dst, GEM.id).prefix()[1] == ["Zed", "Alpha"]
        assert base.exists()  # the source is left in place

    def test_refuses_a_stamped_source(self, tmp_path):
        from kenning_encounter.meaning import adopt_legacy_sidecar

        base = tmp_path / "meaning_sidecar.json"
        _write(base, OAI.id, NODES)
        with pytest.raises(MeaningUnavailable, match="already stamped"):
            adopt_legacy_sidecar(base, GEM)

    def test_refuses_to_overwrite(self, tmp_path):
        from kenning_encounter.meaning import adopt_legacy_sidecar

        base = tmp_path / "meaning_sidecar.json"
        _write(base, None, NODES)
        adopt_legacy_sidecar(base, GEM)
        with pytest.raises(MeaningUnavailable, match="already exists"):
            adopt_legacy_sidecar(base, GEM)

    def test_cli_requires_a_named_voice(self, tmp_path, capsys):
        from kenning_encounter.meaning import _main

        base = tmp_path / "meaning_sidecar.json"
        _write(base, None, NODES)
        with pytest.raises(SystemExit):
            _main(["--adopt", "gemini", "--sidecar", str(base)])
        assert _main(["--adopt", "gemini:gemini-3.5-flash-lite",
                      "--sidecar", str(base)]) == 0
        assert _main(["--adopt", "gemini:gemini-3.5-flash-lite",
                      "--sidecar", str(base)]) == 1
