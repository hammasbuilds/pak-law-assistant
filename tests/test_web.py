"""The web page's server, on the bundled three-provision sample: no dataset needed."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from paklaw import audit, web


@pytest.fixture(scope="module")
def app() -> web.App:
    return web.App(None)  # None = the sample shipped in the package


class TestApp:
    def test_info_says_it_is_the_sample(self, app):
        info = app.info()
        assert info["sample"] is True and info["source"] == "sample"
        assert info["provisions"] >= 1 and info["statutes"] == len(info["statute_list"])
        assert info["samples"] and all({"question", "as_of"} <= set(s) for s in info["samples"])

    def test_an_answer_carries_the_dates_in_force(self, app):
        result = app.ask({"question": "punishment for defamation", "as_of": "2024-01-01"})
        assert result["corpus_warning"]  # the sample says so in every result
        for passage in result["passages"]:
            assert passage["in_force_from"]
            assert "act_title" in passage

    def test_the_date_decides_which_text_is_in_force(self, app):
        """The sample carries PECA s.20 before and after its 2022 substitution."""
        before = app.ask({"question": "section 20 PECA", "as_of": "2021-01-01"})
        after = app.ask({"question": "section 20 PECA", "as_of": "2023-01-01"})
        assert before["passages"] and after["passages"]
        assert before["passages"][0]["in_force_to"] == "2022-02-20"
        assert after["passages"][0]["in_force_to"] is None
        assert before["passages"][0]["text"] != after["passages"][0]["text"]

    def test_a_missing_date_means_today_not_an_error(self, app):
        result = app.ask({"question": "section 20 PECA"})
        assert result["as_of"]

    def test_an_empty_question_is_refused_with_a_message(self, app):
        with pytest.raises(web.InputError, match="question"):
            app.ask({"question": "   ", "as_of": "2024-01-01"})

    def test_an_unknown_statute_is_named_without_listing_every_one(self, app):
        with pytest.raises(web.InputError) as caught:
            app.ask({"question": "theft", "as_of": "2024-01-01", "statute": "XYZ Act"})
        message = str(caught.value)
        assert "'XYZ Act'" in message and "loaded:" not in message and len(message) < 200

    def test_empty_draft_is_refused_with_a_message(self, app):
        with pytest.raises(web.InputError, match="Paste or upload"):
            app.check({"text": "  ", "as_of": "2024-01-01"})

    def test_a_bad_date_is_refused_with_a_message(self, app):
        with pytest.raises(web.InputError):
            app.ask({"question": "theft", "as_of": "1 Jan 2024"})

    def test_check_labels_every_citation(self, app):
        result = app.check(
            {"text": "Charged under section 20 PECA and section 9999 PPC.", "as_of": "2024-01-01"}
        )
        assert result["total"] >= 1
        for citation in result["citations"]:
            assert citation["label"] and citation["tone"] in {"good", "warn", "bad", "plain"}

    def test_every_audit_status_has_a_label(self):
        assert set(audit.ALL_STATUSES) <= set(web.STATUS_LABELS)

    def test_history_lists_each_version(self, app):
        result = app.history({"citation": "section 20 PECA"})
        assert not result["refused"]
        assert len(result["history"]) == 2 and "text" in result["history"][0]


class TestCorpusResolution:
    def test_explicit_beats_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PAKLAW_CORPUS", str(tmp_path / "env.jsonl"))
        assert web.resolve_corpus("given.jsonl") == "given.jsonl"
        assert web.resolve_corpus(None) == str(tmp_path / "env.jsonl")

    def test_without_either_it_is_the_clone_file_or_the_sample(self, monkeypatch):
        monkeypatch.delenv("PAKLAW_CORPUS", raising=False)
        expected = str(web.CLONE_CORPUS) if web.CLONE_CORPUS.exists() else None
        assert web.resolve_corpus(None) == expected


class TestServer:
    @pytest.fixture()
    def base(self, app):
        server = web.make_server(app, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_address[1]}"
        server.shutdown()
        server.server_close()

    def _post(self, url: str, payload: object) -> tuple[int, dict]:
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_the_page_and_info_are_served(self, base):
        with urllib.request.urlopen(base + "/", timeout=10) as response:
            assert "/api/ask" in response.read().decode("utf-8")
        with urllib.request.urlopen(base + "/api/info", timeout=10) as response:
            assert json.loads(response.read())["sample"] is True

    def test_ask_round_trip(self, base):
        status, body = self._post(
            base + "/api/ask", {"question": "section 20 PECA", "as_of": "2023-01-01"}
        )
        assert status == 200 and body["passages"]

    def test_bad_input_is_a_400_with_the_reason(self, base):
        status, body = self._post(base + "/api/ask", {"question": "", "as_of": "2023-01-01"})
        assert status == 400 and "question" in body["error"]

    def test_a_body_that_is_not_an_object_is_a_400(self, base):
        status, body = self._post(base + "/api/check", ["x"])
        assert status == 400 and "JSON object" in body["error"]

    def test_unknown_endpoints_are_404(self, base):
        status, _ = self._post(base + "/api/nothing", {})
        assert status == 404
