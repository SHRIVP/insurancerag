import httpx
import pytest

import providers


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._data


def test_collection_names_are_separated_by_provider():
    assert providers.collection_name("cohere") == "underwriting_guidelines"
    ollama = providers.collection_name("ollama")
    assert ollama.startswith("underwriting_guidelines__") and ollama != "underwriting_guidelines"


def test_ollama_query_gets_instruction_prefix_documents_do_not(monkeypatch):
    sent = []
    monkeypatch.setattr(providers, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: sent.append(json) or _Resp({"embeddings": [[0.1]] * len(json["input"])}))
    providers.embed(["what is CGL"], "search_query")
    providers.embed(["CGL covers..."], "search_document")
    assert sent[0]["input"][0].startswith("Instruct:") and sent[0]["input"][0].endswith("what is CGL")
    assert sent[1]["input"] == ["CGL covers..."]


def test_ollama_generate_extracts_valid_citations(monkeypatch):
    monkeypatch.setattr(providers, "GENERATE_PROVIDER", "ollama")
    payload = {"message": {"content": "Covered [0]. Also excluded [1, 7]. See [9]."}}
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: _Resp(payload))
    docs = [{"id": "0", "title": "a", "snippet": "x"}, {"id": "1", "title": "b", "snippet": "y"}]
    text, spans = providers.generate_answer("q", docs)
    assert text.startswith("Covered")
    assert [s["document_ids"] for s in spans] == [["0"], ["1"]]  # 7 and 9 aren't real docs


def test_ollama_error_is_actionable(monkeypatch):
    monkeypatch.setattr(providers, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: _Resp({"error": "model not found"}, 404))
    with pytest.raises(RuntimeError, match="ollama pull"):
        providers.embed(["x"], "search_document")
