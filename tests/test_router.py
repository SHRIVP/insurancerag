from typesafe_sdk import SystemOneResponse

import router


def _resp(path_probs, path_conf, lob, lob_p, one_lob):
    lob_probs = {k: (lob_p if k == lob else (1 - lob_p) / 3) for k in router.LOBS}
    return SystemOneResponse.model_validate({
        "model": "jev", "usage": {"input_tokens": 1, "output_tokens": 1},
        "answers": {
            "path": {"type": "choice", "choice": max(path_probs, key=path_probs.get),
                     "confidence": path_conf, "probabilities": path_probs},
            "lob": {"type": "choice", "choice": lob, "confidence": 0.9, "probabilities": lob_probs},
            "one_lob": {"type": "noul", "noul": one_lob},
        },
    })


class FakeClient:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def system_one(self, state, questions):
        self.calls.append((state, set(questions)))
        if self.exc:
            raise self.exc
        return self.response


def test_confident_okf_with_lob_filter():
    c = FakeClient(_resp({"okf": 0.97, "rag": 0.03}, 0.95, "NAICS Classification", 0.9, 0.9))
    d = router.route("How does NAICS Canada classify real estate?", client=c)
    assert (d.path, d.lob_filter) == ("okf", "NAICS Classification")
    assert c.calls[0] == ({"question": "How does NAICS Canada classify real estate?"}, {"path", "lob", "one_lob"})


def test_low_confidence_falls_back_to_rag():
    d = router.route("Does CGL cover completed ops?",
                     client=FakeClient(_resp({"okf": 0.52, "rag": 0.48}, 0.04, "Commercial General Liability", 0.95, 0.98)))
    assert (d.path, d.lob_filter) == ("rag", "Commercial General Liability")


def test_lob_filter_needs_companion_noul():
    d = router.route("Compare auto and property deductibles",
                     client=FakeClient(_resp({"okf": 0.0, "rag": 1.0}, 1.0, "Personal Auto", 0.8, 0.1)))
    assert (d.path, d.lob_filter) == ("rag", None)


def test_naics_code_overrides_to_okf(monkeypatch):
    monkeypatch.setattr(router, "known_naics_codes", lambda: CODES)
    d = router.route("What does 524126 cover?",
                     client=FakeClient(_resp({"okf": 0.1, "rag": 0.9}, 0.9, "Personal Auto", 0.9, 0.9)))
    assert (d.path, d.lob_filter) == ("okf", "NAICS Classification")


def test_service_failure_falls_back(monkeypatch):
    monkeypatch.setattr(router, "known_naics_codes", lambda: CODES)
    assert router.route("What is CGL?", client=FakeClient(exc=RuntimeError("down"))).path == "rag"
    assert router.route("NAICS 524126?", client=FakeClient(exc=RuntimeError("down"))).path == "okf"


CODES = frozenset({"52", "23", "1111", "5241", "524126", "236115"})


def test_has_naics_code_accepts_real_codes():
    assert router.has_naics_code("What does 524126 cover?", CODES)
    assert router.has_naics_code("NAICS 52 overview", CODES)
    assert router.has_naics_code("Which industries are in sector 23?", CODES)
    assert router.has_naics_code("industry group 5241", CODES)


def test_has_naics_code_ignores_ordinary_numbers():
    for q in [
        "Is a $5000 deductible typical for commercial property?",
        "Can we write a 1500 sq ft frame building?",
        "What limits apply for 2 vehicles with 25000 BI?",
        "How should a 150000 TIV warehouse be rated?",
        "Is the 2022 form different?",
        "Is a 1111 sq ft building eligible?",       # real code, but a quantity
        "Quote 52 vehicles for a fleet",            # real sector, no NAICS wording
        "A $524,126 claim was filed",               # money, not a code
    ]:
        assert not router.has_naics_code(q, CODES), q
