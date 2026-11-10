"""
Jev-powered query router: decide, per question, whether to answer from the
OKF knowledge bundle or from vector RAG — and which line of business to scope
retrieval to.

One TypeSafe System One request asks three independent questions in parallel:

  path      Choice  okf | rag      — which retrieval strategy fits the question
  lob       Choice  4 LOBs         — speculative: which line of business it's about
  one_lob   Noul    yes/no         — companion to `lob`: does exactly one LOB apply?

Policy is plain code (thresholds below), so it can be tuned without touching
the questions:
  * A NAICS code written in the question always routes to OKF (exact lookup).
  * Otherwise take Jev's `path` if its confidence clears PATH_MIN_CONFIDENCE;
    when unsure, fall back to RAG (the broader, semantic path).
  * Apply the LOB filter only when the companion Noul and the LOB pick are
    both confident; otherwise search all LOBs.
  * If TypeSafe is unavailable (no key, network error), fall back to RAG with
    no filter so answering still works.

Usage (CLI):
    python router.py "What NAICS code covers insurance carriers?"
    python router.py --route-only "Does a CGL policy cover completed operations?"
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

from rag_query import RAGResult, format_response

PATH_MIN_CONFIDENCE = 0.6
LOB_MIN_PROBABILITY = 0.7
ONE_LOB_MIN = 0.7
JEV_MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")

LOBS = {
    "Commercial Property": (
        "Commercial property insurance: buildings, business personal property, "
        "construction class, protection, occupancy, exposure, business income, "
        "property eligibility and underwriting."
    ),
    "Commercial General Liability": (
        "Commercial general liability (CGL): bodily injury and property damage to "
        "third parties, premises/operations, products-completed operations, personal "
        "and advertising injury, the ISO CG 00 01 policy form, exclusions and limits."
    ),
    "Personal Auto": (
        "Personal / private passenger auto insurance: liability, collision, "
        "comprehensive, uninsured motorist, PIP, drivers, MVRs, rating and "
        "statistical reporting."
    ),
    "NAICS Classification": (
        "Industry classification with NAICS (USA 2022 or Canada 2017): which code "
        "or industry a business belongs to, what a code covers, how a sector is "
        "organised."
    ),
}

PATH_CRITERIA = {
    "okf": (
        "Structured knowledge lookup. The question names or asks for one specific "
        "entity that would have its own page in a curated knowledge base: a NAICS "
        "code or industry title, the classification of a particular kind of "
        "business, the parent or sub-industries of a code, or what one named "
        "document or form section says."
    ),
    "rag": (
        "Semantic passage search. The question is open-ended or conceptual: it asks "
        "for explanation, eligibility or underwriting reasoning, coverage analysis, "
        "comparison, or a synthesis across several documents, rather than pointing "
        "at one specific entity."
    ),
}

# A candidate code: 2–6 digits, not part of a money amount ($5000, 5,000) or a
# decimal, and not followed by a unit ("1500 sq ft", "25000 BI", "100 miles").
_NAICS_CODE_IN_TEXT = re.compile(
    r"(?<![\d$,.])(\d{2,6})(?![\d,.]\d)(?!\s*(?:%|k\b|sq|square|ft|feet|miles?|mph|"
    r"vehicles?|cars?|employees?|years?|yrs?|days?|months?|bi\b|pd\b|csl\b|tiv\b|"
    r"deductible|limit|dollars?|usd))",
    re.I,
)
_NAICS_CONTEXT = re.compile(r"\b(naics|industry|sector|subsector|classif\w*|code)\b", re.I)


@dataclass
class RouteDecision:
    path: str                          # "okf" | "rag"
    lob_filter: str | None
    reason: str
    path_probabilities: dict[str, float] = field(default_factory=dict)
    path_confidence: float | None = None
    lob_probabilities: dict[str, float] = field(default_factory=dict)
    one_lob: float | None = None


def build_questions() -> dict:
    from typesafe_sdk import Choice, Noul

    lob_list = "; ".join(f"{k}: {v}" for k, v in LOBS.items())
    return {
        "path": Choice(
            instructions=(
                "An underwriting assistant can answer `question` in one of two ways. "
                "Which retrieval strategy fits this question best?"
            ),
            criteria=PATH_CRITERIA,
        ),
        "lob": Choice(
            instructions=(
                "Which line of business is `question` mainly about? Assume it is "
                "about exactly one of these."
            ),
            criteria=LOBS,
        ),
        "one_lob": Noul(
            instructions=(
                "Is `question` about exactly one of these lines of business, rather "
                f"than several of them or none of them? Lines of business — {lob_list}"
            ),
        ),
    }


@lru_cache(maxsize=1)
def known_naics_codes() -> frozenset[str]:
    """Every code with a page in the OKF bundle (read from file names — fast)."""
    from okf_retrieve import OKF_DIR

    return frozenset(
        p.stem for p in OKF_DIR.glob("naics/*/*.md") if p.stem != "index"
    )


def has_naics_code(question: str, codes: frozenset[str] | None = None) -> bool:
    """
    True only when the question cites an actual NAICS code: the number must
    exist in the OKF bundle and not look like an amount, a year or a quantity.
    Six-digit codes count on their own; shorter ones (sectors, groups) need
    NAICS/industry/code wording nearby, since 2–5 digit numbers are common in
    underwriting questions.
    """
    codes = known_naics_codes() if codes is None else codes
    has_context = bool(_NAICS_CONTEXT.search(question))
    for m in _NAICS_CODE_IN_TEXT.finditer(question):
        code = m.group(1)
        if code not in codes:
            continue
        if len(code) == 6 or has_context:
            return True
    return False


def decide(question: str, response) -> RouteDecision:
    """Turn a SystemOneResponse into a routing decision (pure policy, no I/O)."""
    path_ans = response.choices["path"]
    lob_ans = response.choices["lob"]
    one_lob = response.nouls["one_lob"].noul

    lob_p = lob_ans.probabilities.get(lob_ans.choice, 0.0)
    lob_filter = (
        lob_ans.choice if one_lob >= ONE_LOB_MIN and lob_p >= LOB_MIN_PROBABILITY else None
    )

    if has_naics_code(question):
        path, reason = "okf", "question contains a NAICS code → exact OKF lookup"
        lob_filter = "NAICS Classification"
    elif path_ans.confidence >= PATH_MIN_CONFIDENCE:
        path = path_ans.choice
        reason = f"Jev chose {path} (confidence {path_ans.confidence:.2f})"
    else:
        path = "rag"
        reason = (
            f"Jev unsure (leaned {path_ans.choice}, confidence "
            f"{path_ans.confidence:.2f}) → default to rag"
        )

    return RouteDecision(
        path=path,
        lob_filter=lob_filter,
        reason=reason,
        path_probabilities=dict(path_ans.probabilities),
        path_confidence=path_ans.confidence,
        lob_probabilities=dict(lob_ans.probabilities),
        one_lob=one_lob,
    )


def route(question: str, client=None) -> RouteDecision:
    """Ask Jev how to route `question`. Never raises: falls back to RAG."""
    try:
        if client is None:
            from typesafe_sdk import TypeSafeClient

            client = TypeSafeClient(model=JEV_MODEL)
        response = client.system_one({"question": question}, build_questions())
    except Exception as exc:  # missing key, network, rate limit, …
        if has_naics_code(question):
            return RouteDecision("okf", "NAICS Classification", f"router unavailable ({exc.__class__.__name__}); NAICS code → okf")
        return RouteDecision("rag", None, f"router unavailable ({exc.__class__.__name__}) → rag")
    return decide(question, response)


def answer(question: str, client=None) -> tuple[RouteDecision, RAGResult]:
    """Route with Jev, then run the chosen retrieval path."""
    decision = route(question, client=client)
    if decision.path == "okf":
        from okf_retrieve import query_okf

        result = query_okf(question, lob_filter=decision.lob_filter)
    else:
        from rag_query import query

        result = query(question, lob_filter=decision.lob_filter)
    return decision, result


def format_decision(d: RouteDecision) -> str:
    probs = ", ".join(f"{k}={v:.2f}" for k, v in d.path_probabilities.items())
    return (
        f"ROUTE: {d.path.upper()}   LOB filter: {d.lob_filter or 'all'}\n"
        f"  why: {d.reason}"
        + (f"\n  path probabilities: {probs}" if probs else "")
        + (f"\n  one-LOB noul: {d.one_lob:.2f}" if d.one_lob is not None else "")
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    route_only = "--route-only" in args
    q = " ".join(a for a in args if a != "--route-only").strip()
    if not q:
        q = "What NAICS code covers insurance carriers?"
    if route_only:
        print(format_decision(route(q)))
    else:
        decision, result = answer(q)
        print(format_decision(decision))
        print(format_response(result))
