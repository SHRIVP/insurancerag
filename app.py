"""
Local web UI for testing the underwriting assistant.

    python app.py            # → http://localhost:7860 (or the next free port)
    PORT=8080 python app.py  # fixed port

Ask a question, let Jev route it (or force OKF / RAG), and see the routing
decision, the grounded answer, and the sources side by side.
"""

from __future__ import annotations

import os
import time

import gradio as gr

import providers
from router import RouteDecision, route

MODES = ["Auto (Jev)", "Force OKF", "Force RAG"]

EXAMPLES = [
    ["What does NAICS 524126 cover?", "Auto (Jev)"],
    ["How does NAICS Canada classify real estate businesses?", "Auto (Jev)"],
    ["What sub-industries are under insurance carriers?", "Auto (Jev)"],
    ["Does a CGL policy cover completed operations claims?", "Auto (Jev)"],
    ["What are eligibility criteria for frame construction commercial property?", "Auto (Jev)"],
    ["What is uninsured motorist coverage?", "Auto (Jev)"],
    ["How do auto and property deductibles compare?", "Auto (Jev)"],
]


def _decision_md(d: RouteDecision, seconds: float | None = None) -> str:
    lines = [
        f"### Route: **{d.path.upper()}**",
        f"- **LOB filter:** {d.lob_filter or 'all lines'}",
        f"- **Why:** {d.reason}",
    ]
    if d.path_probabilities:
        probs = " · ".join(f"{k} {v:.2f}" for k, v in d.path_probabilities.items())
        lines.append(f"- **Path probabilities:** {probs} (confidence {d.path_confidence:.2f})")
    if d.lob_probabilities:
        top = sorted(d.lob_probabilities.items(), key=lambda kv: kv[1], reverse=True)[:2]
        lines.append("- **LOB guess:** " + " · ".join(f"{k} {v:.2f}" for k, v in top))
    if d.one_lob is not None:
        lines.append(f"- **Exactly one LOB (Noul):** {d.one_lob:.2f}")
    if seconds is not None:
        lines.append(f"- **Answered in:** {seconds:.1f}s")
    return "\n".join(lines)


def ask(question: str, mode: str):
    question = (question or "").strip()
    if not question:
        return "Enter a question.", "", []

    t0 = time.time()
    if mode == "Force OKF":
        decision = RouteDecision("okf", None, "forced by user")
    elif mode == "Force RAG":
        decision = RouteDecision("rag", None, "forced by user")
    else:
        decision = route(question)

    try:
        if decision.path == "okf":
            from okf_retrieve import query_okf

            result = query_okf(question, lob_filter=decision.lob_filter)
        else:
            from rag_query import query

            result = query(question, lob_filter=decision.lob_filter)
    except Exception as exc:
        return _decision_md(decision), f"**Error:** `{exc.__class__.__name__}`: {exc}", []

    rows, seen = [], set()
    for c in sorted(result.citations, key=lambda c: c.relevance_score, reverse=True):
        key = (c.source, c.page)
        if key in seen:
            continue
        seen.add(key)
        rows.append([round(c.relevance_score, 3), c.source, c.page, c.lob])
    return _decision_md(decision, time.time() - t0), result.answer, rows


with gr.Blocks(title="Underwriting Assistant") as demo:
    gr.Markdown(
        "# P&C Underwriting Assistant\n"
        f"Jev routes each question to **OKF** (structured knowledge bundle) or **RAG** "
        f"(vector search). Embeddings: `{providers.EMBED_PROVIDER}` · "
        f"generation: `{providers.GENERATE_PROVIDER}`"
        + (f" (`{providers.OLLAMA_GENERATE_MODEL}`)" if providers.GENERATE_PROVIDER == "ollama" else "")
    )
    with gr.Row():
        question = gr.Textbox(
            label="Question", placeholder="e.g. What does NAICS 524126 cover?", scale=4, lines=2
        )
        mode = gr.Radio(MODES, value=MODES[0], label="Retrieval", scale=1)
    submit = gr.Button("Ask", variant="primary")
    with gr.Row():
        with gr.Column(scale=1):
            decision_out = gr.Markdown(label="Routing")
        with gr.Column(scale=2):
            answer_out = gr.Markdown(label="Answer")
    sources_out = gr.Dataframe(
        headers=["Relevance", "Source", "Page", "LOB"],
        label="Sources",
        interactive=False,
    )
    gr.Examples(EXAMPLES, inputs=[question, mode])

    outputs = [decision_out, answer_out, sources_out]
    submit.click(ask, [question, mode], outputs)
    question.submit(ask, [question, mode], outputs)


if __name__ == "__main__":
    # No PORT set → Gradio takes the first free port from 7860 upward.
    port = int(os.environ["PORT"]) if os.environ.get("PORT") else None
    demo.launch(server_name="127.0.0.1", server_port=port)
