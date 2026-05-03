# Multi-Domain Support Triage Agent

Terminal-based AI agent that triages support tickets across HackerRank, Claude, and Visa, using only the local support corpus shipped under `data/`.

## Architecture at a glance

```
support_issues.csv ─► main.py
                       │
                       ├─► classifier.py     (rules: company, request_type,
                       │                       product_area, risk, escalate?)
                       │
                       ├─► corpus.py         (TF-IDF retrieval over data/)
                       │
                       └─► responder.py      (LLM call w/ corpus grounding,
                                              with a deterministic fallback)
                       │
                       └─► output.csv
```

The flow is intentionally rule-first: every escalation has a named reason, and
the LLM is constrained to phrasing — never to deciding policy. That keeps the
output auditable for the judge interview.

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point; reads CSV, runs the agent, writes predictions |
| `agent.py` | Top-level orchestrator that wires classifier → retriever → responder |
| `classifier.py` | Rule-based routing: company, request_type, product_area, risk |
| `corpus.py` | Loads `data/{hackerrank,claude,visa}/`, builds per-company TF-IDF |
| `responder.py` | LLM-grounded reply (Claude API) + deterministic fallback |
| `schema.py` | Shared dataclasses |
| `requirements.txt` | Pinned dependencies |

## Setup

```bash
cd code
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your API key (optional — see "no-LLM mode" below):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

A `.env.example` is provided. Copy it to `.env` if you want to keep keys
locally; we read from real env vars only.

## Run

Smoke test on the sample file (5 rows):

```bash
python main.py \
    --input  ../support_issues/sample_support_issues.csv \
    --output ../support_issues/sample_output.csv \
    --corpus ../data \
    --limit 5 \
    --verbose
```

Full run on the real input (this is what we submit):

```bash
python main.py \
    --input  ../support_issues/support_issues.csv \
    --output ../support_issues/output.csv \
    --corpus ../data
```

### No-LLM mode

If `ANTHROPIC_API_KEY` isn't set, the agent automatically falls back to
deterministic, corpus-grounded templated replies. You can also force this
with `--no-llm`:

```bash
python main.py --no-llm \
    --input  ../support_issues/support_issues.csv \
    --output ../support_issues/output.csv \
    --corpus ../data
```

This guarantees judges can re-run the submission without needing API
credentials, and the rule-based outputs remain identical run-to-run.

## Design choices

**Why TF-IDF instead of embeddings?**
The corpus is small (hundreds of help-center articles), the user wording
overlaps heavily with the docs, and TF-IDF is deterministic and free.
Embeddings could improve recall on paraphrased queries but at the cost of
either a network dependency or an offline model — neither earns its keep here.
The retriever interface is pluggable (`CorpusIndex.search`), so swapping in an
embedding backend later is a single-file change.

**Why rules for classification, LLM for phrasing?**
Two reasons. (1) The judges want every escalation justified with a clear
reason — rules give us human-readable, deterministic explanations. (2) Rules
can't hallucinate. The LLM only ever sees pre-classified context and
retrieved snippets, and is told never to invent policies.

**How escalation is decided.**
A ticket is escalated if any of these is true:

1. A high-risk pattern fires (fraud, account access, security, legal,
   self-harm, threats, billing dispute, disputed assessment).
2. The ticket is in-scope but retrieval found nothing above a confidence
   threshold (and it isn't a feature request, where "noted" is fine).
3. The pipeline hits an unhandled error (defense-in-depth — see `main.py`).

Out-of-scope or invalid messages are *replied* to with a polite "I can't help
with this", not escalated. Escalation should mean a human needs to act.

**Prompt-injection defense.**
The classifier flags injection patterns and the retriever strips suspicious
lines before searching. The LLM system prompt also instructs the model to
treat embedded instructions as data, never as commands.

## Output schema

The output CSV preserves the input columns (issue, subject, company) for
auditability and adds the five required output columns:

- `status`: `replied` | `escalated`
- `product_area`: e.g. `billing_and_payments`, `account_access`, `assessments_and_tests`
- `response`: user-facing answer (≤ ~1200 chars)
- `justification`: routing rationale (≤ ~350 chars)
- `request_type`: `product_issue` | `feature_request` | `bug` | `invalid`

## Reproducibility

- No randomness anywhere in the pipeline.
- `numpy` / `sklearn` are version-pinned.
- All file ordering is sorted explicitly (`sorted(...)`).
- Re-running on the same input produces byte-identical CSV output in
  `--no-llm` mode, and stable classifications/escalations with the LLM on.
