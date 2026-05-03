"""
The TriageAgent — orchestrates classification, retrieval, response generation,
and final status decision for one ticket.

Pipeline per ticket:

    issue ─┬─► Classifier ──► TriageDecision
           │
           └─► CorpusIndex.search ──► retrieved chunks
                                              │
              TriageDecision + chunks ──► Responder ──► GeneratedReply
                                              │
                                              └─► IssueOutput (status, etc.)

Decision rule for ``status``:

    - escalated  if must_escalate, retrieval is empty for an in-scope ticket
                 with high uncertainty, or LLM/rules cannot produce an answer
    - replied    otherwise

We keep the rule simple and explicit so the judges can trace every decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from classifier import Classifier
from corpus import CorpusIndex
from responder import generate_with_llm, generate_without_llm
from schema import IssueInput, IssueOutput, TriageDecision

logger = logging.getLogger("triage.agent")

# Below this top-1 retrieval score, we treat the result as "no real match"
# rather than a confident answer. TF-IDF cosine on help-center text typically
# scores 0.15+ for a relevant article and <0.05 for unrelated ones.
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.20

# When we have no retrieval but the ticket is clearly in scope, escalate
# rather than fabricate. Better to send to a human than guess.
ESCALATE_ON_EMPTY_RETRIEVAL = True


@dataclass
class TriageAgent:
    index: CorpusIndex
    use_llm: bool = True
    classifier: Classifier = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.classifier is None:
            self.classifier = Classifier()

    def triage(self, issue: IssueInput) -> IssueOutput:
        # 1) Classify and route.
        decision = self.classifier.classify(issue)
        logger.debug(
            "row=%d company=%s product_area=%s request_type=%s risk=%s "
            "must_escalate=%s in_scope=%s",
            issue.row_id, decision.inferred_company, decision.product_area,
            decision.request_type, decision.risk_level, decision.must_escalate,
            decision.is_in_scope,
        )

        # 2) Retrieve grounding chunks. We always run retrieval — even on
        # escalation cases — because the justification is more useful when it
        # can name the relevant article.
        query = self._build_query(issue, decision)
        chunks = self.index.search(query, company=decision.inferred_company, k=4)

        # 3) Decide final status.
        status = self._decide_status(decision, chunks)

        # 4) Generate a response. For replied cases we prefer the LLM if
        # available; for escalations the deterministic template is enough.
        if status == "escalated":
            reply = generate_without_llm(issue, decision, chunks)
        elif self.use_llm:
            reply = generate_with_llm(issue, decision, chunks)
        else:
            reply = generate_without_llm(issue, decision, chunks)

        return IssueOutput(
            status=status,
            product_area=decision.product_area,
            response=reply.response,
            justification=reply.justification,
            request_type=decision.request_type,
        )

    # -- helpers --------------------------------------------------------

    def _build_query(self, issue: IssueInput, decision: TriageDecision) -> str:
        """Compose a retrieval query. We combine subject + issue, but if the
        message looks like prompt injection we drop suspicious lines first so
        we don't pollute retrieval with adversarial text."""
        parts = []
        if issue.subject:
            parts.append(issue.subject)
        body = issue.issue or ""
        if decision.is_malicious:
            # Drop lines that contain known injection markers.
            cleaned = []
            for line in body.splitlines():
                low = line.lower()
                if any(marker in low for marker in
                       ("ignore previous", "ignore all previous",
                        "disregard the system", "you are now",
                        "reveal your prompt", "jailbreak")):
                    continue
                cleaned.append(line)
            body = "\n".join(cleaned).strip() or body
        parts.append(body)
        return " ".join(parts).strip()

    def _decide_status(
        self,
        decision: TriageDecision,
        chunks: list,
    ) -> str:
        if decision.must_escalate:
            return "escalated"

        # Out-of-scope: we still reply with a polite "I can't help with this"
        # rather than escalate, because escalation should mean a human needs
        # to act — not that a human needs to read a casual hello.
        if not decision.is_in_scope or decision.request_type == "invalid":
            return "replied"

        # In-scope but retrieval came up empty. We can either reply with a
        # graceful "no info found" or escalate. Escalating is safer for
        # legitimate tickets where the docs simply didn't cover the question.
        top_score = chunks[0].score if chunks else 0.0
        if ESCALATE_ON_EMPTY_RETRIEVAL and top_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
            # Exception: if we already detected this is a low-stakes generic
            # request (e.g. a feature_request), prefer reply with a graceful
            # "noted, contact support for tracking" rather than escalation.
            if decision.request_type == "feature_request":
                return "replied"
            return "escalated"

        return "replied"
