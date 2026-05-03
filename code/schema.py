"""Data classes for the triage pipeline. Kept dependency-free."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# These literals come straight from problem_statement.md. We keep them as
# string constants so a typo on our side becomes a load-time error.
Status = Literal["replied", "escalated"]
RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]
Company = Literal["HackerRank", "Claude", "Visa"]


@dataclass
class IssueInput:
    row_id: int
    issue: str
    subject: str
    company: Optional[str]  # "HackerRank" | "Claude" | "Visa" | None


@dataclass
class RetrievedChunk:
    company: str
    title: str
    source_path: str
    text: str
    score: float


@dataclass
class TriageDecision:
    """Intermediate state — the result of routing/classification before we
    decide to compose a response."""
    inferred_company: Optional[str]
    product_area: str
    request_type: RequestType
    risk_level: Literal["low", "medium", "high"]
    risk_reasons: list[str] = field(default_factory=list)
    must_escalate: bool = False
    escalate_reason: Optional[str] = None
    is_in_scope: bool = True
    is_malicious: bool = False  # prompt-injection / abusive content
    multi_request: bool = False


@dataclass
class IssueOutput:
    status: Status
    product_area: str
    response: str
    justification: str
    request_type: RequestType
