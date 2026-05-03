"""
Corpus loading + retrieval.

Loads the support corpus from ``data/{hackerrank,claude,visa}/`` and exposes a
``CorpusIndex`` that can return the most relevant chunks for a query. We use
TF-IDF with sklearn — it's deterministic, requires no API key, and works well
for support documentation where the user's wording usually overlaps with the
docs themselves.

The corpus directory layout is inferred at load time. The loader handles the
common formats we expect a help-center scrape to land in: HTML, Markdown,
plain text, and JSON files containing ``{title, body}`` records.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from schema import RetrievedChunk

logger = logging.getLogger("triage.corpus")

# The three companies the agent supports. Folder names are matched
# case-insensitively against directory names under data/.
KNOWN_COMPANIES = ("hackerrank", "claude", "visa")

# Cap chunk size so the LLM context stays bounded. Help-center articles are
# usually short, but a few are long FAQs we want to split.
TARGET_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


@dataclass
class Chunk:
    company: str  # canonical: "HackerRank" | "Claude" | "Visa"
    title: str
    source_path: str
    text: str


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _strip_html(s: str) -> str:
    """Cheap HTML→text. We don't need a full parser; the corpus is help-center
    pages, not arbitrary web content. ``<script>`` and ``<style>`` blocks are
    nuked first so their contents don't leak into the chunk text."""
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = _HTML_TAG_RE.sub(" ", s)
    # Decode the handful of HTML entities that actually show up in support docs.
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
           .replace("&lt;", "<").replace("&gt;", ">")
           .replace("&quot;", '"').replace("&#39;", "'"))
    return _WS_RE.sub(" ", s).strip()


def _strip_frontmatter(s: str) -> tuple[str, str]:
    """Strip YAML/TOML frontmatter delimited by --- or +++ lines.
    Returns (title_from_frontmatter_or_empty, body_without_frontmatter)."""
    title = ""
    s = s.lstrip()
    if not s.startswith("---"):
        return title, s
    end = s.find("\n---", 3)
    if end == -1:
        return title, s
    front = s[3:end]
    # Try to extract a title field from the frontmatter.
    m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', front, flags=re.MULTILINE)
    if m:
        title = m.group(1).strip()
    return title, s[end + 4:].lstrip()


def _strip_markdown(s: str) -> str:
    _, s = _strip_frontmatter(s)   # remove YAML frontmatter before anything else
    s = _MD_LINK_RE.sub(r"\1", s)  # keep link text, drop URL
    s = re.sub(r"`{1,3}[^`]*`{1,3}", " ", s)  # drop code spans/blocks
    s = re.sub(r"^#+\s*", "", s, flags=re.MULTILINE)  # heading marks
    s = re.sub(r"^[-*+]\s+", "", s, flags=re.MULTILINE)  # list bullets
    return _WS_RE.sub(" ", s).strip()


def _title_from_filename(p: Path) -> str:
    name = p.stem.replace("_", " ").replace("-", " ").strip()
    return name or p.name


def _load_one_file(path: Path) -> list[tuple[str, str]]:
    """Return a list of ``(title, text)`` pairs for one file. Most files give
    a single record; JSON files may give many."""
    suffix = path.suffix.lower()
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return []

    if suffix in {".html", ".htm"}:
        text = _strip_html(raw)
        # Pull a <title> tag if present, otherwise fall back to the filename.
        m = re.search(r"<title>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
        title = _strip_html(m.group(1)) if m else _title_from_filename(path)
        return [(title, text)] if text else []

    if suffix in {".md", ".markdown"}:
        fm_title, body = _strip_frontmatter(raw)
        text = _strip_markdown(body)
        m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        title = (m.group(1).strip() if m else fm_title) or _title_from_filename(path)
        return [(title, text)] if text else []

    if suffix == ".txt":
        text = _WS_RE.sub(" ", raw).strip()
        return [(_title_from_filename(path), text)] if text else []

    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Bad JSON in %s — skipping", path)
            return []
        records: list[tuple[str, str]] = []
        # Be liberal in what we accept; help-center exports vary.
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or item.get("name") or item.get("question")
                     or _title_from_filename(path))
            body = (item.get("body") or item.get("content") or item.get("text")
                    or item.get("answer") or "")
            if isinstance(body, list):
                body = " ".join(str(x) for x in body)
            body = _strip_html(str(body)) if "<" in str(body) else str(body)
            body = _WS_RE.sub(" ", body).strip()
            if body:
                records.append((str(title).strip() or _title_from_filename(path), body))
        return records

    # Quietly ignore unknown extensions (images, PDFs, .DS_Store, etc.).
    return []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Split long articles on sentence boundaries with a soft target size.
    Most help-center articles fit in one chunk, so the common path is fast."""
    if len(text) <= TARGET_CHUNK_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 <= TARGET_CHUNK_CHARS:
            buf = f"{buf} {sent}".strip()
        else:
            if buf:
                chunks.append(buf)
            # Carry a small tail forward for context continuity.
            tail = buf[-CHUNK_OVERLAP_CHARS:] if len(buf) > CHUNK_OVERLAP_CHARS else ""
            buf = (tail + " " + sent).strip()
    if buf:
        chunks.append(buf)
    return chunks


def _canonical_company(folder_name: str) -> Optional[str]:
    """Map a folder name to the canonical company string used in CSV output."""
    n = folder_name.lower()
    if "hackerrank" in n:
        return "HackerRank"
    if "claude" in n or "anthropic" in n:
        return "Claude"
    if "visa" in n:
        return "Visa"
    return None


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class CorpusIndex:
    """TF-IDF index built per-company so we can scope retrieval to one domain
    once we know which company a ticket belongs to."""

    def __init__(self) -> None:
        self.per_company: dict[str, "_PerCompanyIndex"] = {}

    @classmethod
    def build(cls, root: Path) -> "CorpusIndex":
        idx = cls()
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            company = _canonical_company(child.name)
            if company is None:
                logger.debug("Skipping unrecognized corpus folder: %s", child)
                continue
            chunks = list(idx._load_company(child, company))
            if not chunks:
                logger.warning("No chunks loaded for %s from %s", company, child)
                continue
            idx.per_company[company] = _PerCompanyIndex.fit(chunks)
            logger.info("Loaded %d chunks for %s", len(chunks), company)
        return idx

    @staticmethod
    def _load_company(folder: Path, company: str) -> Iterable[Chunk]:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            for title, text in _load_one_file(path):
                for piece in _chunk_text(text):
                    if len(piece) < 40:  # discard near-empty fragments
                        continue
                    yield Chunk(
                        company=company,
                        title=title,
                        source_path=str(path.relative_to(folder.parent)),
                        text=piece,
                    )

    def total_chunks(self) -> int:
        return sum(idx.size for idx in self.per_company.values())

    def search(self, query: str, company: Optional[str], k: int = 5) -> list[RetrievedChunk]:
        """Top-k chunks for a query. If ``company`` is given, restrict to that
        company's index; otherwise search all of them and merge by score."""
        if not query.strip():
            return []
        results: list[RetrievedChunk] = []
        targets = [company] if company and company in self.per_company \
            else list(self.per_company.keys())
        for c in targets:
            results.extend(self.per_company[c].search(query, k=k))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


class _PerCompanyIndex:
    """Wraps a single TF-IDF matrix for one company. We import sklearn lazily
    so the rule-based fallback works on minimal environments too."""

    def __init__(self, chunks: list[Chunk], vectorizer, matrix) -> None:
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

    @property
    def size(self) -> int:
        return len(self.chunks)

    @classmethod
    def fit(cls, chunks: list[Chunk]) -> "_PerCompanyIndex":
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Index title + body together. Repeating the title gives it a small
        # weight boost — titles are usually a strong topic signal.
        docs = [f"{c.title} {c.title} {c.text}" for c in chunks]
        # max_df is brittle when the corpus is tiny — disable it for very
        # small indices (mostly relevant in tests with 1–2 docs per company).
        max_df = 1.0 if len(docs) < 5 else 0.95
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=max_df,
            sublinear_tf=True,
        )
        matrix = vec.fit_transform(docs)
        return cls(chunks, vec, matrix)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        # ``argpartition`` is O(n) — argsort is fine here too since corpora
        # are small (~hundreds to a few thousand chunks).
        top = sims.argsort()[::-1][:k]
        out: list[RetrievedChunk] = []
        for i in top:
            score = float(sims[i])
            if score <= 0.0:
                continue
            ch = self.chunks[i]
            out.append(RetrievedChunk(
                company=ch.company, title=ch.title,
                source_path=ch.source_path, text=ch.text, score=score,
            ))
        return out
