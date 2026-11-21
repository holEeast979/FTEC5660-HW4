"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from rank_bm25 import BM25Okapi

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

_COMPANY_REGISTRY: dict[str, str] = {}

_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*", re.IGNORECASE)
_QUERY_STOP_WORDS = {
    "a", "about", "after", "all", "an", "and", "annual", "any", "are", "as",
    "at", "be", "because", "before", "between", "by", "call", "can", "did",
    "do", "does", "during", "end", "ended", "for", "fourth", "from", "full",
    "had", "has", "have", "held", "how", "in", "into", "is", "it", "its",
    "more", "much", "of", "on", "or", "our", "report", "results", "review",
    "set", "than", "that", "the", "their", "this", "to", "up", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "year",
    "q", "fy", "usd", "million", "billion", "percent", "per", "share",
    "earning", "earnings", "management", "give", "gives", "gave", "month",
    "months",
    "december", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november",
    "generate", "generated", "earn", "earned", "announce", "announced",
    "large", "firm", "rate", "rates", "number", "value",
}
_NON_ENTITY_FIRST_WORDS = {
    "a", "an", "annual", "accounts", "asset", "assets", "at", "cash", "chief",
    "consolidated", "december", "director", "directors", "during", "employee",
    "employees", "executive", "expenses", "financial", "fourth", "full",
    "gross", "how", "income", "in", "independent", "investment", "liabilities",
    "management", "africa", "america", "americas", "asia", "europe", "middle",
    "north", "pacific", "south",
    "net", "on", "operating", "other", "q", "revenue", "risk", "statement",
    "total", "usd", "what", "which", "who", "year",
}
_BROKER_ACTIONS = (
    r"(?:give|gives|gave|set|sets|assign|assigns|forecast|forecasts|estimate|"
    r"estimates|rate|rates|recommend|recommends|reiterate|reiterates)"
)
_TOKEN_ALIASES = {
    "audit": {"audit", "auditor"},
    "auditor": {"audit", "auditor"},
    "borrow": {"borrow", "debt"},
    "cash": {"cash", "equivalent"},
    "ceo": {"ceo", "chief", "executive", "officer"},
    "cfo": {"cfo", "chief", "financial", "officer"},
    "city": {"city", "headquarter"},
    "employee": {"employee", "staff", "headcount"},
    "equivalent": {"cash", "equivalent"},
    "forecast": {"estimate", "forecast", "projection"},
    "guide": {"guide", "guidance", "outlook"},
    "headquarter": {"city", "headquarter"},
    "headcount": {"employee", "headcount", "staff"},
    "income": {"income", "profit"},
    "profit": {"income", "profit"},
    "programme": {"program", "programme"},
    "program": {"program", "programme"},
    "rating": {"rating", "recommendation"},
    "receivable": {"account", "receivable"},
    "account": {"account", "receivable"},
    "repurchase": {"buyback", "repurchase"},
    "revenue": {"revenue", "sales"},
    "sales": {"revenue", "sales"},
    "staff": {"employee", "headcount", "staff"},
}


def _plain_text(node: Any) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _tokens(text: Any) -> list[str]:
    """Lower-case word and number tokens with light English suffix normalization."""
    result = []
    for raw in _TOKEN_RE.findall(str(text or "").casefold()):
        token = raw.replace(",", "")
        if token.isdigit():
            result.append(token)
            continue
        if token in {"guidance", "guided"}:
            token = "guide"
        elif token in {"audited", "auditing"}:
            token = "audit"
        elif token == "headquartered":
            token = "headquarter"
        elif token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 6:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        result.append(token)
    return result


def _aliases(token: str) -> set[str]:
    return set(_TOKEN_ALIASES.get(token, {token}))


def _document_metadata(html: str) -> tuple[str, int, str, BeautifulSoup]:
    soup = BeautifulSoup(html, "lxml")
    title = _plain_text(soup.title)
    headings = [_plain_text(tag) for tag in soup.find_all(["h1", "h2", "h3"])]
    h1 = _plain_text(soup.find("h1"))
    headline = " ".join([title, *headings[:5]])
    headline_lower = headline.casefold()

    if re.search(r"earnings?\s+(?:conference\s+)?call|conference\s+call", headline_lower):
        doc_type = "earnings_call"
    elif re.search(r"equity\s+research|results\s+review|research\s+report", headline_lower):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    year_match = re.search(
        r"\b(?:fy\s*[-:]?\s*|annual report\s+|full year\s+)(20\d{2})\b",
        headline,
        re.IGNORECASE,
    )
    if not year_match:
        year_match = re.search(r"\b(20\d{2})\b", headline)
    fiscal_year = int(year_match.group(1)) if year_match else 0

    company = re.sub(r"\s*\([^)]*\)\s*$", "", h1).strip()
    if not company:
        parts = [part.strip() for part in title.split("|") if part.strip()]
        if doc_type == "research_report" and len(parts) > 1:
            company = re.sub(r"\s*\([^)]*\)\s*$", "", parts[1]).strip()
        elif parts:
            company = re.split(
                r"\b(?:annual report|earnings call transcript)\b",
                parts[0],
                flags=re.IGNORECASE,
            )[0].strip()
    if not company:
        prefix = re.split(
            r"\b(?:annual report|earnings|fy\d{4})\b",
            _plain_text(soup.body),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        company = prefix.strip(" |—-")

    return company, fiscal_year, doc_type, soup


def _chunk_windows(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    overlap = min(360, MAX_CHUNK_CHARS // 4)
    stride = MAX_CHUNK_CHARS - overlap
    chunks = []
    for start in range(0, len(text), stride):
        piece = text[start:start + MAX_CHUNK_CHARS].strip()
        if piece:
            chunks.append(piece)
        if start + MAX_CHUNK_CHARS >= len(text):
            break
    return chunks


def _company_mentions(question: str) -> list[str]:
    raw_words = re.findall(
        r"[a-z0-9]+", unicodedata.normalize("NFKC", question).casefold()
    )
    normalized_question = " " + " ".join(raw_words) + " "
    mentions: list[tuple[int, str, str]] = []

    for key, display in _COMPANY_REGISTRY.items():
        words = re.findall(r"[a-z0-9]+", key)
        if not words:
            continue
        pattern = (
            r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, words)) + r"(?![a-z0-9])"
        )
        for match in re.finditer(pattern, normalized_question):
            mentions.append((match.start(), key, display))

    broker_spans = []
    broker_pattern = re.compile(
        rf"\b(?:does|did|do)\s+(?P<name>.+?)\s+{_BROKER_ACTIONS}\b",
        re.IGNORECASE,
    )
    for match in broker_pattern.finditer(question):
        broker_spans.append((match.start("name"), match.end("name")))

    proper_name = re.compile(
        r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+(?:[A-Z][A-Za-z0-9&.-]*|and|of|the)){1,4}\b"
    )
    generic = _NON_ENTITY_FIRST_WORDS | _QUERY_STOP_WORDS
    for match in proper_name.finditer(question):
        candidate = match.group().strip(" .,-")
        first = re.findall(r"[A-Za-z]+", candidate)
        if not first or first[0].casefold() in generic:
            continue
        if re.search(
            r"\b(?:fy|q)\s*\d{1,4}\b|\b(?:19|20)\d{2}\b",
            candidate,
            re.IGNORECASE,
        ):
            continue
        if any(match.start() < end and match.end() > start for start, end in broker_spans):
            continue
        key = company_key(candidate)
        if not key or key in _COMPANY_REGISTRY:
            continue
        mentions.append((match.start(), key, candidate))

    mentions.sort(key=lambda item: item[0])
    companies = []
    seen = set()
    for _, key, display in mentions:
        if key not in seen:
            seen.add(key)
            companies.append(display)
    return companies


def _query_profile(question: str, companies: list[str]) -> tuple[list[str], list[set[str]]]:
    company_words = set()
    for company in companies:
        company_words.update(_tokens(company))

    core = []
    for token in _tokens(question):
        if token.isdigit() or token in _QUERY_STOP_WORDS or token in company_words:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        if token not in core:
            core.append(token)

    groups = [_aliases(token) for token in core]
    expanded = []
    for token in core:
        for variant in _aliases(token):
            if variant not in expanded:
                expanded.append(variant)
    return expanded, groups


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table") or soup
    caption = _plain_text(table.find("caption"))
    lines = [caption] if caption else []

    rows = table.find_all("tr")
    thead = table.find("thead")
    header_rows = thead.find_all("tr") if thead else []
    if not header_rows and rows:
        first_cells = rows[0].find_all(["th", "td"], recursive=False)
        if first_cells and all(cell.name == "th" for cell in first_cells):
            header_rows = [rows[0]]

    headers: dict[int, list[str]] = {}
    header_ids = {id(row) for row in header_rows}
    for row in header_rows:
        for column, cell in enumerate(row.find_all(["th", "td"], recursive=False)):
            value = _plain_text(cell)
            if column and value:
                headers.setdefault(column, []).append(value)

    for row in rows:
        if id(row) in header_ids or "section" in (row.get("class") or []):
            continue
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 2:
            continue
        row_label = _plain_text(cells[0])
        if not row_label:
            continue
        rendered = []
        for column, cell in enumerate(cells[1:], start=1):
            value = _plain_text(cell)
            if not value:
                continue
            column_header = " / ".join(headers.get(column, [])) or f"Column {column}"
            rendered.append(f"{column_header}: {value}")
        if not rendered:
            continue
        row_line = f"{row_label} | " + " | ".join(rendered)
        if len(row_line) <= MAX_LINE_CHARS:
            lines.append(row_line)
        else:
            for item in rendered:
                lines.append(f"{row_label} | {item}"[:MAX_LINE_CHARS])
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    company, fiscal_year, doc_type, soup = _document_metadata(html)
    key = company_key(company)
    if key:
        _COMPANY_REGISTRY.setdefault(key, company)

    blocks = []
    content_root = soup.body or soup
    for element in content_root.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        if element.find_parent(["nav", "footer"]) is not None:
            continue
        if element.name != "table" and element.find_parent("table") is not None:
            continue
        if element.name == "p" and element.find_parent("li") is not None:
            continue
        text = table_to_text(str(element)) if element.name == "table" else _plain_text(element)
        if text:
            blocks.append(text)

    if not blocks:
        fallback = _plain_text(content_root)
        if fallback:
            blocks.append(fallback)
    document_text = "\n\n".join(blocks)
    metadata = {
        "source": file_name,
        "company": company,
        "fiscal_year": fiscal_year,
        "doc_type": doc_type,
    }
    return [
        Document(page_content=piece, metadata=dict(metadata))
        for piece in _chunk_windows(document_text)
    ]


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    companies = _company_mentions(question)
    year_match = re.search(r"\b(?:fy\s*)?((?:19|20)\d{2})\b", question, re.IGNORECASE)
    fiscal_year = int(year_match.group(1)) if year_match else None

    text = question.casefold()
    if re.search(r"\bearnings?\s+(?:conference\s+)?call\b|\bconference\s+call\b", text):
        doc_type = "earnings_call"
    elif re.search(
        r"\b(?:target price|price target|rating|forecast|forecasting|estimate|"
        r"results review|research report|broker)\b",
        text,
    ):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    return {"companies": companies, "fiscal_year": fiscal_year, "doc_type": doc_type}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    documents = list(chunks or [])
    tokenized = []
    sources: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(documents):
        meta = chunk.metadata or {}
        text = " ".join([
            chunk.page_content,
            str(meta.get("company", "")),
            str(meta.get("fiscal_year", "")),
            str(meta.get("doc_type", "")),
        ])
        tokenized.append(_tokens(text))
        source = str(meta.get("source", ""))
        info = sources.setdefault(source, {
            "company": str(meta.get("company", "")),
            "company_key": company_key(meta.get("company", "")),
            "fiscal_year": as_year(meta.get("fiscal_year")),
            "doc_type": meta.get("doc_type"),
            "indices": [],
            "text_parts": [],
        })
        info["indices"].append(index)
        info["text_parts"].append(chunk.page_content)

    for info in sources.values():
        info["text"] = "\n".join(info.pop("text_parts"))
        info["tokens"] = _tokens(info["text"])
        info["token_set"] = set(info["tokens"])

    bm25 = BM25Okapi(tokenized) if tokenized and any(tokenized) else None
    return {
        "chunks": documents,
        "tokenized": tokenized,
        "bm25": bm25,
        "sources": sources,
        "company_keys": {
            info["company_key"] for info in sources.values() if info["company_key"]
        },
    }


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    if not isinstance(index, dict) or not index.get("chunks") or not index.get("sources"):
        return []
    parsed = parse_query(question)
    companies = parsed.get("companies") or []
    company_keys = []
    for company in companies:
        key = company_key(company)
        if key and key not in company_keys:
            company_keys.append(key)
    if not company_keys or any(key not in index["company_keys"] for key in company_keys):
        return []

    query_terms, core_groups = _query_profile(question, companies)
    if not query_terms:
        return []
    bm25 = index.get("bm25")
    scores = bm25.get_scores(query_terms) if bm25 else [0.0] * len(index["chunks"])

    selected_sources: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    year = as_year(parsed.get("fiscal_year"))
    doc_type = parsed.get("doc_type")
    for key in company_keys:
        company_sources = [
            (source, info) for source, info in index["sources"].items()
            if info["company_key"] == key and info["doc_type"] == doc_type
        ]
        if year is None:
            if company_sources:
                latest = max((info["fiscal_year"] or 0) for _, info in company_sources)
                company_sources = [
                    (source, info) for source, info in company_sources
                    if (info["fiscal_year"] or 0) == latest
                ]
        elif doc_type == "annual_report":
            exact = [
                (source, info) for source, info in company_sources
                if info["fiscal_year"] == year
            ]
            company_sources = exact or [
                (source, info) for source, info in company_sources
                if info["fiscal_year"] == year + 1
            ]
        elif doc_type == "earnings_call":
            company_sources = [
                (source, info) for source, info in company_sources
                if info["fiscal_year"] in {year, year - 1}
            ]
        else:
            year_text = re.compile(rf"(?<!\d){year}[a-z]?(?!\d)", re.IGNORECASE)
            company_sources = [
                (source, info) for source, info in company_sources
                if year_text.search(info["text"])
            ]
        if not company_sources:
            return []
        selected_sources[key] = company_sources

    # Require the requested concepts to appear in the selected company's documents.
    for key in company_keys:
        source_infos = selected_sources[key]
        available_tokens = set().union(*(info["token_set"] for _, info in source_infos))
        if core_groups:
            coverage = sum(bool(group & available_tokens) for group in core_groups) / len(core_groups)
            if coverage < 0.65:
                return []
            if len(core_groups) >= 3 and coverage < 1.0:
                has_near_pair = False
                for _, info in source_infos:
                    token_positions: dict[str, list[int]] = {}
                    for position, token in enumerate(info["tokens"]):
                        token_positions.setdefault(token, []).append(position)
                    for left, right in zip(core_groups, core_groups[1:]):
                        left_positions = [
                            position for token in left
                            for position in token_positions.get(token, [])
                        ]
                        right_positions = [
                            position for token in right
                            for position in token_positions.get(token, [])
                        ]
                        if any(abs(a - b) <= 5 for a in left_positions for b in right_positions):
                            has_near_pair = True
                            break
                    if has_near_pair:
                        break
                if not has_near_pair:
                    return []

    ranked_by_company: dict[str, list[int]] = {}
    for key in company_keys:
        candidate_indices = [
            idx for _, info in selected_sources[key] for idx in info["indices"]
        ]
        candidate_indices.sort(key=lambda idx: scores[idx], reverse=True)
        candidate_indices = [idx for idx in candidate_indices if scores[idx] > 0]
        if not candidate_indices:
            return []
        ranked_by_company[key] = candidate_indices

    # Interleave companies so comparisons give the generator evidence for each side.
    results = []
    seen = set()
    depth = 0
    while len(results) < MAX_RETRIEVED:
        added = False
        for key in company_keys:
            candidates = ranked_by_company.get(key, [])
            if depth < len(candidates):
                idx = candidates[depth]
                signature = (
                    index["chunks"][idx].metadata.get("source"),
                    index["chunks"][idx].page_content,
                )
                if signature not in seen:
                    seen.add(signature)
                    results.append(index["chunks"][idx])
                    added = True
                    if len(results) >= MAX_RETRIEVED:
                        break
        if not added:
            break
        depth += 1
    return results


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
