"""Run the public-set ablation sequence required by Homework 4.

Run from the repository root:
    python3 ablation.py --data-folder public_test

The script calls the provided end-to-end runner once per configuration. It expects
DEEPSEEK_API_KEY in the environment or in a local .env file.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import re
import sys
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


def _plain_table(table_html: str) -> str:
    return BeautifulSoup(table_html, "lxml").get_text(" ", strip=True)


def _legacy_chunk(file_name: str, html: str, convert_tables: bool = False) -> list[Document]:
    company, fiscal_year, doc_type, soup = hw4._document_metadata(html)
    if convert_tables:
        for table in list(soup.find_all("table")):
            table.replace_with(soup.new_string(hw4.table_to_text(str(table))))
    text = soup.get_text(" ", strip=True)
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.create_documents([text])
    metadata = {
        "source": file_name,
        "company": company,
        "fiscal_year": fiscal_year,
        "doc_type": doc_type,
    }
    for chunk in chunks:
        chunk.metadata = dict(metadata)
    return chunks


def _no_query(question: str) -> dict:
    return {}


def _build_index(chunks: list[Document], custom_tokens: bool = False) -> dict[str, Any]:
    documents = list(chunks)
    if custom_tokens:
        corpus = [hw4._tokens(chunk.page_content) for chunk in documents]
    else:
        corpus = [chunk.page_content.lower().split() for chunk in documents]
    return {
        "chunks": documents,
        "tokens": corpus,
        "bm25": BM25Okapi(corpus) if corpus and any(corpus) else None,
    }


def _eligible_sources(
    index: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    for chunk in index["chunks"]:
        meta = chunk.metadata or {}
        source = str(meta.get("source", ""))
        item = sources.setdefault(source, {
            "company": hw4.company_key(meta.get("company", "")),
            "year": hw4.as_year(meta.get("fiscal_year")),
            "doc_type": meta.get("doc_type"),
            "text": "",
        })
        item["text"] += "\n" + chunk.page_content

    result: dict[str, list[str]] = {}
    companies = parsed.get("companies") or []
    keys = [hw4.company_key(name) for name in companies]
    keys = list(dict.fromkeys(key for key in keys if key))
    if not keys:
        return result
    for key in keys:
        docs = [
            (source, item) for source, item in sources.items()
            if item["company"] == key and item["doc_type"] == parsed.get("doc_type")
        ]
        year = hw4.as_year(parsed.get("fiscal_year"))
        doc_type = parsed.get("doc_type")
        if year is None:
            if docs:
                newest = max(item["year"] or 0 for _, item in docs)
                docs = [(source, item) for source, item in docs if (item["year"] or 0) == newest]
        elif doc_type == "annual_report":
            exact = [(source, item) for source, item in docs if item["year"] == year]
            docs = exact or [(source, item) for source, item in docs if item["year"] == year + 1]
        elif doc_type == "earnings_call":
            docs = [(source, item) for source, item in docs if item["year"] in {year, year - 1}]
        else:
            docs = [
                (source, item) for source, item in docs
                if re.search(rf"(?<!\d){year}[a-z]?(?!\d)", item["text"], re.IGNORECASE)
            ]
        result[key] = [source for source, _ in docs]
    return result


def _retrieve(
    index: dict[str, Any],
    question: str,
    *,
    filter_metadata: bool,
    custom_tokens: bool,
    query_profile: bool = False,
    per_company: bool,
) -> list[Document]:
    if not index.get("chunks") or not index.get("bm25"):
        return []
    parsed = hw4.parse_query(question)
    if query_profile:
        query, _ = hw4._query_profile(question, parsed.get("companies") or [])
    elif custom_tokens:
        query = hw4._tokens(question)
    else:
        query = question.lower().split()
    scores = index["bm25"].get_scores(query)

    candidates = list(range(len(index["chunks"])))
    if filter_metadata:
        source_map = _eligible_sources(index, parsed)
        allowed = {source for values in source_map.values() for source in values}
        if not allowed or any(not values for values in source_map.values()):
            return []
        candidates = [
            idx for idx in candidates
            if index["chunks"][idx].metadata.get("source") in allowed
        ]
    candidates.sort(key=lambda idx: scores[idx], reverse=True)

    if per_company and len(parsed.get("companies") or []) > 1:
        source_map = _eligible_sources(index, parsed)
        by_company = {}
        for company in parsed["companies"]:
            allowed = set(source_map.get(hw4.company_key(company), []))
            by_company[hw4.company_key(company)] = [
                idx for idx in candidates
                if index["chunks"][idx].metadata.get("source") in allowed
            ]
        result = []
        depth = 0
        while len(result) < hw4.MAX_RETRIEVED:
            added = False
            for company in parsed["companies"]:
                docs = by_company.get(hw4.company_key(company), [])
                if depth < len(docs):
                    result.append(index["chunks"][docs[depth]])
                    added = True
                    if len(result) >= hw4.MAX_RETRIEVED:
                        break
            if not added:
                break
            depth += 1
        return result

    return [index["chunks"][idx] for idx in candidates[:hw4.MAX_RETRIEVED]]


def _metric(log: str, pattern: str) -> str:
    match = re.search(pattern, log)
    return f"{float(match.group(1)) / 100:.3f}" if match else ""


def _extract_metrics(log: str, name: str) -> dict[str, str]:
    task3 = re.search(
        r"answerable questions with chunks:\s*([\d.]+)%\s*\|\s*"
        r"unanswerable questions with \[\]:\s*([\d.]+)%",
        log,
    )
    return {
        "variant": name,
        "task_1_1": _metric(log, r"Task 1\.1 table_to_text: ([\d.]+)%"),
        "task_1_2": _metric(log, r"Task 1\.2 chunk_document: ([\d.]+)%"),
        "task_2_1": _metric(log, r"Task 2\.1 valid, non-empty retrieval on answerable questions: ([\d.]+)%"),
        "task_2_2": _metric(log, r"Task 2\.2 parse_query fields right: ([\d.]+)%"),
        "task_2_3": _metric(log, r"Task 2\.3 retrieval recall@\d+ on answerable questions: ([\d.]+)%"),
        "task_3_kept": f"{float(task3.group(1)) / 100:.3f}" if task3 else "",
        "task_3_rejected": f"{float(task3.group(2)) / 100:.3f}" if task3 else "",
        "task_4": _metric(log, r"Task 4\s+answer accuracy: ([\d.]+)%"),
    }


def _run_variant(
    name: str,
    *,
    data_folder: str,
    output_path: Path,
    table_fn: Callable[[str], str],
    chunk_fn: Callable[[str, str], list[Document]],
    parse_fn: Callable[[str], dict],
    index_fn: Callable[[list[Document]], dict[str, Any]],
    retrieve_fn: Callable[[dict[str, Any], str], list[Document]],
) -> tuple[dict[str, str], str]:
    hw4.table_to_text = table_fn
    hw4.chunk_document = chunk_fn
    hw4.parse_query = parse_fn
    hw4.build_index = index_fn
    hw4.retrieve = retrieve_fn
    hw4._COMPANY_REGISTRY.clear()
    hw4.USAGE = hw4.UsageTracker()
    hw4.GENERATOR_USAGE = hw4.UsageTracker()
    hw4._generator = None

    old_argv = sys.argv
    sys.argv = [
        "hw4.py",
        "--data-folder", data_folder,
        "--output", str(output_path),
    ]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            hw4.main()
    finally:
        sys.argv = old_argv
    log = buffer.getvalue()
    print(f"Finished {name}; results: {output_path}")
    return _extract_metrics(log, name), log


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 4 public-set ablation study")
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output-dir", default="ablation_results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_parse = _no_query
    baseline_index = lambda chunks: _build_index(chunks, custom_tokens=False)
    baseline_retrieve = lambda index, question: _retrieve(
        index, question, filter_metadata=False, custom_tokens=False, per_company=False
    )
    filtered_retrieve = lambda index, question: _retrieve(
        index, question, filter_metadata=True, custom_tokens=False, per_company=False
    )
    tokenized_index = hw4.build_index
    tokenized_retrieve = lambda index, question: _retrieve(
        index, question, filter_metadata=True, custom_tokens=True,
        query_profile=True, per_company=False,
    )
    comparison_retrieve = lambda index, question: _retrieve(
        index, question, filter_metadata=True, custom_tokens=True,
        query_profile=True, per_company=True,
    )

    variants = [
        {
            "name": "B0",
            "description": "Naive baseline",
            "table": _plain_table,
            "chunk": lambda f, h: _legacy_chunk(f, h, convert_tables=False),
            "parse": baseline_parse,
            "index": baseline_index,
            "retrieve": baseline_retrieve,
        },
        {
            "name": "V1",
            "description": "Table to text",
            "table": hw4.table_to_text,
            "chunk": lambda f, h: _legacy_chunk(f, h, convert_tables=True),
            "parse": baseline_parse,
            "index": baseline_index,
            "retrieve": baseline_retrieve,
        },
        {
            "name": "V2",
            "description": "Section-aware chunks and metadata",
            "table": hw4.table_to_text,
            "chunk": hw4.chunk_document,
            "parse": baseline_parse,
            "index": baseline_index,
            "retrieve": baseline_retrieve,
        },
        {
            "name": "V3",
            "description": "Question parsing and metadata filters",
            "table": hw4.table_to_text,
            "chunk": hw4.chunk_document,
            "parse": hw4.parse_query,
            "index": baseline_index,
            "retrieve": filtered_retrieve,
        },
        {
            "name": "V4",
            "description": "Normalized BM25 tokens and query expansion",
            "table": hw4.table_to_text,
            "chunk": hw4.chunk_document,
            "parse": hw4.parse_query,
            "index": tokenized_index,
            "retrieve": tokenized_retrieve,
        },
        {
            "name": "V5",
            "description": "Per-company comparison retrieval",
            "table": hw4.table_to_text,
            "chunk": hw4.chunk_document,
            "parse": hw4.parse_query,
            "index": tokenized_index,
            "retrieve": comparison_retrieve,
        },
        {
            "name": "V6",
            "description": "Answerability and evidence threshold",
            "table": hw4.table_to_text,
            "chunk": hw4.chunk_document,
            "parse": hw4.parse_query,
            "index": hw4.build_index,
            "retrieve": hw4.retrieve,
        },
    ]

    rows = []
    for variant in variants:
        output_path = (
            Path("results.csv") if variant["name"] == "V6"
            else output_dir / f"{variant['name']}.csv"
        )
        metrics, log = _run_variant(
            variant["name"],
            data_folder=args.data_folder,
            output_path=output_path,
            table_fn=variant["table"],
            chunk_fn=variant["chunk"],
            parse_fn=variant["parse"],
            index_fn=variant["index"],
            retrieve_fn=variant["retrieve"],
        )
        metrics["description"] = variant["description"]
        rows.append(metrics)
        log_path = output_dir / f"{variant['name']}.log"
        log_path.write_text(log, encoding="utf-8")
        for line in log.splitlines():
            if line.startswith((
                "Task 1.", "Task 2.", "Task 3", "Task 4", "Estimated points:",
                "Wrote ", "Your usage:", "Generator usage:", "Time:",
            )):
                print(line)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
