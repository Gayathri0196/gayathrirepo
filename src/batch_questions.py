"""
batch_questions.py -- Generic Batch Question Answering
-----------------------------------------------------
Extracts natural-language questions from output documents and answers each question using
retrieval + reranking + evidence-based generation.

This module is intentionally format-agnostic: it does not try to fill any
specific output template structure.
"""

import os
import re
import logging
import shutil
from typing import List, Tuple, Optional

import spacy

from retriever import retrieve_with_scores
from qa_chain import get_qa_chain, rerank_documents, answer_with_context
from fallback import check_fallback, get_fallback_response

logger = logging.getLogger(__name__)

DEFAULT_QUESTION_PDF = os.path.join("..", "output_files", "lms_validation_plan_sample.pdf")
DEFAULT_OUT_TXT = os.path.join("..", "output_files", "batch_answers.txt")
DEFAULT_FETCHED_QUESTIONS_TXT = os.path.join("..", "output_files", "fetched_questions.txt")


def _read_file_text_with_docling(file_path: str) -> str:
    """
    Extract text from a document using Docling.

    Supports structured formats like PDF/DOCX/PPTX and returns markdown/text.
    """
    try:
        from docling.document_converter import (
            DocumentConverter,
            InputFormat,
            PdfFormatOption,
        )
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except Exception as e:
        raise RuntimeError(
            "Docling is not available. Install it with: pip install docling"
        ) from e

    # Disable OCR to avoid extra OCR model downloads; rely on backend text extraction.
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False
    pdf_options.force_backend_text = True

    # Reduce noisy warnings and patch HF symlink behavior for Windows user-mode sessions.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub import file_download as hf_file_download

        if not hasattr(hf_file_download, "_original_create_symlink"):
            hf_file_download._original_create_symlink = hf_file_download._create_symlink

            def _safe_create_symlink(src, dst, new_blob=False):
                try:
                    return hf_file_download._original_create_symlink(src, dst, new_blob=new_blob)
                except OSError:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

            hf_file_download._create_symlink = _safe_create_symlink
    except Exception:
        pass

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )
    result = converter.convert(file_path)
    doc = getattr(result, "document", None)
    if doc is None:
        return ""

    for method_name in ("export_to_markdown", "export_to_text", "to_markdown", "to_text"):
        method = getattr(doc, method_name, None)
        if callable(method):
            try:
                out = method()
                if out:
                    return str(out)
            except Exception:
                continue

    # Last resort fallback.
    return str(doc)


# Lazy-loaded spaCy model (loaded once on first use)
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])
        except OSError:
            # Fallback: blank English model with sentencizer if model not installed
            _NLP = spacy.blank("en")
            _NLP.add_pipe("sentencizer")
    return _NLP


# Regex to strip Q1:, Q2:, Question 1:, 1. style prefixes
_Q_PREFIX = re.compile(
    r"^\s*(?:q(?:uestion)?\s*\d*\s*[:.)]|\d{1,2}[.)]\s)",
    re.IGNORECASE,
)
# Regex to detect answer lines (skip them)
_A_PREFIX = re.compile(r"^\s*a(?:nswer)?\s*\d*\s*[:.)]\s*", re.IGNORECASE)


def _extract_questions_from_text(raw_text: str, source_name: str) -> List[str]:
    """Extract question sentences from raw text using line heuristics + spaCy segmentation."""
    nlp = _get_nlp()

    raw_lines = [ln.rstrip() for ln in (raw_text or "").splitlines()]

    # Strip Q/A prefixes per line; skip answer lines entirely.
    candidate_lines: List[str] = []
    line_questions: List[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            candidate_lines.append("")
            continue
        if _A_PREFIX.match(stripped):
            continue

        # Remove common bullets/check marks before processing.
        stripped = re.sub(r"^[\-\*•☐☒]+\s*", "", stripped)

        # Skip option-only checklist lines that are not questions.
        if re.match(r"^(?:[A-Za-z].{0,120})$", stripped) and "?" not in stripped and (
            "testing" in stripped.lower() or "uat" in stripped.lower() or "system" in stripped.lower()
        ):
            continue

        # Direct line-level extraction for explicit question labels.
        has_question_label = bool(re.match(r"^\s*q(?:uestion)?\s*\d*\s*[:.)]", stripped, re.IGNORECASE))
        has_numeric_label = bool(re.match(r"^\s*\d{1,2}[.)]\s+", stripped))

        # Start a fresh paragraph at each labeled question line.
        if (has_question_label or has_numeric_label) and candidate_lines and candidate_lines[-1] != "":
            candidate_lines.append("")

        if (has_question_label or has_numeric_label) and "?" in stripped:
            parts = [p.strip() + "?" for p in stripped.split("?") if p.strip()]
            for p in parts:
                q = _Q_PREFIX.sub("", p).strip()
                if len(q) >= 6 and not _A_PREFIX.match(q):
                    line_questions.append(q)

        cleaned = _Q_PREFIX.sub("", stripped).strip()
        candidate_lines.append(cleaned or stripped)

    # Group consecutive non-blank lines into paragraphs.
    paragraphs: List[str] = []
    current: List[str] = []
    for line in candidate_lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))

    questions: List[str] = list(line_questions)
    for para in paragraphs:
        if not para.strip():
            continue
        doc = nlp(para)
        for sent in doc.sents:
            text = sent.text.strip()
            if not text or "?" not in text:
                continue
            # Some parser outputs can contain multiple '?' in one sentence; split them.
            parts = [p.strip() + "?" for p in text.split("?") if p.strip()]
            for p in parts:
                q = _Q_PREFIX.sub("", p).strip()
                q = re.sub(r"\s+", " ", q)
                # Drop obvious checklist-only/noise fragments.
                if re.fullmatch(r"[☐☒\sA-Za-z0-9()/\-]+", q) and "?" not in q:
                    continue
                if len(q) >= 6 and not _A_PREFIX.match(q):
                    questions.append(q)

    # De-duplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for q in questions:
        key = q.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(q.strip())

    logger.info(
        "Extracted %d questions from '%s' using spaCy sentence segmentation",
        len(deduped),
        source_name,
    )
    return deduped


def extract_questions_from_pdf(pdf_path: str) -> List[str]:
    """Extract questions from a PDF using Docling + spaCy."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Question source PDF not found: {pdf_path}")

    raw_text = _read_file_text_with_docling(pdf_path)
    return _extract_questions_from_text(raw_text, os.path.basename(pdf_path))


def _answer_questions(questions: List[str], out_txt_path: str, limit: Optional[int] = None) -> Tuple[str, int]:
    if limit is not None and limit > 0:
        questions = questions[:limit]

    qa_chain = get_qa_chain(enable_reranking=True)
    meta = getattr(qa_chain, "metadata", {})
    llm = meta.get("llm")

    lines_out: List[str] = []
    for i, q in enumerate(questions, 1):
        results = retrieve_with_scores(q)
        docs = [doc for doc, _ in results]

        if check_fallback(docs):
            answer = get_fallback_response()["answer"]
            sources = []
        else:
            if meta.get("enable_reranking") and results:
                reranked = rerank_documents(llm, q, results)
                if reranked:
                    docs = [r[0] for r in reranked[:8]]

            answer = answer_with_context(llm, q, docs)
            sources = []
            for d in docs[:5]:
                m = d.metadata
                sources.append(
                    f"[{m.get('section_id', '?')}] {m.get('section_title', '?')}"
                    f" (pages {m.get('page_start', '?')}-{m.get('page_end', '?')})"
                )

        lines_out.append(f"Q{i}: {q}")
        lines_out.append(f"A{i}: {answer}")
        if sources:
            lines_out.append("Sources:")
            for s in sources:
                lines_out.append(f"- {s}")
        lines_out.append("")

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out).strip() + "\n")

    logger.info("Answered %d questions", len(questions))
    logger.info("Saved batch answers to %s", out_txt_path)
    return out_txt_path, len(questions)


def _save_fetched_questions(questions: List[str], out_path: str) -> str:
    """Persist fetched questions to disk for audit/debug visibility."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [f"Q{i}: {q}" for i, q in enumerate(questions, 1)]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    logger.info("Saved fetched questions to %s", out_path)
    return out_path


def answer_questions_from_pdf(
    pdf_path: str = DEFAULT_QUESTION_PDF,
    out_txt_path: str = DEFAULT_OUT_TXT,
    fetched_questions_path: str = DEFAULT_FETCHED_QUESTIONS_TXT,
    limit: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Extract all questions from the provided PDF and answer them using the
    retrieval pipeline.

    Returns:
        (output_text_file_path, answered_count)
    """
    questions = extract_questions_from_pdf(pdf_path)
    if limit is not None and limit > 0:
        questions = questions[:limit]

    _save_fetched_questions(questions, fetched_questions_path)
    out_path, count = _answer_questions(questions, out_txt_path, limit=None)
    logger.info("Answered %d questions from %s", count, os.path.basename(pdf_path))
    return out_path, count
