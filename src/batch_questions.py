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
import html
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import spacy

from retriever import retrieve_with_scores
from qa_chain import get_qa_chain, rerank_documents, answer_with_context
from fallback import check_fallback, get_fallback_response
from image_extraction import get_image_extractor
from image_processing import get_image_processor, get_image_cache

logger = logging.getLogger(__name__)

DEFAULT_QUESTION_PDF = os.path.join("..", "output_files", "lms_validation_plan_sample.pdf")
DEFAULT_OUT_TXT = os.path.join("..", "output_files", "batch_answers.txt")
DEFAULT_FETCHED_QUESTIONS_TXT = os.path.join("..", "output_files", "fetched_questions.txt")


DOMAIN_SENTENCE = "sentence_based"
DOMAIN_CHECKLIST = "checklist"
DOMAIN_YES_NO = "yes_no"

_YES_NO_OPTION_MAP = {
    "yes": "Yes",
    "no": "No",
    "na": "NA",
    "n/a": "NA",
    "not applicable": "NA",
}
_CHECKED_BOX_GLYPHS = {"☒", "☑"}
_UNCHECKED_BOX_GLYPHS = {"☐"}
_MARK_CHARS = {"x", "X", "✓", "✔", "✗", "✘"}


def _ocr_fallback_enabled() -> bool:
    """Enable OCR fallback only when explicitly requested via environment."""
    return os.getenv("ENABLE_OCR_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class QuestionBlock:
    question_id: str
    question_text: str
    source_section: str
    answer_lines: List[str] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    note_label: str = ""          # Original label text, e.g. "Notes:" or "Additional Notes:"
    note_text: str = ""           # Any pre-filled note content captured from the PDF


@dataclass
class ExtractionBundle:
    by_domain: Dict[str, List[Dict[str, Any]]]
    question_texts: List[str]
    gate_hierarchy: List[Dict[str, Any]] = field(default_factory=list)
    extracted_images: List[Dict[str, Any]] = field(default_factory=list)
    image_analyses: List[Dict[str, Any]] = field(default_factory=list)


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9/ ]+", " ", (text or "").lower())).strip()


def _overlap_score(left: str, right: str) -> float:
    left_tokens = set(_normalize_match_text(left).split())
    right_tokens = set(_normalize_match_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def _canonical_option_value(option_text: str) -> Optional[str]:
    normalized = _normalize_match_text(_clean_option_text(option_text))
    if normalized in _YES_NO_OPTION_MAP:
        return _YES_NO_OPTION_MAP[normalized]

    match = re.match(r"^(yes|no|n/?a|not applicable)\b", normalized)
    if match:
        return _YES_NO_OPTION_MAP.get(match.group(1))
    return None


def _clean_option_text(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]?\s*(?:\[[ xX]\]|☐|☒)\s*)+", "", html.unescape(text or "")).strip()
    cleaned = re.sub(r"^[\-\*•]+\s*", "", cleaned)
    return re.sub(r"\s+", " ", cleaned)


def _extraction_confidence(flags: List[str]) -> str:
    return "low" if flags else "high"


def _read_file_text_with_docling(file_path: str, do_ocr: bool = False) -> str:
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

    # Default to backend text extraction; OCR can be enabled as a fallback when needed.
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = do_ocr
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
_CHECKBOX_PREFIX = re.compile(r"^\s*[-*]?\s*(?:\[[ xX]\]|☐|☒)\s*")
_QUESTION_START = re.compile(
    r"^(who|what|when|where|why|how|which|will|is|are|do|does|did|can|could|should|would|has|have|had)\b",
    re.IGNORECASE,
)
_BLANK_FILLER = re.compile(r"^(?:(?:_+|\\_+)\s*){4,}$")
_NOTE_LABEL = re.compile(r"^(?:additional\s+notes|notes?)\s*:?$", re.IGNORECASE)


def _looks_like_question_clause(text: str) -> bool:
    """Return True when a clause is likely an actual question prompt."""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) < 6:
        return False
    if _QUESTION_START.match(t):
        return True
    # Keep explicit question labels that survive OCR cleanup.
    return bool(re.match(r"^q(?:uestion)?\s*\d*\b", t, re.IGNORECASE))


def _normalize_question_text(text: str) -> str:
    """Normalize question prompt formatting and ensure trailing '?' for interrogatives."""
    t = html.unescape(re.sub(r"\s+", " ", text)).strip()
    if not t:
        return t
    if _looks_like_question_clause(t) and not t.endswith("?"):
        return f"{t}?"
    return t


def _is_meaningful_checkbox_prompt(text: str) -> bool:
    """
    Keep checklist options as-is (including short Yes/No options) to preserve
    template fidelity.
    """
    if not text:
        return False

    t = re.sub(r"\s+", " ", text).strip(" -")
    return bool(t)


def _extract_question_blocks_from_text(raw_text: str, source_name: str) -> List[QuestionBlock]:
    """Extract question blocks from Docling text while preserving sections and options."""
    raw_lines = [html.unescape(ln.rstrip()) for ln in (raw_text or "").splitlines()]

    blocks: List[QuestionBlock] = []
    current_block: Optional[QuestionBlock] = None
    current_section = "Unspecified Section"

    def flush_current() -> None:
        nonlocal current_block
        if not current_block:
            return

        current_block.question_text = _normalize_question_text(current_block.question_text)
        blocks.append(current_block)
        current_block = None

    def start_block(question_text: str) -> None:
        nonlocal current_block
        current_block = QuestionBlock(
            question_id=f"Q{len(blocks) + 1}",
            question_text=question_text,
            source_section=current_section,
        )

    for line in raw_lines:
        stripped = (line or "").strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            flush_current()
            current_section = re.sub(r"^#+\s*", "", stripped).strip() or current_section
            continue

        # Ignore answer-prefixed lines from mixed source docs.
        if _A_PREFIX.match(stripped):
            if current_block:
                answer_text = _A_PREFIX.sub("", stripped).strip()
                if answer_text:
                    current_block.answer_lines.append(answer_text)
            continue

        # Capture checklist options and attach them to the latest question.
        if _CHECKBOX_PREFIX.match(stripped):
            option_core = _clean_option_text(stripped)
            if option_core and current_block and _is_meaningful_checkbox_prompt(option_core):
                current_block.options.append(option_core)
            continue

        # Ignore markdown table/header noise.
        if stripped.startswith("|"):
            continue

        cleaned = _Q_PREFIX.sub("", re.sub(r"^[\-\*•]+\s*", "", stripped)).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            continue

        if _NOTE_LABEL.match(cleaned):
            # Attach the note label to the current block so it appears in the output.
            if current_block:
                current_block.note_label = cleaned
            continue

        # If we are inside a note (note_label is set but block is still open), capture
        # the note body lines rather than treating them as question content.
        if current_block and current_block.note_label and not _BLANK_FILLER.match(cleaned):
            if "?" not in cleaned and not _looks_like_question_clause(cleaned):
                current_block.note_text = (
                    (current_block.note_text + " " + cleaned).strip()
                    if current_block.note_text
                    else cleaned
                )
                continue

        if _BLANK_FILLER.match(cleaned):
            continue

        # Start a new question block for interrogative prompts.
        if "?" in cleaned or _looks_like_question_clause(cleaned):
            flush_current()
            prompt_text = cleaned.split("?")[0].strip() + "?" if "?" in cleaned else cleaned.rstrip(":")
            start_block(prompt_text)
            continue

        if current_block and not current_block.options:
            current_block.answer_lines.append(cleaned)

    flush_current()

    logger.info(
        "Extracted %d structured question blocks from '%s'",
        len(blocks),
        source_name,
    )
    return blocks


def _classify_question_block(block: QuestionBlock) -> str:
    if block.options:
        canonical_values = {_canonical_option_value(option) for option in block.options}
        canonical_values.discard(None)
        if "Yes" in canonical_values and "No" in canonical_values and len(canonical_values) <= 3:
            return DOMAIN_YES_NO
        return DOMAIN_CHECKLIST
    return DOMAIN_SENTENCE


def _extract_pdf_line_index(pdf_path: str) -> List[Dict[str, Any]]:
    try:
        import pdfplumber
    except Exception as e:
        raise RuntimeError(
            "pdfplumber is not available. Install it with: pip install pdfplumber"
        ) from e

    pdf_lines: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            chars = page.chars or []
            glyph_boxes = []
            for char in chars:
                text = (char.get("text") or "").strip()
                if text in _CHECKED_BOX_GLYPHS | _UNCHECKED_BOX_GLYPHS:
                    glyph_boxes.append(
                        {
                            "x0": char["x0"],
                            "x1": char["x1"],
                            "top": char["top"],
                            "bottom": char["bottom"],
                            "state": "checked" if text in _CHECKED_BOX_GLYPHS else "unchecked",
                            "evidence_type": "glyph",
                        }
                    )

            rect_boxes = []
            for rect in page.rects or []:
                width = rect["x1"] - rect["x0"]
                height = rect["bottom"] - rect["top"]
                if width < 6 or height < 6 or width > 18 or height > 18:
                    continue
                if abs(width - height) > 2:
                    continue

                interior_marks = [
                    char for char in chars
                    if char["x0"] >= rect["x0"] - 1
                    and char["x1"] <= rect["x1"] + 1
                    and char["top"] >= rect["top"] - 1
                    and char["bottom"] <= rect["bottom"] + 1
                    and (char.get("text") or "").strip() in _MARK_CHARS
                ]
                rect_boxes.append(
                    {
                        "x0": rect["x0"],
                        "x1": rect["x1"],
                        "top": rect["top"],
                        "bottom": rect["bottom"],
                        "state": "checked" if interior_marks else "unchecked",
                        "evidence_type": "rectangle",
                    }
                )

            page_boxes = glyph_boxes + rect_boxes
            words = page.extract_words(keep_blank_chars=False, use_text_flow=True) or []
            line_map: Dict[int, List[Dict[str, Any]]] = {}
            for word in words:
                line_key = int(round(float(word["top"]) / 3.0) * 3)
                line_map.setdefault(line_key, []).append(word)

            for line_words in line_map.values():
                line_words = sorted(line_words, key=lambda item: item["x0"])
                top = min(word["top"] for word in line_words)
                bottom = max(word["bottom"] for word in line_words)
                line_text = " ".join(word["text"] for word in line_words).strip()
                line_boxes = [
                    box for box in page_boxes
                    if ((box["top"] + box["bottom"]) / 2) >= top - 4
                    and ((box["top"] + box["bottom"]) / 2) <= bottom + 4
                ]
                pdf_lines.append(
                    {
                        "page": page_number,
                        "text": line_text,
                        "normalized_text": _normalize_match_text(line_text),
                        "top": top,
                        "bottom": bottom,
                        "words": [
                            {
                                "text": word["text"],
                                "normalized": _normalize_match_text(word["text"]),
                                "x0": word["x0"],
                                "x1": word["x1"],
                            }
                            for word in line_words
                        ],
                        "checkboxes": sorted(line_boxes, key=lambda item: item["x0"]),
                    }
                )

    return pdf_lines


def _find_best_anchor_index(question_text: str, pdf_lines: List[Dict[str, Any]]) -> Optional[int]:
    best_index = None
    best_score = 0.0
    for index, line in enumerate(pdf_lines):
        score = _overlap_score(question_text, line["text"])
        if _normalize_match_text(question_text) in line["normalized_text"]:
            score += 0.4
        if score > best_score:
            best_index = index
            best_score = score
    if best_score < 0.35:
        return None
    return best_index


def _find_option_span(line_words: List[Dict[str, Any]], option_text: str) -> Optional[Tuple[float, float]]:
    option_tokens = [token for token in _normalize_match_text(option_text).split() if token]
    normalized_words = [word["normalized"] for word in line_words if word["normalized"]]
    if not option_tokens or not normalized_words:
        return None

    for size in range(min(len(option_tokens), 5), 0, -1):
        probe = option_tokens[:size]
        for start in range(0, max(len(normalized_words) - size + 1, 0)):
            if normalized_words[start:start + size] == probe:
                return line_words[start]["x0"], line_words[start + size - 1]["x1"]
    return None


def _find_checkbox_evidence(
    option_text: str,
    pdf_lines: List[Dict[str, Any]],
    anchor_index: Optional[int],
) -> Optional[Dict[str, Any]]:
    if anchor_index is None:
        search_space = enumerate(pdf_lines)
    else:
        start = max(0, anchor_index)
        end = min(len(pdf_lines), anchor_index + 5)
        search_space = ((index, pdf_lines[index]) for index in range(start, end))

    best_match: Optional[Tuple[float, Dict[str, Any]]] = None
    for index, line in search_space:
        score = _overlap_score(option_text, line["text"])
        span = _find_option_span(line["words"], option_text)
        if span is None and score < 0.45:
            continue

        relevant_boxes = line["checkboxes"]
        if span is not None:
            relevant_boxes = [box for box in line["checkboxes"] if box["x1"] <= span[0] + 8]
        if not relevant_boxes and line["checkboxes"]:
            relevant_boxes = [line["checkboxes"][-1]]
        if not relevant_boxes:
            continue

        candidate = {
            "page": line["page"],
            "line_text": line["text"],
            "line_index": index,
            "option_text": option_text,
            "boxes": [
                {
                    "x0": round(box["x0"], 2),
                    "x1": round(box["x1"], 2),
                    "top": round(box["top"], 2),
                    "bottom": round(box["bottom"], 2),
                    "state": box["state"],
                    "evidence_type": box["evidence_type"],
                }
                for box in relevant_boxes
            ],
        }
        weighted_score = score + (0.2 if span is not None else 0.0)
        if best_match is None or weighted_score > best_match[0]:
            best_match = (weighted_score, candidate)

    return best_match[1] if best_match else None


def _build_sentence_record(block: QuestionBlock) -> Dict[str, Any]:
    return {
        "domain": DOMAIN_SENTENCE,
        "question_id": block.question_id,
        "question_text": block.question_text,
        "descriptive_answer": " ".join(block.answer_lines).strip() or None,
        "source_section": block.source_section,
        "note_label": block.note_label or None,
        "note_text": block.note_text or None,
        "extraction_method": "Docling",
        "confidence": "high",
        "flags": [],
    }


def _build_yes_no_record(block: QuestionBlock, pdf_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    anchor_index = _find_best_anchor_index(block.question_text, pdf_lines)
    evidence_map: Dict[str, Dict[str, Any]] = {}
    flags: List[str] = []
    selected_answer = None

    if anchor_index is None:
        flags.append("question_anchor_not_found_in_pdf")

    selected_options: List[str] = []
    branch_options: List[Dict[str, Any]] = []
    for option in block.options:
        canonical = _canonical_option_value(option)
        if canonical is None:
            continue
        evidence = _find_checkbox_evidence(option, pdf_lines, anchor_index)
        if evidence is None:
            flags.append(f"missing_checkbox_evidence_for_{canonical.lower()}")
            continue
        evidence_map[canonical] = evidence
        is_selected = any(box["state"] == "checked" for box in evidence["boxes"])
        branch_options.append(
            {
                "branch_name": canonical,
                "branch_text": option,
                "selected": is_selected,
                "checkbox_evidence": evidence,
            }
        )
        if is_selected:
            selected_options.append(canonical)

    if len(selected_options) == 1 and selected_options[0] in {"Yes", "No"}:
        selected_answer = selected_options[0]
    elif len(selected_options) > 1:
        flags.append("multiple_checkbox_options_selected")
    else:
        flags.append("selected_answer_unclear")

    return {
        "domain": DOMAIN_YES_NO,
        "question_id": block.question_id,
        "question_text": block.question_text,
        "selected_answer": selected_answer,
        "branch_options": branch_options,
        "checkbox_evidence": evidence_map or None,
        "source_section": block.source_section,
        "note_label": block.note_label or None,
        "note_text": block.note_text or None,
        "extraction_method": "Docling + pdfplumber",
        "confidence": _extraction_confidence(flags),
        "flags": sorted(set(flags)),
    }


def _build_checklist_records(block: QuestionBlock, pdf_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anchor_index = _find_best_anchor_index(block.question_text, pdf_lines)
    flags: List[str] = []
    if anchor_index is None:
        flags.append("question_anchor_not_found_in_pdf")

    records: List[Dict[str, Any]] = []
    for item_index, option in enumerate(block.options, 1):
        evidence = _find_checkbox_evidence(option, pdf_lines, anchor_index)
        selected_value = None
        item_flags = list(flags)
        table_row_reference = None
        if evidence is None:
            item_flags.append("missing_checkbox_evidence")
        else:
            table_row_reference = f"Page {evidence['page']} / line {evidence['line_index'] + 1}"
            if any(box["state"] == "checked" for box in evidence["boxes"]):
                selected_value = _canonical_option_value(option) or "Yes"
            elif any(box["state"] == "unchecked" for box in evidence["boxes"]):
                selected_value = "No"

        records.append(
            {
                "domain": DOMAIN_CHECKLIST,
                "checklist_item_id": f"{block.question_id}.{item_index}",
                "parent_question_id": block.question_id,
                "question_text": block.question_text,
                "checklist_text": option,
                "selected_value": selected_value,
                "table_row_reference": table_row_reference,
                "checkbox_evidence": evidence,
                "source_section": block.source_section,
                "note_label": block.note_label or None,
                "note_text": block.note_text or None,
                "extraction_method": "Docling + pdfplumber",
                "confidence": _extraction_confidence(item_flags),
                "flags": sorted(set(item_flags)),
            }
        )
    return records


def _build_question_catalog(by_domain: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}

    for record in by_domain.get(DOMAIN_SENTENCE, []):
        catalog[record["question_id"]] = {
            "question_id": record["question_id"],
            "question_text": record["question_text"],
            "domain": DOMAIN_SENTENCE,
            "source_section": record["source_section"],
            "record": record,
        }

    for record in by_domain.get(DOMAIN_YES_NO, []):
        catalog[record["question_id"]] = {
            "question_id": record["question_id"],
            "question_text": record["question_text"],
            "domain": DOMAIN_YES_NO,
            "source_section": record["source_section"],
            "record": record,
        }

    checklist_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in by_domain.get(DOMAIN_CHECKLIST, []):
        checklist_groups.setdefault(record["parent_question_id"], []).append(record)

    for parent_question_id, group in checklist_groups.items():
        first = group[0]
        catalog[parent_question_id] = {
            "question_id": parent_question_id,
            "question_text": first["question_text"],
            "domain": DOMAIN_CHECKLIST,
            "source_section": first["source_section"],
            "record": sorted(group, key=lambda item: _question_sort_key(item["checklist_item_id"])),
        }

    return catalog


def _render_validation_storage_text(entry: Dict[str, Any]) -> str:
    if entry["domain"] == DOMAIN_SENTENCE:
        record = entry["record"]
        return f"{record['question_text']}\n\n{record.get('descriptive_answer') or ''}".rstrip()

    if entry["domain"] == DOMAIN_YES_NO:
        record = entry["record"]
        return f"{record['question_text']}\n{_yes_no_line(record)}"

    group = entry["record"]
    lines = [group[0]["question_text"], ""]
    for item in group:
        lines.append(f"{_checklist_item_symbol(item)} {item['checklist_text']}")
    return "\n".join(lines)


def _build_gate_hierarchy(by_domain: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Build gate parent→branch→dependent-child structure for all yes/no gate questions.

    Dependent children for each gate are all immediately following questions
    within the same source section, stopping at the next yes/no gate question
    or a section boundary.  This scoping is applied uniformly to every gate
    question (no hardcoded per-section overrides).
    """
    catalog = _build_question_catalog(by_domain)
    ordered_question_ids = sorted(catalog, key=_question_sort_key)
    hierarchy: List[Dict[str, Any]] = []

    for index, question_id in enumerate(ordered_question_ids):
        entry = catalog[question_id]
        if entry["domain"] != DOMAIN_YES_NO:
            continue

        record = entry["record"]
        dependent_ids: List[str] = []
        for next_question_id in ordered_question_ids[index + 1:]:
            next_entry = catalog[next_question_id]
            # Stop at a section boundary
            if next_entry["source_section"] != entry["source_section"]:
                break
            # Stop at the next gate question (another yes/no)
            if next_entry["domain"] == DOMAIN_YES_NO:
                break
            dependent_ids.append(next_question_id)
        selected_branch = record.get("selected_answer") or "Unclear"
        dependency_status = "resolved"
        if dependent_ids and selected_branch == "Unclear":
            dependency_status = "unresolved_parent_choice"

        branches: List[Dict[str, Any]] = []
        for branch in record.get("branch_options") or [
            {"branch_name": "Yes", "branch_text": "Yes", "selected": False, "checkbox_evidence": None},
            {"branch_name": "No", "branch_text": "No", "selected": False, "checkbox_evidence": None},
        ]:
            branch_name = branch["branch_name"]
            if selected_branch == "Unclear":
                child_entries: List[Dict[str, Any]] = []
            elif selected_branch == branch_name:
                child_entries = [
                    {
                        "question_id": dep_id,
                        "question_text": catalog[dep_id]["question_text"],
                        "domain": catalog[dep_id]["domain"],
                        "dependency_status": "resolved",
                        "stored_text": _render_validation_storage_text(catalog[dep_id]),
                    }
                    for dep_id in dependent_ids
                ]
            else:
                child_entries = []

            branches.append(
                {
                    "branch_name": branch_name,
                    "branch_text": branch.get("branch_text") or branch_name,
                    "symbol": "☒" if selected_branch == branch_name else "☐",
                    "selected": selected_branch == branch_name,
                    "checkbox_evidence": branch.get("checkbox_evidence"),
                    "children": child_entries,
                }
            )

        unresolved_children = []
        if selected_branch == "Unclear":
            unresolved_children = [
                {
                    "question_id": dep_id,
                    "question_text": catalog[dep_id]["question_text"],
                    "domain": catalog[dep_id]["domain"],
                    "dependency_status": "unresolved_parent_choice",
                    "stored_text": _render_validation_storage_text(catalog[dep_id]),
                }
                for dep_id in dependent_ids
            ]

        hierarchy.append(
            {
                "parent_gate_question": {
                    "question_id": question_id,
                    "question_text": record["question_text"],
                    "source_section": record["source_section"],
                    "stored_text": _render_validation_storage_text(entry),
                },
                "branch_options": branches,
                "selected_branch": selected_branch,
                "dependency_status": dependency_status,
                "dependent_sub_questions": unresolved_children if selected_branch == "Unclear" else [],
            }
        )

    return hierarchy


def extract_questions_from_pdf(pdf_path: str) -> ExtractionBundle:
    """
    Extract and classify questions from a PDF using Docling and pdfplumber.
    Also extract and analyze images from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Question source PDF not found: {pdf_path}")

    raw_text = _read_file_text_with_docling(pdf_path, do_ocr=False)
    blocks = _extract_question_blocks_from_text(raw_text, os.path.basename(pdf_path))

    # OCR fallback: scanned/flattened PDFs often under-extract with text-only mode.
    # Keep this opt-in because OCR backends may require external model downloads.
    if len(blocks) < 25 and _ocr_fallback_enabled():
        try:
            raw_text_ocr = _read_file_text_with_docling(pdf_path, do_ocr=True)
            ocr_blocks = _extract_question_blocks_from_text(
                raw_text_ocr, f"{os.path.basename(pdf_path)} (ocr)"
            )
            if len(ocr_blocks) > len(blocks):
                logger.info(
                    "Using OCR extraction for '%s' (%d -> %d question blocks)",
                    os.path.basename(pdf_path),
                    len(blocks),
                    len(ocr_blocks),
                )
                blocks = ocr_blocks
        except Exception as exc:
            logger.warning("OCR fallback failed for '%s': %s", os.path.basename(pdf_path), exc)
    elif len(blocks) < 25:
        logger.info(
            "Skipping OCR fallback for '%s' (set ENABLE_OCR_FALLBACK=1 to enable).",
            os.path.basename(pdf_path),
        )

    pdf_lines = _extract_pdf_line_index(pdf_path)
    by_domain: Dict[str, List[Dict[str, Any]]] = {
        DOMAIN_SENTENCE: [],
        DOMAIN_CHECKLIST: [],
        DOMAIN_YES_NO: [],
    }
    question_texts: List[str] = []

    for block in blocks:
        question_texts.append(block.question_text)
        domain = _classify_question_block(block)
        if domain == DOMAIN_SENTENCE:
            by_domain[DOMAIN_SENTENCE].append(_build_sentence_record(block))
        elif domain == DOMAIN_YES_NO:
            by_domain[DOMAIN_YES_NO].append(_build_yes_no_record(block, pdf_lines))
        else:
            by_domain[DOMAIN_CHECKLIST].extend(_build_checklist_records(block, pdf_lines))

    gate_hierarchy = _build_gate_hierarchy(by_domain)
    
    # --- Extract and process images from question PDF ---
    extracted_images = []
    image_analyses = []
    
    try:
        logger.info(f"Extracting images from question PDF: {os.path.basename(pdf_path)}")
        image_extractor = get_image_extractor()
        image_processor = get_image_processor()
        image_cache = get_image_cache()
        
        # Extract images
        image_metadata_list = image_extractor.extract_all_images(pdf_path)
        extracted_images = image_metadata_list
        
        if image_metadata_list:
            logger.info(f"Processing {len(image_metadata_list)} images from question PDF...")
            
            for image_meta in image_metadata_list:
                image_path = image_meta["image_path"]
                
                # Check cache first
                cached_analysis = image_cache.get(image_path)
                if cached_analysis:
                    analysis = cached_analysis
                    logger.info(f"Using cached analysis for {image_meta['image_filename']}")
                else:
                    # Process image
                    analysis = image_processor.process_image_complete(image_path)
                    image_cache.set(image_path, analysis)
                
                image_analyses.append(analysis)
        else:
            logger.info(f"No images found in question PDF")
    
    except Exception as e:
        logger.warning(f"Error extracting/processing images from question PDF: {e}")
    
    return ExtractionBundle(
        by_domain=by_domain,
        question_texts=question_texts,
        gate_hierarchy=gate_hierarchy,
        extracted_images=extracted_images,
        image_analyses=image_analyses,
    )


def _parse_checked_items_from_answer(answer_text: str) -> List[str]:
    """Return the text of all ☒-marked lines in an LLM answer."""
    checked: List[str] = []
    for line in answer_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("☒"):
            item = re.sub(r"^☒\s*", "", stripped).strip()
            if item:
                checked.append(item)
    return checked


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _is_unknown_or_empty_answer(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    return "i don't know based on the provided documents" in normalized


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.6:
        return "Medium"
    return "Low"


def _render_answered_question_block(
    question_id: str,
    answer_record: Dict[str, Any],
    sentence_records: Dict[str, Dict[str, Any]],
    yes_no_records: Dict[str, Dict[str, Any]],
    checklist_groups: Dict[str, List[Dict[str, Any]]],
    note_lookup: Dict[str, tuple],
    prefix: str = "",
) -> List[str]:
    """Render one question in validation format with LLM-applied answer and source provenance."""
    raw_answer = answer_record.get("answer_text", "")
    sources = answer_record.get("sources", [])
    evidence = answer_record.get("evidence_sections", "")
    confidence_score = answer_record.get("confidence_score")
    confidence_label = answer_record.get("confidence_label")
    question_label = f"{prefix}{question_id}."
    result: List[str] = []

    if question_id in sentence_records:
        record = sentence_records[question_id]
        result.append(f"{question_label} {record['question_text']}")
        # Strip the trailing Evidence: line from the answer body for display
        body = re.sub(r"\s*Evidence:.*$", "", raw_answer, flags=re.IGNORECASE | re.DOTALL).strip()
        if _is_unknown_or_empty_answer(body):
            result.append(f"{prefix}Not Applicable")
        else:
            result.append(f"{prefix}{body}")
        result.append("")

    elif question_id in checklist_groups:
        group = sorted(
            checklist_groups[question_id],
            key=lambda item: _question_sort_key(item["checklist_item_id"]),
        )
        result.append(f"{question_label} {group[0]['question_text']}")
        result.append("")
        # Determine which items the LLM ticked
        checked_texts = _parse_checked_items_from_answer(raw_answer)
        checked_normalized = [_normalize_for_match(t) for t in checked_texts]
        any_ticked = False
        for item in group:
            item_norm = _normalize_for_match(item["checklist_text"])
            is_ticked = any(
                item_norm in cn or cn in item_norm or _overlap_score(item["checklist_text"], ct) >= 0.5
                for cn, ct in zip(checked_normalized, checked_texts)
            )
            if is_ticked:
                any_ticked = True
            symbol = "☒" if is_ticked else "☐"
            result.append(f"{prefix}{symbol} {item['checklist_text']}")

        if not any_ticked:
            # If present, select an explicit Not Applicable option.
            na_selected = False
            for idx, item in enumerate(group):
                if "not applicable" in _normalize_for_match(item["checklist_text"]):
                    result[-len(group) + idx] = f"{prefix}☒ {item['checklist_text']}"
                    na_selected = True
                    break
            if not na_selected:
                result.append(f"{prefix}Not Applicable")

        result.append("")

    elif question_id in yes_no_records:
        record = yes_no_records[question_id]
        result.append(f"{question_label} {record['question_text']}")
        # Determine LLM choice
        first_word = (raw_answer.strip().split()[0] if raw_answer.strip() else "").rstrip(".,;").lower()
        if first_word == "yes":
            llm_choice = "Yes"
        elif first_word == "no":
            llm_choice = "No"
        else:
            checked = _parse_checked_items_from_answer(raw_answer)
            cv = _canonical_option_value(checked[0]) if checked else None
            llm_choice = cv if cv in {"Yes", "No"} else None
        yes_sym = "☒" if llm_choice == "Yes" else "☐"
        no_sym  = "☒" if llm_choice == "No"  else "☐"
        result.append(f"{prefix}{yes_sym} Yes  {no_sym} No")
        if llm_choice is None:
            result.append(f"{prefix}Not Applicable")
        result.append("")

    # Append notes placeholder if the question has one
    if question_id in note_lookup:
        label, body = note_lookup[question_id]
        note_header = label if label else "Additional Notes:"
        result.append(f"{prefix}{note_header}")
        result.append(f"{prefix}{body}" if body else f"{prefix}")
        result.append("")

    # Source provenance
    if sources or evidence or confidence_score is not None:
        result.append(f"{prefix}[Sources]")
        if confidence_score is not None:
            label = confidence_label or _confidence_label(float(confidence_score))
            result.append(f"{prefix}  Confidence: {label} ({float(confidence_score):.2f})")
        if evidence:
            result.append(f"{prefix}  Evidence sections: {evidence}")
        for s in sources:
            result.append(f"{prefix}  {s}")
        result.append("")

    return result


def _build_answered_lines(
    extraction: ExtractionBundle,
    answer_index: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Build batch_answers.txt lines: each question in validation format with LLM answers + sources."""
    sentence_records = {
        rec["question_id"]: rec for rec in extraction.by_domain.get(DOMAIN_SENTENCE, [])
    }
    yes_no_records = {
        rec["question_id"]: rec for rec in extraction.by_domain.get(DOMAIN_YES_NO, [])
    }
    checklist_groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in extraction.by_domain.get(DOMAIN_CHECKLIST, []):
        checklist_groups.setdefault(rec["parent_question_id"], []).append(rec)

    note_lookup: Dict[str, tuple] = {}
    for domain_records in extraction.by_domain.values():
        for rec in domain_records:
            qid = rec.get("question_id") or rec.get("parent_question_id")
            if qid and (rec.get("note_label") or rec.get("note_text")):
                note_lookup.setdefault(qid, (rec.get("note_label", ""), rec.get("note_text", "")))

    ordered_ids = sorted(
        set(sentence_records) | set(yes_no_records) | set(checklist_groups),
        key=_question_sort_key,
    )

    # Build gate dependent map (child -> parent) so we can nest them
    gate_dependents: Dict[str, List[str]] = {}
    dependent_of: Dict[str, str] = {}
    for gate in extraction.gate_hierarchy:
        parent_id = gate["parent_gate_question"]["question_id"]
        dep_ids: List[str] = []
        seen: set = set()
        for branch in gate.get("branch_options", []):
            for child in branch.get("children", []):
                cid = child["question_id"]
                if cid not in seen:
                    dep_ids.append(cid)
                    seen.add(cid)
        for child in gate.get("dependent_sub_questions", []):
            cid = child["question_id"]
            if cid not in seen:
                dep_ids.append(cid)
                seen.add(cid)
        if dep_ids:
            gate_dependents[parent_id] = dep_ids
            for cid in dep_ids:
                dependent_of[cid] = parent_id

    lines: List[str] = []
    for question_id in ordered_ids:
        if question_id in dependent_of:
            continue  # emitted nested under its gate parent
        ans = answer_index.get(question_id, {"answer_text": "", "sources": [], "evidence_sections": ""})
        lines.extend(_render_answered_question_block(
            question_id, ans, sentence_records, yes_no_records,
            checklist_groups, note_lookup,
        ))
        if question_id in gate_dependents:
            lines.append("  [Dependent sub-questions]")
            for dep_id in gate_dependents[question_id]:
                dep_ans = answer_index.get(dep_id, {"answer_text": "", "sources": [], "evidence_sections": ""})
                lines.extend(_render_answered_question_block(
                    dep_id, dep_ans, sentence_records, yes_no_records,
                    checklist_groups, note_lookup, prefix="  ",
                ))
    return lines


def _build_answer_prompt_map(extraction: ExtractionBundle) -> Dict[str, str]:
    """Build question prompts for answering, preserving checkbox options when present."""
    prompt_map: Dict[str, str] = {}

    for record in extraction.by_domain.get(DOMAIN_SENTENCE, []):
        prompt_map[record["question_id"]] = record["question_text"]

    for record in extraction.by_domain.get(DOMAIN_YES_NO, []):
        prompt_map[record["question_id"]] = (
            f"{record['question_text']}\n"
            "Options:\n"
            "☐ Yes\n"
            "☐ No"
        )

    checklist_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in extraction.by_domain.get(DOMAIN_CHECKLIST, []):
        checklist_groups.setdefault(record["parent_question_id"], []).append(record)

    for parent_question_id, group in checklist_groups.items():
        sorted_group = sorted(group, key=lambda item: _question_sort_key(item["checklist_item_id"]))
        lines = [sorted_group[0]["question_text"], "Options:"]
        for item in sorted_group:
            lines.append(f"☐ {item['checklist_text']}")
        prompt_map[parent_question_id] = "\n".join(lines)

    return prompt_map


def _answer_questions(
    extraction: ExtractionBundle,
    out_txt_path: str,
    limit: Optional[int] = None,
) -> Tuple[str, int]:
    """Answer all questions, write batch_answers.txt in validation format with sources."""
    questions = extraction.question_texts
    if limit is not None and limit > 0:
        questions = questions[:limit]

    qa_chain = get_qa_chain(enable_reranking=True)
    meta = getattr(qa_chain, "metadata", {})
    llm = meta.get("llm")

    # Map question text -> question_id
    text_to_id: Dict[str, str] = {}
    for domain_records in extraction.by_domain.values():
        for rec in domain_records:
            qid = rec.get("question_id") or rec.get("parent_question_id")
            if qid and rec.get("question_text"):
                text_to_id[rec["question_text"]] = qid

    prompt_map = _build_answer_prompt_map(extraction)

    answer_index: Dict[str, Dict[str, Any]] = {}

    for i, q in enumerate(questions, 1):
        question_id = text_to_id.get(q, f"Q{i}")
        answer_prompt = prompt_map.get(question_id, q)
        results = retrieve_with_scores(answer_prompt)
        docs = [doc for doc, _ in results]
        confidence_score = 0.25

        if check_fallback(docs):
            answer = get_fallback_response()["answer"]
            sources: List[str] = []
            evidence_sections = ""
            confidence_score = 0.2
        else:
            reranked = None
            if meta.get("enable_reranking") and results:
                reranked = rerank_documents(llm, answer_prompt, results)
                if reranked:
                    docs = [r[0] for r in reranked[:8]]

            answer = answer_with_context(llm, answer_prompt, docs)
            sources = []
            section_ids: List[str] = []
            for d in docs[:5]:
                m = d.metadata
                sources.append(
                    f"[{m.get('section_id', '?')}] {m.get('section_title', '?')}"
                    f" (pages {m.get('page_start', '?')}-{m.get('page_end', '?')})"
                )
                section_ids.append(str(m.get("section_id", "?")))
            evidence_sections = ", ".join(sorted(set(section_ids)))

            # Confidence is primarily retrieval quality, adjusted by answer quality.
            if reranked:
                confidence_score = float(max(0.0, min(1.0, reranked[0][1])))
            elif results:
                confidence_score = float(max(0.0, min(1.0, results[0][1])))
            else:
                confidence_score = 0.25

            if len(sources) <= 1:
                confidence_score = max(0.0, confidence_score - 0.1)

            if _is_unknown_or_empty_answer(answer):
                confidence_score = min(confidence_score, 0.35)

        answer_index[question_id] = {
            "question_id": question_id,
            "answer_text": answer,
            "sources": sources,
            "evidence_sections": evidence_sections,
            "confidence_score": round(confidence_score, 2),
            "confidence_label": _confidence_label(confidence_score),
        }

    lines_out = _build_answered_lines(extraction, answer_index)
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out).strip() + "\n")

    logger.info("Answered %d questions; saved to %s", len(questions), out_txt_path)
    return out_txt_path, len(questions)


def _build_extraction_summary(extraction: ExtractionBundle) -> Dict[str, int]:
    checklist_records = extraction.by_domain.get(DOMAIN_CHECKLIST, [])
    checklist_parent_ids = {record["parent_question_id"] for record in checklist_records}
    total_question_ids = set()

    for record in extraction.by_domain.get(DOMAIN_SENTENCE, []):
        total_question_ids.add(record["question_id"])
    for record in extraction.by_domain.get(DOMAIN_YES_NO, []):
        total_question_ids.add(record["question_id"])
    total_question_ids.update(checklist_parent_ids)

    sentence_count = len(extraction.by_domain.get(DOMAIN_SENTENCE, []))
    yes_no_count = len(extraction.by_domain.get(DOMAIN_YES_NO, []))
    checklist_question_count = len(checklist_parent_ids)
    checklist_item_count = len(checklist_records)

    return {
        "total_parent_questions": len(total_question_ids),
        "total_extracted_entries": sentence_count + yes_no_count + checklist_item_count,
        "sentence_based_questions": sentence_count,
        "yes_no_questions": yes_no_count,
        "gate_questions": len(extraction.gate_hierarchy),
        "checklist_questions": checklist_question_count,
        "checklist_items": checklist_item_count,
    }


def _format_checkbox_evidence_summary(evidence_map: Optional[Dict[str, Dict[str, Any]]]) -> str:
    if not evidence_map:
        return "<unavailable>"

    parts: List[str] = []
    for option_name, evidence in evidence_map.items():
        boxes = evidence.get("boxes") or []
        states = sorted({box.get("state", "unknown") for box in boxes})
        page = evidence.get("page", "?")
        parts.append(f"{option_name}: page {page}, state={'+'.join(states)}")
    return "; ".join(parts)


def _question_sort_key(question_id: str) -> Tuple[int, int]:
    match = re.match(r"^Q(\d+)(?:\.(\d+))?$", question_id)
    if not match:
        return (10**9, 10**9)
    return int(match.group(1)), int(match.group(2) or 0)


def _checklist_item_symbol(record: Dict[str, Any]) -> str:
    selected_value = record.get("selected_value")
    if record.get("confidence") == "high" and selected_value not in {None, "No"}:
        return "☒"
    return "☐"


def _yes_no_line(record: Dict[str, Any]) -> str:
    selected_answer = record.get("selected_answer")
    if selected_answer == "Yes":
        return "☒ Yes ☐ No"
    if selected_answer == "No":
        return "☐ Yes ☒ No"
    return "☐ Yes ☐ No"


def _build_validation_ready_lines(extraction: ExtractionBundle) -> List[str]:
    sentence_records = {
        record["question_id"]: record for record in extraction.by_domain.get(DOMAIN_SENTENCE, [])
    }
    yes_no_records = {
        record["question_id"]: record for record in extraction.by_domain.get(DOMAIN_YES_NO, [])
    }
    checklist_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in extraction.by_domain.get(DOMAIN_CHECKLIST, []):
        checklist_groups.setdefault(record["parent_question_id"], []).append(record)

    ordered_question_ids = sorted(
        set(sentence_records) | set(yes_no_records) | set(checklist_groups),
        key=_question_sort_key,
    )

    # Build gate parent -> ordered dependent ids map from the gate hierarchy
    gate_dependents: Dict[str, List[str]] = {}
    dependent_of: Dict[str, str] = {}  # child_id -> parent_gate_id
    for gate in extraction.gate_hierarchy:
        parent_id = gate["parent_gate_question"]["question_id"]
        dep_ids: List[str] = []
        seen: set = set()
        for branch in gate.get("branch_options", []):
            for child in branch.get("children", []):
                cid = child["question_id"]
                if cid not in seen:
                    dep_ids.append(cid)
                    seen.add(cid)
        for child in gate.get("dependent_sub_questions", []):
            cid = child["question_id"]
            if cid not in seen:
                dep_ids.append(cid)
                seen.add(cid)
        if dep_ids:
            gate_dependents[parent_id] = dep_ids
            for cid in dep_ids:
                dependent_of[cid] = parent_id

    # Build a lookup for note_label/note_text per question_id from all domain records
    note_lookup: Dict[str, tuple] = {}
    for domain_records in extraction.by_domain.values():
        for rec in domain_records:
            qid = rec.get("question_id") or rec.get("parent_question_id")
            if qid and (rec.get("note_label") or rec.get("note_text")):
                note_lookup.setdefault(qid, (rec.get("note_label", ""), rec.get("note_text", "")))

    def _emit_note(question_id: str, prefix: str = "") -> List[str]:
        if question_id not in note_lookup:
            return []
        label, body = note_lookup[question_id]
        note_header = label if label else "Additional Notes:"
        lines: List[str] = [f"{prefix}{note_header}"]
        lines.append(f"{prefix}{body}" if body else f"{prefix}")
        lines.append("")
        return lines

    # Emit a single question block (no indent prefix by default)
    def _emit_question(question_id: str, prefix: str = "") -> List[str]:
        question_label = f"{prefix}{question_id}."
        result: List[str] = []
        if question_id in sentence_records:
            record = sentence_records[question_id]
            result.append(f"{question_label} {record['question_text']}")
            if record.get("descriptive_answer"):
                result.append(f"{prefix}{record['descriptive_answer']}")
            result.append("")
        elif question_id in checklist_groups:
            group = sorted(
                checklist_groups[question_id],
                key=lambda item: _question_sort_key(item["checklist_item_id"]),
            )
            result.append(f"{question_label} {group[0]['question_text']}")
            result.append("")
            for item in group:
                result.append(f"{prefix}{_checklist_item_symbol(item)} {item['checklist_text']}")
            result.append("")
        elif question_id in yes_no_records:
            record = yes_no_records[question_id]
            result.append(f"{question_label} {record['question_text']}")
            result.append(f"{prefix}{_yes_no_line(record)}")
            result.append("")
        result.extend(_emit_note(question_id, prefix=prefix))
        return result

    lines: List[str] = []
    for question_id in ordered_question_ids:
        # Dependent questions are emitted nested under their gate parent; skip here
        if question_id in dependent_of:
            continue

        lines.extend(_emit_question(question_id))

        # If this is a gate question, emit its dependents indented underneath
        if question_id in gate_dependents:
            lines.append("  [Dependent sub-questions — resolved when branch is selected]")
            for dep_id in gate_dependents[question_id]:
                for dep_line in _emit_question(dep_id, prefix="  "):
                    lines.append(dep_line)

    return lines


def _save_gate_hierarchy_report(extraction: ExtractionBundle, report_path: str) -> None:
    """Write a human-readable gate hierarchy report showing parents and branch children."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_lines: List[str] = [
        "GATE HIERARCHY REPORT",
        "=" * 60,
        "",
    ]

    if not extraction.gate_hierarchy:
        report_lines.append("(No gate questions found)")
    else:
        for gate_index, gate in enumerate(extraction.gate_hierarchy, start=1):
            parent = gate["parent_gate_question"]
            report_lines.append(
                f"[Gate {gate_index}]  {parent['question_id']}  "
                f"(Section: {parent['source_section'] or 'N/A'})"
            )
            report_lines.append(f"  Q: {parent['question_text']}")
            report_lines.append(f"  Selected branch: {gate['selected_branch']}")
            report_lines.append(f"  Dependency status: {gate['dependency_status']}")
            report_lines.append("  Branch options:")
            for branch in gate.get("branch_options", []):
                report_lines.append(
                    f"    {branch['symbol']} {branch['branch_name']}"
                )
                for child in branch.get("children", []):
                    report_lines.append(
                        f"      -> {child['question_id']}: {child['question_text']}"
                    )

            if gate.get("dependent_sub_questions"):
                report_lines.append("  Unresolved dependents (branch unclear):")
                for child in gate["dependent_sub_questions"]:
                    report_lines.append(
                        f"      ? {child['question_id']}: {child['question_text']}"
                    )

            report_lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines).rstrip() + "\n")

    logger.info("Saved gate hierarchy report to %s", report_path)


def _save_fetched_questions(extraction: ExtractionBundle, out_path: str) -> str:
    """Persist extracted questions by domain for audit/debug visibility."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = _build_extraction_summary(extraction)
    lines = _build_validation_ready_lines(extraction)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    json_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "gate_hierarchy": extraction.gate_hierarchy,
                "stored_text": "\n".join(lines).strip(),
                "domains": extraction.by_domain,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    report_path = os.path.join(os.path.dirname(out_path), "gate_hierarchy_report.txt")
    _save_gate_hierarchy_report(extraction, report_path)

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
    extraction = extract_questions_from_pdf(pdf_path)

    _save_fetched_questions(extraction, fetched_questions_path)
    out_path, count = _answer_questions(extraction, out_txt_path, limit=limit)
    logger.info("Answered %d questions from %s", count, os.path.basename(pdf_path))
    return out_path, count
