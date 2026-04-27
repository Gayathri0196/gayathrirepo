"""
ingestion.py -- Fully Dynamic Document Ingestion & Chunking
------------------------------------------------------------
Reads any PDF and dynamically creates chunks based on the actual
content structure found in that specific file.

Nothing is hardcoded to a particular document -- headings, domains,
and chunk counts are all derived from the input file itself.

Chunking strategy:
  1. Auto-detect heading style from the document content
  2. Validate detected headings (filter noise like years, table rows)
  3. Split text at validated headings
  4. Merge undersized chunks, split oversized ones
  5. Derive domain labels dynamically from the detected section titles
"""

import os
import re
import json
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# -- Configuration -------------------------------------------------------------

# Source inputs
SOURCE_DIR = os.path.join("..", "input file")
SOURCE_PDF = os.path.join(SOURCE_DIR, "LMS Test Plan Sample.pdf")

# Output path for cached processed chunks
PROCESSED_DIR = os.path.join("..", "data", "processed_docs")

# --- Chunking Parameters ---
MIN_CHUNK_CHARS = 80           # chunks shorter than this get merged
MAX_CHUNK_CHARS = 3000         # chunks longer than this get split
OVERLAP_CHARS = 150            # overlap between split sub-chunks
FALLBACK_CHUNK_SIZE = 1500     # target size when no headings found
FALLBACK_OVERLAP = 200         # overlap for paragraph-based fallback


# -- Heading Detection Patterns ------------------------------------------------
# Multiple patterns tried in order; the one that finds the most valid headings wins.

HEADING_PATTERNS = [
    # Pattern 1: Numbered with dots -- "1.0 Title", "3.1 Title", "1.2.1 Title"
    {
        "name": "numbered_dot",
        "regex": re.compile(r"^(\d{1,2}(?:\.\d+)+\.?)\s+(.+)$", re.MULTILINE),
        "id_group": 1,
        "title_group": 2,
    },
    # Pattern 2: Numbered without dots -- "1 Title", "2 Title" (only if few matches)
    {
        "name": "numbered_plain",
        "regex": re.compile(r"^(\d{1,2})\s+([A-Z][A-Za-z].{5,})$", re.MULTILINE),
        "id_group": 1,
        "title_group": 2,
    },
    # Pattern 3: Lettered sections -- "A. Title", "B. Title"
    {
        "name": "lettered",
        "regex": re.compile(r"^([A-Z])\.\s+(.{5,})$", re.MULTILINE),
        "id_group": 1,
        "title_group": 2,
    },
    # Pattern 4: ALL CAPS lines (often headings in documents)
    {
        "name": "all_caps",
        "regex": re.compile(r"^([A-Z][A-Z\s&/]{4,})$", re.MULTILINE),
        "id_group": 0,   # full match is both id and title
        "title_group": 0,
    },
]


def _is_valid_heading(section_id: str, title: str) -> bool:
    """
    Validate that a detected match is actually a section heading.
    Filters out years (2026.1), version strings, page numbers, etc.
    """
    # If section_id is numeric with dots, check the major number
    parts = section_id.replace(".", " ").split()
    if parts and parts[0].isdigit():
        major = int(parts[0])
        if major > 99:  # filters years like 2026
            return False

    # Title must have at least 1 real alphabetic word (2+ letters)
    alpha_words = [w for w in title.split() if re.search(r"[a-zA-Z]{2,}", w)]
    if len(alpha_words) < 1:
        return False

    # Reject common noise patterns
    title_lower = title.strip().lower()
    noise_prefixes = ("page ", "(mock", "version ", "v1.", "v2.", "v3.", "http")
    if title_lower.startswith(noise_prefixes):
        return False

    return True


def _detect_headings(full_text: str):
    """
    Try all heading patterns against the document text.
    Return the list of validated heading matches from the best pattern.
    """
    best_headings = []
    best_pattern_name = "none"

    for pattern in HEADING_PATTERNS:
        raw_matches = list(pattern["regex"].finditer(full_text))
        valid = []
        last_major = -1

        for m in raw_matches:
            if pattern["id_group"] == 0:
                sid = m.group(0).strip()
                title = m.group(0).strip()
            else:
                sid = m.group(pattern["id_group"]).rstrip(".")
                title = m.group(pattern["title_group"]).strip()

            if not _is_valid_heading(sid, title):
                continue

            # Monotonic order check for numbered patterns
            parts = sid.split(".")
            if parts[0].isdigit():
                major = int(parts[0])
                if major < last_major:
                    continue
                last_major = major

            valid.append({
                "match": m,
                "section_id": sid,
                "section_title": title,
            })

        if len(valid) > len(best_headings):
            best_headings = valid
            best_pattern_name = pattern["name"]

    logger.info(f"Heading detection: pattern='{best_pattern_name}', found {len(best_headings)} valid headings")
    return best_headings


def _derive_domain(section_title: str) -> str:
    """
    Dynamically derive a domain label from the section title itself.
    Uses the core meaningful words from the title as the domain name.
    No hardcoded mapping -- works with any document.
    """
    # Remove numbering prefix if present
    cleaned = re.sub(r"^\d[\d.]*\s*", "", section_title).strip()

    # Remove common noise words to get the core topic
    noise = {"the", "a", "an", "of", "for", "and", "in", "to", "with", "on", "is", "by"}
    words = cleaned.split()
    core_words = [w for w in words if w.lower() not in noise and len(w) > 1]

    if not core_words:
        return section_title.strip() if section_title.strip() else "General"

    # Capitalize as a clean domain label
    domain = " ".join(core_words)
    # Truncate overly long domain labels to first few meaningful words
    if len(domain) > 40:
        domain = " ".join(core_words[:4])

    return domain


# -- PDF Text Extraction ------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extract text from each page of a PDF using PyMuPDF.
    Returns list of {"page": int, "text": str}.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, 1):
        text = page.get_text("text") or ""
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if cleaned:
            pages.append({"page": i, "text": cleaned})
    doc.close()

    logger.info(f"Extracted text from {len(pages)} pages of '{os.path.basename(pdf_path)}'")
    return pages


# -- Helper: resolve page number from character offset -------------------------

def _resolve_pages(start: int, end: int, page_boundaries: list, last_page: int):
    """Return (page_start, page_end) for a character range."""
    page_start = 1
    page_end = last_page
    for offset, pnum in page_boundaries:
        if offset <= start:
            page_start = pnum
        if offset <= end:
            page_end = pnum
    return page_start, page_end


# -- Fallback: paragraph-based chunking ----------------------------------------

def _chunk_by_paragraphs(full_text: str, source_filename: str, page_boundaries: list, last_page: int) -> list:
    """
    Fallback when no section headings are detected.
    Splits by paragraphs with target size and overlap.
    """
    logger.info("No headings found -- using paragraph-based chunking")
    paragraphs = re.split(r"\n\s*\n", full_text)
    chunks = []
    current_text = ""
    part_num = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_text) + len(para) > FALLBACK_CHUNK_SIZE and current_text:
            part_num += 1
            # Find page range for this chunk
            chunk_start = full_text.find(current_text[:50])
            if chunk_start < 0:
                chunk_start = 0
            ps, pe = _resolve_pages(chunk_start, chunk_start + len(current_text), page_boundaries, last_page)
            chunks.append({
                "text": current_text.strip(),
                "section_id": f"p{part_num}",
                "section_title": f"Part {part_num}",
                "domain": "General",
                "source_doc": source_filename,
                "page_start": ps,
                "page_end": pe,
            })
            # Overlap: carry last portion into next chunk
            if FALLBACK_OVERLAP > 0:
                current_text = current_text[-FALLBACK_OVERLAP:].strip() + "\n\n" + para
            else:
                current_text = para
        else:
            current_text = (current_text + "\n\n" + para).strip()

    if current_text.strip():
        part_num += 1
        chunk_start = full_text.find(current_text[:50])
        if chunk_start < 0:
            chunk_start = 0
        ps, pe = _resolve_pages(chunk_start, chunk_start + len(current_text), page_boundaries, last_page)
        chunks.append({
            "text": current_text.strip(),
            "section_id": f"p{part_num}",
            "section_title": f"Part {part_num}",
            "domain": "General",
            "source_doc": source_filename,
            "page_start": ps,
            "page_end": pe,
        })

    logger.info(f"Created {len(chunks)} paragraph-based chunks")
    return chunks


# -- Main Chunking Logic -------------------------------------------------------

def chunk_by_sections(pages: list, source_filename: str) -> list:
    """
    Dynamically chunk document based on its actual content structure.
    Adapts to any PDF -- number of chunks depends on the document.
    """
    # Combine all pages into one text block with page markers
    full_text = ""
    page_boundaries = []

    for p in pages:
        offset = len(full_text)
        page_boundaries.append((offset, p["page"]))
        full_text += p["text"] + "\n\n"

    last_page = pages[-1]["page"] if pages else 1

    # --- Step 1: Auto-detect headings from document content ---
    headings = _detect_headings(full_text)

    # If no headings detected, fall back to paragraph chunking
    if not headings:
        return _chunk_by_paragraphs(full_text, source_filename, page_boundaries, last_page)

    # --- Step 2: Create raw chunks between detected headings ---
    raw_chunks = []
    for i, h in enumerate(headings):
        start = h["match"].start()
        end = headings[i + 1]["match"].start() if i + 1 < len(headings) else len(full_text)
        section_text = full_text[start:end].strip()
        page_start, page_end = _resolve_pages(start, end, page_boundaries, last_page)

        raw_chunks.append({
            "text": section_text,
            "section_id": h["section_id"],
            "section_title": h["section_title"],
            "domain": _derive_domain(h["section_title"]),
            "source_doc": source_filename,
            "page_start": page_start,
            "page_end": page_end,
        })

    # --- Step 3: Merge undersized chunks ---
    # Forward-merge tiny first chunk
    if len(raw_chunks) >= 2 and len(raw_chunks[0]["text"]) < MIN_CHUNK_CHARS:
        tiny = raw_chunks[0]
        nxt = raw_chunks[1]
        nxt["text"] = tiny["text"] + "\n\n" + nxt["text"]
        nxt["page_start"] = min(tiny["page_start"], nxt["page_start"])
        logger.info(f"  Merged tiny first chunk [{tiny['section_id']}] ({len(tiny['text'])} chars) into [{nxt['section_id']}]")
        raw_chunks = raw_chunks[1:]

    # Backward-merge remaining small chunks
    merged_chunks = []
    for chunk in raw_chunks:
        if len(chunk["text"]) < MIN_CHUNK_CHARS and merged_chunks:
            prev = merged_chunks[-1]
            prev["text"] = prev["text"] + "\n\n" + chunk["text"]
            prev["page_end"] = max(prev["page_end"], chunk["page_end"])
            logger.info(f"  Merged small [{chunk['section_id']}] ({len(chunk['text'])} chars) into [{prev['section_id']}]")
        else:
            merged_chunks.append(chunk)

    # --- Step 4: Split oversized chunks at paragraph boundaries ---
    final_chunks = []
    for chunk in merged_chunks:
        if len(chunk["text"]) <= MAX_CHUNK_CHARS:
            final_chunks.append(chunk)
            continue

        # Split at double-newline (paragraph) boundaries
        paragraphs = re.split(r"\n\s*\n", chunk["text"])
        current_text = ""
        part_num = 0

        for para in paragraphs:
            if len(current_text) + len(para) > MAX_CHUNK_CHARS and current_text:
                part_num += 1
                overlap = ""
                if part_num > 1 and OVERLAP_CHARS > 0:
                    overlap = current_text[-OVERLAP_CHARS:].strip() + "\n\n"
                final_chunks.append({
                    "text": current_text.strip(),
                    "section_id": f"{chunk['section_id']}_p{part_num}",
                    "section_title": f"{chunk['section_title']} (part {part_num})",
                    "domain": chunk["domain"],
                    "source_doc": chunk["source_doc"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                })
                current_text = overlap + para
            else:
                current_text = (current_text + "\n\n" + para).strip()

        if current_text.strip():
            part_num += 1
            sid = f"{chunk['section_id']}_p{part_num}" if part_num > 1 else chunk["section_id"]
            stitle = f"{chunk['section_title']} (part {part_num})" if part_num > 1 else chunk["section_title"]
            final_chunks.append({
                "text": current_text.strip(),
                "section_id": sid,
                "section_title": stitle,
                "domain": chunk["domain"],
                "source_doc": chunk["source_doc"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
            })

        if part_num > 1:
            logger.info(f"  Split oversized [{chunk['section_id']}] ({len(chunk['text'])} chars) into {part_num} parts")

    # --- Log summary ---
    total_chars = sum(len(c["text"]) for c in final_chunks)
    avg_chars = total_chars // len(final_chunks) if final_chunks else 0
    logger.info(f"Created {len(final_chunks)} chunks from '{source_filename}'")
    logger.info(f"  Total chars: {total_chars}, Avg per chunk: {avg_chars}")
    for c in final_chunks:
        logger.info(
            f"  [{c['section_id']}] {c['section_title']} -> domain: {c['domain']} "
            f"(pages {c['page_start']}-{c['page_end']}, {len(c['text'])} chars)"
        )

    return final_chunks


# -- Cache Processed Chunks ----------------------------------------------------

def save_chunks(chunks: list, output_dir: str = PROCESSED_DIR) -> str:
    """Save chunks as JSON for inspection/caching. Returns output path."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to '{output_path}'")
    return output_path


def _slugify_filename(filename: str) -> str:
    """Create a stable source key from a filename for multi-file section IDs."""
    stem = os.path.splitext(filename)[0].lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or "doc"


def _list_source_pdfs(input_dir: str) -> list:
    """Return sorted list of PDF paths found in input directory."""
    if not os.path.isdir(input_dir):
        return []
    pdfs = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(".pdf"):
            pdfs.append(os.path.join(input_dir, name))
    return sorted(pdfs)


def _apply_multisource_ids(chunks: list, source_filename: str) -> list:
    """Prefix section IDs with source key to avoid collisions across files."""
    source_key = _slugify_filename(source_filename)
    for chunk in chunks:
        original_id = chunk.get("section_id", "?")
        chunk["section_original_id"] = original_id
        chunk["section_id"] = f"{source_key}:{original_id}"
    return chunks


# -- Main Entry Point ----------------------------------------------------------

def ingest(pdf_paths: list | None = None) -> list:
    """
    Full ingestion pipeline:
    1. Discover source PDFs (or use explicit paths)
    2. Extract text from each source PDF
    3. Dynamically detect structure and chunk accordingly
    4. Cache combined chunks to disk
    5. Return list of chunks
    """
    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE START")
    logger.info("=" * 60)

    paths = list(pdf_paths) if pdf_paths else _list_source_pdfs(SOURCE_DIR)
    if not paths and os.path.exists(SOURCE_PDF):
        # Backward-compatible fallback to legacy single file.
        paths = [SOURCE_PDF]
    if not paths:
        raise FileNotFoundError(
            f"No PDF source files found in '{SOURCE_DIR}'. Add one or more .pdf files and rerun ingest."
        )

    chunks = []
    is_multisource = len(paths) > 1
    logger.info(f"Found {len(paths)} source PDF(s) for ingestion")

    for pdf_path in paths:
        source_filename = os.path.basename(pdf_path)
        pages = extract_text_from_pdf(pdf_path)
        doc_chunks = chunk_by_sections(pages, source_filename)
        if is_multisource:
            doc_chunks = _apply_multisource_ids(doc_chunks, source_filename)
        chunks.extend(doc_chunks)

    # Keep deterministic ordering for repeatable retrieval/cache builds.
    chunks.sort(key=lambda c: (c.get("source_doc", ""), c.get("section_id", "")))

    save_chunks(chunks)

    logger.info(
        f"INGESTION COMPLETE: {len(chunks)} chunks from {len(paths)} source PDF(s) ready for embedding"
    )
    return chunks


if __name__ == "__main__":
    # Allow running standalone for testing
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    chunks = ingest()
    print(f"\n[OK] Ingested {len(chunks)} chunks:")
    for c in chunks:
        print(f"  [{c['section_id']}] {c['section_title']} ({c['domain']}) - {len(c['text'])} chars")
