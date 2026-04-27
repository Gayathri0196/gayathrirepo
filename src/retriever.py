"""
retriever.py -- Hybrid Retriever (Semantic + Lexical + RRF)
-----------------------------------------------------------
Retrieves broader, interlinked context by combining:
  - Semantic vector retrieval (Chroma similarity)
  - Lexical retrieval over processed chunks.json
  - Reciprocal Rank Fusion (RRF) across rankings
  - MMR diversification
  - Related section linking (neighbors + parents)
"""

import logging
import os
import re
import json
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from vector_store import load_vector_store

logger = logging.getLogger(__name__)

# -- Configuration -------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.2
TOP_K = 8
MMR_K = 6
MMR_FETCH_K = 20
NEIGHBOR_HOPS = 1
RRF_K = 60

PROCESSED_CHUNKS_JSON = os.path.join("..", "data", "processed_docs", "chunks.json")

QUERY_ALIASES = {
    "st": "system testing",
    "uat": "user acceptance testing",
    "iq": "installation qualification",
    "oq": "operational qualification",
    "csv": "computer system validation",
    "dm": "data migration",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at",
    "is", "are", "be", "will", "there", "this", "that", "with", "as", "it",
    "by", "from", "into", "about", "what", "which", "who", "how", "when",
    "where", "why", "do", "does", "did", "can", "could", "should", "would",
    "all", "any", "each", "also", "than", "then", "if", "yes", "no",
}

_CHUNK_CACHE = {"loaded": False, "by_sid": {}, "ordered_sids": []}


def _normalize_sid(sid: str) -> str:
    return (sid or "").strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _query_tokens(text: str) -> List[str]:
    toks = []
    for t in _tokenize(text):
        if len(t) <= 1 or t in STOPWORDS:
            continue
        toks.append(t)
    return toks


def _load_chunk_cache() -> None:
    if _CHUNK_CACHE["loaded"]:
        return

    by_sid = {}
    ordered_sids = []
    if os.path.exists(PROCESSED_CHUNKS_JSON):
        try:
            with open(PROCESSED_CHUNKS_JSON, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for c in chunks:
                sid = _normalize_sid(c.get("section_id", ""))
                if not sid:
                    continue
                by_sid[sid] = c
                ordered_sids.append(sid)
        except Exception as e:
            logger.warning(f"Could not load processed chunks cache: {e}")

    _CHUNK_CACHE["by_sid"] = by_sid
    _CHUNK_CACHE["ordered_sids"] = ordered_sids
    _CHUNK_CACHE["loaded"] = True


def _expand_query_variants(query: str) -> List[str]:
    variants = [re.sub(r"\s+", " ", query).strip()]
    q_low = f" {query.lower()} "

    expanded = query
    for short, long_form in QUERY_ALIASES.items():
        expanded = re.sub(rf"\b{re.escape(short)}\b", long_form, expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\s+", " ", expanded).strip()
    if expanded and expanded not in variants:
        variants.append(expanded)

    if "yes" in q_low or "no" in q_low or "check" in q_low or "checkbox" in q_low:
        focus = f"{expanded} selected option check yes no applicability"
        focus = re.sub(r"\s+", " ", focus).strip()
        if focus not in variants:
            variants.append(focus)

    return variants[:3]


def _parent_section_ids(sid: str) -> List[str]:
    parts = sid.split(".")
    parents = []
    while len(parts) > 1:
        parts = parts[:-1]
        parents.append(".".join(parts))
    return parents


def _related_section_ids(seed_sids: List[str], hops: int = NEIGHBOR_HOPS) -> List[str]:
    _load_chunk_cache()
    ordered = _CHUNK_CACHE["ordered_sids"]
    if not ordered:
        return []

    index = {sid: i for i, sid in enumerate(ordered)}
    related = set()

    for sid in seed_sids:
        if sid not in index:
            continue

        i = index[sid]
        for h in range(1, hops + 1):
            if i - h >= 0:
                related.add(ordered[i - h])
            if i + h < len(ordered):
                related.add(ordered[i + h])

        for p in _parent_section_ids(sid):
            if p in index:
                related.add(p)

    for sid in seed_sids:
        related.discard(sid)

    return list(related)


def _chunk_to_document(chunk: dict) -> Document:
    return Document(
        page_content=chunk.get("text", ""),
        metadata={
            "domain": chunk.get("domain", "?"),
            "section_id": chunk.get("section_id", "?"),
            "section_title": chunk.get("section_title", "?"),
            "source_doc": chunk.get("source_doc", "?"),
            "page_start": chunk.get("page_start", "?"),
            "page_end": chunk.get("page_end", "?"),
        },
    )


def _lexical_scores(query_variants: List[str]) -> Dict[str, float]:
    _load_chunk_cache()
    by_sid = _CHUNK_CACHE["by_sid"]
    if not by_sid:
        return {}

    score_by_sid: Dict[str, float] = {}
    for qv in query_variants:
        q_tokens = _query_tokens(qv)
        if not q_tokens:
            continue

        q_set = set(q_tokens)
        q_norm = _normalize(qv)
        for sid, chunk in by_sid.items():
            text = chunk.get("text", "")
            title = chunk.get("section_title", "")
            domain = chunk.get("domain", "")
            doc_text = f"{title} {domain} {text}"

            d_tokens = _query_tokens(doc_text)
            if not d_tokens:
                continue

            d_set = set(d_tokens)
            overlap = len(q_set & d_set) / max(1, len(q_set))

            title_set = set(_query_tokens(f"{title} {domain}"))
            title_overlap = len(q_set & title_set) / max(1, len(q_set))

            doc_norm = _normalize(doc_text)
            phrase_bonus = 0.35 if q_norm and q_norm in doc_norm else 0.0

            checkbox_bonus = 0.0
            if ("check" in q_set or "selected" in q_set or "option" in q_set) and "☒" in text:
                checkbox_bonus += 0.15

            score = (overlap * 0.65) + (title_overlap * 0.20) + phrase_bonus + checkbox_bonus
            if score <= 0:
                continue

            if sid not in score_by_sid or score > score_by_sid[sid]:
                score_by_sid[sid] = score

    return score_by_sid


def _rrf_fuse(rank_lists: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for rlist in rank_lists:
        for rank, sid in enumerate(rlist, start=1):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (k + rank)
    return fused


def get_retriever(threshold: float = SIMILARITY_THRESHOLD, top_k: int = TOP_K):
    vectordb = load_vector_store()

    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": threshold,
            "k": top_k,
        },
    )

    logger.info(f"Retriever initialized: threshold={threshold}, top_k={top_k}")
    return retriever


def retrieve_with_scores(query: str, threshold: float = SIMILARITY_THRESHOLD, top_k: int = TOP_K) -> list:
    """Hybrid retrieval returning (Document, score) tuples."""
    vectordb = load_vector_store()
    query_variants = _expand_query_variants(query)

    # 1) Semantic retrieval over query variants.
    merged_raw: Dict[str, Tuple[Document, float]] = {}
    for qv in query_variants:
        variant_results = vectordb.similarity_search_with_relevance_scores(qv, k=max(top_k * 2, 10))
        for doc, score in variant_results:
            sid = doc.metadata.get("section_id", "")
            if not sid:
                continue
            if sid not in merged_raw or score > merged_raw[sid][1]:
                merged_raw[sid] = (doc, score)

    semantic_ranked = sorted(merged_raw.items(), key=lambda kv: kv[1][1], reverse=True)
    semantic_ranked_sids = [sid for sid, _ in semantic_ranked]

    # 2) Lexical retrieval over chunks.json.
    lexical_by_sid = _lexical_scores(query_variants)
    lexical_ranked = sorted(lexical_by_sid.items(), key=lambda kv: kv[1], reverse=True)
    lexical_ranked_sids = [sid for sid, _ in lexical_ranked]

    # 3) Fuse semantic + lexical ranks.
    rrf_scores = _rrf_fuse([semantic_ranked_sids, lexical_ranked_sids], k=RRF_K)
    fused_ranked_sids = [sid for sid, _ in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)]

    _load_chunk_cache()
    by_sid = _CHUNK_CACHE["by_sid"]
    max_lex = max(lexical_by_sid.values()) if lexical_by_sid else 1.0
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

    # 4) Build hybrid candidate list.
    max_candidates = max(top_k * 4, 20)
    candidate_sids = fused_ranked_sids[:max_candidates]

    filtered: List[Tuple[Document, float]] = []
    for sid in candidate_sids:
        sem = merged_raw.get(sid)
        if sem:
            doc, sem_score = sem
            sem_norm = max(0.0, min(1.0, float(sem_score)))
        else:
            chunk = by_sid.get(sid)
            if not chunk:
                continue
            doc = _chunk_to_document(chunk)
            sem_norm = 0.0

        lex_norm = max(0.0, lexical_by_sid.get(sid, 0.0) / max_lex)
        rrf_norm = max(0.0, rrf_scores.get(sid, 0.0) / max_rrf)
        hybrid_score = (sem_norm * 0.55) + (lex_norm * 0.30) + (rrf_norm * 0.15)

        if sem_norm >= threshold or lex_norm >= 0.35 or rrf_norm >= 0.50:
            filtered.append((doc, hybrid_score))

    # Adaptive relaxation with semantic fallback when too sparse.
    semantic_values = list(merged_raw.values())
    if len(filtered) < max(3, top_k // 2):
        relaxed_threshold = max(0.1, threshold - 0.08)
        relaxed = [(doc, score) for doc, score in semantic_values if score >= relaxed_threshold]
        present = {d.metadata.get("section_id", "") for d, _ in filtered}
        for doc, score in relaxed:
            sid = doc.metadata.get("section_id", "")
            if sid not in present:
                filtered.append((doc, max(0.0, min(1.0, float(score)))))
        logger.info(f"Adaptive threshold applied: {threshold:.2f} -> {relaxed_threshold:.2f}")

    # 5) MMR diversification over first query variants.
    mmr_docs = []
    for qv in query_variants[:2]:
        try:
            mmr_docs.extend(
                vectordb.max_marginal_relevance_search(
                    qv,
                    k=MMR_K,
                    fetch_k=max(MMR_FETCH_K, top_k * 3),
                )
            )
        except Exception:
            continue

    score_by_section = {d.metadata.get("section_id", ""): s for d, s in filtered}
    present = {d.metadata.get("section_id", "") for d, _ in filtered}
    for d in mmr_docs:
        sid = d.metadata.get("section_id", "")
        if sid and sid not in present:
            filtered.append((d, score_by_section.get(sid, max(0.12, threshold - 0.05))))
            present.add(sid)

    # 6) Link related sections (neighbors + parents).
    seed_sids = [doc.metadata.get("section_id", "") for doc, _ in filtered]
    related_sids = _related_section_ids(seed_sids, hops=NEIGHBOR_HOPS)
    for sid in related_sids:
        if sid in present:
            continue
        chunk = by_sid.get(sid)
        if not chunk:
            continue
        filtered.append((_chunk_to_document(chunk), max(0.12, threshold - 0.05)))
        present.add(sid)

    # Deduplicate by best score and trim.
    best_by_section = {}
    for doc, score in filtered:
        sid = doc.metadata.get("section_id", "")
        if sid not in best_by_section or score > best_by_section[sid][1]:
            best_by_section[sid] = (doc, score)

    out = list(best_by_section.values())
    out.sort(key=lambda x: x[1], reverse=True)
    out = out[:top_k]

    logger.info(
        f"Query: '{query[:80]}...' -> {len(semantic_values)} semantic raw, "
        f"{len(out)} hybrid results"
    )
    for doc, score in out:
        logger.info(
            f"  Score: {score:.3f} | Domain: {doc.metadata.get('domain', '?')} | "
            f"Section: {doc.metadata.get('section_title', '?')}"
        )

    return out
