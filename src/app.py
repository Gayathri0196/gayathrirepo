"""
app.py -- RAG Agent Orchestration & CLI Interface
-------------------------------------------------
Main entry point for the RAG retrieval agent.

Two modes:
  1. INGEST: Process the PDF -> chunk -> embed -> store in Chroma
     > python app.py ingest

  2. CHAT: Interactive Q&A loop using the RAG pipeline
     > python app.py chat

The chat mode:
  - Retrieves relevant chunks from Chroma
  - If chunks found (above threshold): sends to LLM for answer
  - If no chunks found: returns 'I don't know' (no LLM call)
"""

import os
import sys
import io
import logging
from datetime import datetime

# Ensure we run from src/ directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -- Logging (must be configured BEFORE importing project modules) -------------

log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join(log_dir, "app.log"), encoding="utf-8"
)
stream_handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, stream_handler],
)
logger = logging.getLogger(__name__)

# -- Load environment & project modules ----------------------------------------

from dotenv import load_dotenv
load_dotenv(os.path.join("..", ".env"))

from ingestion import ingest
from vector_store import build_vector_store, load_vector_store
from retriever import retrieve_with_scores
from qa_chain import get_qa_chain, rerank_documents, answer_with_context
from fallback import check_fallback, get_fallback_response
from batch_questions import (
    answer_questions_from_pdf,
    DEFAULT_QUESTION_PDF,
)

# -- Colors for terminal output ------------------------------------------------

class C:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ==============================================================================
# MODE 1: INGEST -- Process PDF -> Build Vector Database
# ==============================================================================

def run_ingest():
    """
    Full ingestion pipeline:
    1. Read source PDF file(s) and split into domain chunks
    2. Embed all chunks using Azure OpenAI embeddings
    3. Store embeddings in Chroma database (data/vectordb/)

    Run this ONCE before using the chat mode.
    You only need to re-run if source PDF files change.
    """
    print(f"\n{C.BOLD}{'=' * 65}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  RAG RETRIEVAL AGENT -- INGESTION{C.RESET}")
    print(f"{C.BOLD}{'=' * 65}{C.RESET}\n")

    # Step 1: Ingest source PDFs into chunks
    print(f"{C.YELLOW}[1/2] Ingesting source PDF files and creating chunks...{C.RESET}")
    chunks = ingest()
    source_docs = sorted({c.get("source_doc", "?") for c in chunks})
    print(f"{C.GREEN}  [OK] Created {len(chunks)} chunks from {len(source_docs)} PDF file(s){C.RESET}\n")

    # Step 2: Build vector database
    print(f"{C.YELLOW}[2/2] Building Chroma vector database...{C.RESET}")
    print(f"  (This embeds all chunks -- may take a minute on first run)")
    vectordb = build_vector_store(chunks)
    print(f"{C.GREEN}  [OK] Vector database built at data/vectordb/{C.RESET}\n")

    # Summary
    print(f"{C.BOLD}{'=' * 65}{C.RESET}")
    print(f"{C.GREEN}  INGESTION COMPLETE{C.RESET}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Source PDFs: {len(source_docs)}")
    print(f"  Database: data/vectordb/")
    print()
    print(f"  Domains found:")
    domains = set(c["domain"] for c in chunks)
    for d in sorted(domains):
        count = sum(1 for c in chunks if c["domain"] == d)
        print(f"    - {d} ({count} chunks)")
    print(f"\n{C.BOLD}{'=' * 65}{C.RESET}")
    print(f"\n  Next: Run {C.CYAN}python app.py chat{C.RESET} to start asking questions.\n")


# ==============================================================================
# MODE 2: CHAT -- Interactive Q&A
# ==============================================================================

def run_chat():
    """
    Interactive chat loop:
    1. User types a question
    2. Retriever searches Chroma for relevant chunks
    3. If chunks found -> LLM generates answer from context
    4. If no chunks -> fallback "I don't know" (no LLM call)
    """
    print(f"\n{C.BOLD}{'=' * 65}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  RAG RETRIEVAL AGENT -- CHAT MODE{C.RESET}")
    print(f"{C.BOLD}{'=' * 65}{C.RESET}")
    print(f"{C.YELLOW}  Initializing...{C.RESET}")

    # Initialize QA chain (loads LLM + retriever)
    try:
        qa_chain = get_qa_chain()
    except EnvironmentError as e:
        print(f"{C.RED}{e}{C.RESET}")
        sys.exit(1)

    print(f"{C.GREEN}  [OK] Agent ready!{C.RESET}")
    print(f"\n  Type your question, or 'quit' to exit.\n")
    print(f"{'-' * 65}")

    while True:
        try:
            question = input(f"\n{C.BOLD}{C.CYAN}You: {C.RESET}").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print(f"\n{C.CYAN}Goodbye!{C.RESET}\n")
                break

            # Step 1: Retrieve relevant chunks with scores
            results = retrieve_with_scores(question)

            # Step 2: Check fallback
            docs = [doc for doc, score in results]
            if check_fallback(docs):
                fallback = get_fallback_response()
                print(f"\n{C.BOLD}{C.CYAN}Agent:{C.RESET}")
                print(f"{C.YELLOW}{fallback['answer']}{C.RESET}")
                logger.info(
                    f"FALLBACK | Q: '{question}' | No relevant docs found"
                )
                continue

            # Step 2.5: LLM-based re-ranking for better accuracy
            meta = getattr(qa_chain, "metadata", {})
            if meta.get("enable_reranking") and results:
                print(f"\n{C.YELLOW}  Re-ranking chunks for accuracy...{C.RESET}")
                reranked = rerank_documents(
                    meta["llm"], question, results
                )
                if reranked:
                    top_sections = [
                        r[0].metadata.get("section_title", "?") for r in reranked[:3]
                    ]
                    print(f"{C.GREEN}  Top matches: {', '.join(top_sections)}{C.RESET}")
                    # Use reranked docs for final answer generation.
                    docs = [r[0] for r in reranked[:8]]

            # Step 3: Use QA chain (LLM) to generate answer
            print(f"\n{C.YELLOW}  Generating answer...{C.RESET}")
            if docs:
                answer_text = answer_with_context(
                    meta["llm"],
                    question,
                    docs,
                )
                source_docs = docs
            else:
                response = qa_chain.invoke({"query": question})
                answer_text = response.get("result", "").strip()
                source_docs = response.get("source_documents", [])

            # Display answer
            print(f"\n{C.BOLD}{C.CYAN}Agent:{C.RESET}")
            print(f"  {answer_text}")

            # Display sources
            if source_docs:
                print(f"\n{C.BOLD}  Sources:{C.RESET}")
                for doc in source_docs:
                    meta = doc.metadata
                    print(
                        f"    - [{meta.get('section_id', '?')}] "
                        f"{meta.get('section_title', '?')} "
                        f"({meta.get('domain', '?')}) "
                        f"-- pages {meta.get('page_start', '?')}-{meta.get('page_end', '?')}"
                    )

            # Log
            logger.info(
                f"ANSWERED | Q: '{question}' | "
                f"Sources: {len(source_docs)} | "
                f"Answer: '{answer_text[:100]}...'"
            )

        except KeyboardInterrupt:
            print(f"\n\n{C.CYAN}Interrupted. Goodbye!{C.RESET}\n")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n{C.RED}Error: {e}{C.RESET}")


# ==============================================================================
# MODE 3: BATCH -- Answer Questions Extracted From a PDF
# ==============================================================================

def run_batch(question_pdf_path: str = DEFAULT_QUESTION_PDF, limit: int = 0):
    """
    Extract question lines from a PDF and answer them using the same generic
    retrieval pipeline used by chat mode.
    """
    print(f"\n{C.BOLD}{'=' * 65}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  RAG RETRIEVAL AGENT -- BATCH QA MODE{C.RESET}")
    print(f"{C.BOLD}{'=' * 65}{C.RESET}\n")

    print(f"{C.YELLOW}  Source questions: {question_pdf_path}{C.RESET}")
    if limit > 0:
        print(f"{C.YELLOW}  Limit: first {limit} questions{C.RESET}")

    try:
        out_path, count = answer_questions_from_pdf(
            pdf_path=question_pdf_path,
            limit=limit if limit > 0 else None,
        )
    except Exception as e:
        logger.error(f"Batch QA failed: {e}", exc_info=True)
        print(f"\n{C.RED}Error: {e}{C.RESET}")
        sys.exit(1)

    print(f"\n{C.GREEN}  [OK] Answered {count} questions{C.RESET}")
    print(f"  Output: {out_path}")
    print(f"\n{C.BOLD}{'=' * 65}{C.RESET}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    if len(sys.argv) < 2:
        print(f"""
{C.BOLD}RAG Retrieval Agent -- LMS Validation{C.RESET}

Usage:
  python app.py ingest    Process PDF and build vector database (run first)
  python app.py chat      Start interactive Q&A chat
    python app.py batch [question_pdf] [limit]

Example:
  python app.py ingest    <-- Run this first
    python app.py chat      <-- Then ask questions
    python app.py batch     <-- Answer all questions in default PDF source
""")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "ingest":
        run_ingest()
    elif command == "chat":
        run_chat()
    elif command == "batch":
        q_pdf = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_QUESTION_PDF
        limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
        run_batch(q_pdf, limit)
    else:
        print(f"{C.RED}Unknown command: '{command}'. Use 'ingest', 'chat', or 'batch'.{C.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
