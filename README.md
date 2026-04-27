# RAG Retrieval Agent - LMS Validation

This project ingests validation PDFs, builds a Chroma vector store with Azure OpenAI embeddings, and answers questions through interactive chat or batch processing.

For team workflow and contribution rules, see [CONTRIBUTING.md](CONTRIBUTING.md).

## What the project does

- Ingests one or more PDFs from `input_files/`
- Splits content into structured chunks with metadata
- Stores embeddings in a persistent Chroma vector database
- Retrieves context with a hybrid semantic + lexical pipeline
- Generates evidence-based answers with Azure OpenAI
- Tracks input tokens, output tokens, and estimated API cost

## Repository layout

```text
gayathrirepo/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .env.example
├── src/
│   ├── app.py                # CLI entry point
│   ├── ingestion.py          # PDF parsing and chunk creation
│   ├── embeddings.py         # Azure embedding client + tracking
│   ├── vector_store.py       # Chroma build/load helpers
│   ├── retriever.py          # Hybrid retrieval pipeline
│   ├── qa_chain.py           # Answering and reranking logic
│   ├── batch_questions.py    # Batch question extraction/answering
│   ├── fallback.py           # No-context fallback response
│   └── token_tracker.py      # Token and cost tracking
├── input_files/              # Source PDFs for ingestion
├── output_files/             # Batch inputs/outputs
├── data/                     # Generated chunks and vector DB
└── logs/                     # Runtime logs and token usage logs
```

## Setup

### Prerequisites

- Python 3.10+
- Azure OpenAI deployment for chat
- Azure OpenAI deployment for embeddings

### Install

```powershell
cd C:\Users\M.Devi\Desktop\gayathrirepo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment

Copy `.env.example` to `.env` and set these values:

```env
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.1
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

AZURE_CHAT_INPUT_COST_PER_1K=0.00125
AZURE_CHAT_OUTPUT_COST_PER_1K=0.011
AZURE_EMBEDDING_INPUT_COST_PER_1K=0.00125
```

## Run the project

Run commands from the `src/` directory.

### Ingest source PDFs

```powershell
cd src
..\.venv\Scripts\python.exe -B app.py ingest
```

Use this when:
- you set up the project for the first time
- you add or replace PDFs in `input_files/`
- you want to rebuild the vector database

### Interactive chat

```powershell
cd src
..\.venv\Scripts\python.exe -B app.py chat
```

### Batch question answering

```powershell
cd src
..\.venv\Scripts\python.exe -B app.py batch
```

Or provide a different question PDF:

```powershell
cd src
..\.venv\Scripts\python.exe -B app.py batch "..\output_files\Your Questions.pdf"
```

Batch outputs are written to:
- `output_files/batch_answers.txt`
- `output_files/fetched_questions.txt`

## API token and cost tracking

The project records usage for:
- embedding calls
- reranking calls
- final answer generation calls

Tracked details:
- input tokens
- output tokens
- total tokens
- overall total cost in USD

Where to find them:
- `logs/token_usage.jsonl` for per-call records
- `logs/app.log` for runtime `USAGE | ...` entries
- terminal summary after `ingest`, `chat`, and `batch`

## Main components

- `src/ingestion.py`: extracts and chunks source PDFs
- `src/vector_store.py`: creates and loads the Chroma database
- `src/retriever.py`: performs hybrid retrieval and section expansion
- `src/qa_chain.py`: reranks passages and generates answers
- `src/batch_questions.py`: extracts batch questions and writes answers
- `src/token_tracker.py`: accumulates token and cost usage across calls

## Troubleshooting

### No PDF source files found

Cause: `input_files/` does not contain any PDFs.

Fix: Add at least one PDF and run ingest again.

### Azure OpenAI configuration not complete

Cause: `.env` is missing required values.

Fix: Fill in `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME`, and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`.

### Batch answers or token logs look stale

Cause: files in `output_files/` and `logs/` are generated artifacts from previous runs.

Fix: rerun the command, or clear those generated files if you want a fresh run history.
