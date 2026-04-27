# RAG Retrieval Agent - LMS Validation

A production-ready document question-answering system that ingests multiple PDF source documents, chunks them intelligently, embeds them using Azure OpenAI, stores them in a Chroma vector database, and answers batch or interactive questions with evidence-backed retrieval and reasoning.

> For detailed setup, branching strategy, and team workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start (New Team Member)

```bash
# 1. Clone the repo
git clone <repository-url>
cd project

# 2. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate          # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up credentials
copy .env.example .env              # Windows
# cp .env.example .env              # macOS/Linux
# Edit .env with your Azure OpenAI credentials

# 5. Add your PDFs to input file/ folder, then ingest
cd src
python -B app.py ingest

# 6. Start chatting
python -B app.py chat
```

## Project Overview

This project enables:
- **Multi-source ingestion**: Process all PDFs in `input file/` folder automatically.
- **Intelligent chunking**: Dynamically detect document structure (headings, sections, domains).
- **Hybrid retrieval**: Combine semantic search, lexical matching, and ranking fusion.
- **Evidence-based answers**: Ground all responses in source document text with citations.
- **Azure OpenAI integration**: Use enterprise-grade embeddings + GPT models for quality.

## Architecture

```
┌─────────────────────┐
│  Input PDFs         │
│ (input file/)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Ingestion + Chunking               │
│  (ingestion.py)                     │
│  - Extract text from PDFs           │
│  - Auto-detect headings             │
│  - Create domain-aware chunks       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Embeddings + Vector Store          │
│  (embeddings.py, vector_store.py)   │
│  - Azure OpenAI text-embedding-3    │
│  - Chroma persistent storage        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Query Processing                   │
│  (retriever.py)                     │
│  - Hybrid retrieval (semantic +     │
│    lexical + RRF + MMR + neighbors) │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Answer Generation                  │
│  (qa_chain.py, batch_questions.py)  │
│  - Optional LLM reranking           │
│  - Azure OpenAI GPT answer          │
│  - Evidence + source citation       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Outputs            │
│ (output file/)      │
│ - batch_answers.txt │
│ - fetched_q's.txt   │
└─────────────────────┘
```

## Folder Structure

```
project/
├── .env                      # Azure OpenAI credentials + config
├── .gitignore                # Git ignore (pycache, .venv, etc.)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/                      # Source code (entry points + modules)
│   ├── __init__.py
│   ├── app.py                # CLI entry point (ingest/chat/batch)
│   ├── ingestion.py          # Multi-source PDF ingestion & chunking
│   ├── embeddings.py         # Azure OpenAI embeddings wrapper
│   ├── vector_store.py       # Chroma vector DB setup
│   ├── retriever.py          # Hybrid retrieval pipeline
│   ├── qa_chain.py           # LLM + QA + reranking logic
│   ├── batch_questions.py    # Batch Q&A processing
│   └── fallback.py           # No-context fallback responses
│
├── data/                     # Generated data (not in git)
│   ├── processed_docs/
│   │   └── chunks.json       # Cached extracted chunks
│   ├── vectordb/             # Chroma persistent DB
│   │   └── (UUID folders + chroma.sqlite3)
│   └── raw_docs/             # (Placeholder for future use)
│
├── input file/               # Source PDFs for ingestion
│   └── LMS Test Plan Sample.pdf
│
├── output file/              # Generated Q&A outputs
│   ├── batch_answers.txt     # Answers from batch mode
│   └── fetched_questions.txt # Extracted questions list
│
└── logs/                     # Runtime logs (not in git)
    ├── app.log               # Application logs
    └── ingestion.log         # Ingestion-specific logs
```

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Azure OpenAI account with GPT + embedding model deployments

### 2. Clone & Install

```bash
cd c:\Users\M.Devi\Desktop\project

# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (versions auto-resolved)
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure `.env`

Create/update `.env` in project root:
```
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo  # or your GPT deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

## Usage

### From project root, use these commands:

#### 1. Ingest Source Documents
Scan all PDFs in `input file/`, chunk them, and build vector DB.

```bash
cd src
..\.venv\Scripts\python.exe -B app.py ingest
```

**What it does:**
- Finds all `.pdf` files in `input file/`
- Extracts text page-by-page
- Auto-detects headings & section structure
- Creates intelligent chunks with metadata
- Embeds chunks with Azure OpenAI
- Stores in Chroma vector DB (`data/vectordb/`)

**Run this:**
- Once after setup
- Whenever source PDFs change
- After adding new PDFs to `input file/`

#### 2. Interactive Chat Mode
Ask questions interactively; get answers grounded in documents.

```bash
cd src
..\.venv\Scripts\python.exe -B app.py chat
```

**Example interaction:**
```
You: Who will execute the System Test?
Agent: The System Testing will be executed by Team A.

  Sources:
    - [2.1] System Testing Approach (pages 5-5)
    - [10.0] Roles and Responsibilities (pages 9-10)
```

**Type:**
- `quit`, `exit`, or `q` to exit

#### 3. Batch Mode (Answer Questions from PDF)
Extract and answer all questions from a PDF file.

```bash
cd src
..\.venv\Scripts\python.exe -B app.py batch "..\input file\Your_Questions.pdf"
```

**Output:**
- [output file/batch_answers.txt](output%20file) — Answers with evidence
- [output file/fetched_questions.txt](output%20file) — Extracted question list

## Key Features

### Multi-Source Ingestion
- Automatically discovers and processes all PDFs in `input file/`
- Maintains source document attribution in chunk metadata
- Handles naming collisions across files with unique section IDs

### Intelligent Chunking
- **Dynamic heading detection**: Identifies section structure per document (numbered, lettered, ALL CAPS)
- **Smart merging**: Combines tiny chunks into neighbors
- **Smart splitting**: Breaks oversized chunks at paragraph boundaries with overlap
- **Domain derivation**: Auto-labels sections by semantic topic

### Hybrid Retrieval
- **Semantic**: Vector similarity from embeddings
- **Lexical**: Keyword matching across chunks
- **Reciprocal Rank Fusion (RRF)**: Combines both rankings
- **Max Marginal Relevance (MMR)**: Diversifies results
- **Neighbor linking**: Expands context to related sections

### LLM-Based Reranking
- Optional second-pass ranking using Azure OpenAI
- Scores passage relevance directly to question
- Combines with semantic scores for final ranking

### Evidence-Based Answers
- Answers grounded only in retrieved documents
- "I don't know..." fallback if no relevant text
- Auto-formatting for yes/no, checkbox, and descriptive questions
- Citation of source sections with page numbers

## Configuration

### Chunking Parameters (src/ingestion.py)
```python
MIN_CHUNK_CHARS = 80           # Merge if smaller
MAX_CHUNK_CHARS = 3000         # Split if larger
OVERLAP_CHARS = 150            # Overlap in splits
```

### Retrieval Parameters (src/retriever.py)
```python
SIMILARITY_THRESHOLD = 0.2     # Minimum semantic score
TOP_K = 8                      # Chunks to retrieve
```

### LLM Parameters (src/qa_chain.py)
- Temperature: 0.0 (deterministic)
- Max tokens: 1024
- Strict prompt enforces evidence-based answers

## Dependencies

See [requirements.txt](requirements.txt):
- **LangChain**: Orchestration & chains
- **Azure OpenAI**: Embeddings + GPT
- **Chroma**: Vector database
- **PyMuPDF**: PDF text extraction
- **spaCy**: Sentence segmentation
- **python-dotenv**: .env loading

## Troubleshooting

### "No PDF source files found"
**Cause:** `input file/` folder is empty or no `.pdf` files  
**Fix:** Add at least one `.pdf` to `input file/` and rerun `ingest`

### "Azure OpenAI credentials not set"
**Cause:** `.env` file missing or incomplete  
**Fix:** Copy `.env` template, fill in real credentials

### Slow ingestion on first run
**Cause:** Azure embeddings model download (~80MB)  
**Fix:** Normal on first run; subsequent ingests are faster

### Dependency conflicts
**Cause:** Incompatible package versions  
**Fix:** Delete `.venv/`, reinstall with `pip install -r requirements.txt`

## Performance Notes

- **Ingestion**: ~30 seconds for typical 10-page document (includes embedding download on first run)
- **Chat/Batch**: ~2-5 seconds per question (Azure OpenAI latency)
- **Vector DB**: Persistent; reuse across runs after first ingest

## Example Workflow

```bash
# 1. Setup (one-time)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Add your source PDF(s)
# Copy files to: input file/

# 3. Ingest
cd src
..\.venv\Scripts\python.exe -B app.py ingest

# 4. Chat or batch
..\.venv\Scripts\python.exe -B app.py chat           # Interactive
# OR
..\.venv\Scripts\python.exe -B app.py batch          # Batch mode
```

## License & Attribution

This project uses:
- Azure OpenAI (Microsoft)
- LangChain (LangChain Inc.)
- Chroma (Chroma Inc.)
- PyMuPDF (Artifex Software)

## Questions?

Review logs in `logs/` for debug details. Check source code docstrings for component-level documentation.
