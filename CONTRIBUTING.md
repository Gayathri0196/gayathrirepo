# Contributing Guide

## Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd project
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Copy the example file and fill in your Azure credentials
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```
Then edit `.env` with your Azure OpenAI endpoint, API key, and deployment names.  
**Never commit your `.env` file.** It is already in `.gitignore`.

### 5. Add Input Documents
Place your PDF files in the `input file/` folder.  
This folder is ignored by git — each team member manages their own local PDFs.

### 6. Run the Pipeline
```bash
cd src

# Step 1: Ingest PDFs and build vector database
python -B app.py ingest

# Step 2: Interactive chat
python -B app.py chat

# Step 3: Batch question answering from a PDF
python -B app.py batch "path/to/questions.pdf"
```

---

## Project Structure
```
project/
├── src/                     # All source code
│   ├── app.py               # Main CLI entry point
│   ├── ingestion.py         # PDF extraction and chunking
│   ├── embeddings.py        # Azure OpenAI embedding setup
│   ├── vector_store.py      # Chroma vector DB management
│   ├── retriever.py         # Hybrid retrieval logic
│   ├── qa_chain.py          # LLM answer generation
│   ├── batch_questions.py   # Batch Q&A pipeline
│   └── fallback.py          # No-context fallback responses
├── input file/              # Place source PDFs here (git-ignored)
├── output file/             # Generated answers (git-ignored)
├── data/                    # Vector DB and chunk cache (git-ignored)
├── logs/                    # Runtime logs (git-ignored)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md                # Full project documentation
```

---

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, production-ready code |
| `develop` | Integration branch for features |
| `feature/<name>` | Individual feature branches |
| `fix/<name>` | Bug fix branches |

### Workflow
1. Create a branch from `develop`:  
   `git checkout -b feature/your-feature-name develop`
2. Make changes, commit with descriptive messages.
3. Push and open a Pull Request to `develop`.
4. After review, merge to `develop`.
5. Periodically, `develop` is merged into `main` for releases.

---

## Commit Message Convention
```
<type>: <short description>

Examples:
feat: add multi-source PDF ingestion
fix: handle empty chunk list in retriever
docs: update README setup instructions
refactor: simplify QA chain prompt
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

---

## Important Rules
- **Do not commit `.env`** — contains secrets.
- **Do not commit PDFs** in `input file/` or `output file/`.
- **Do not commit `data/vectordb/`** — it is large and rebuild-able.
- Always run `python -B app.py ingest` after pulling changes to rebuild the vector DB.
- Use Python 3.10+ for compatibility.
