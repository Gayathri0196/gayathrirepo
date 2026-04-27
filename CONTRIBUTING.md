# Contributing Guide

## Repository Branch Structure

```
main                         <- Stable production code (Team Lead manages)
 └── develop                 <- Integration branch (merge all features here)
      ├── feature/member-ingestion    <- PDF ingestion & chunking work
      ├── feature/member-retriever    <- Retrieval & vector store work
      ├── feature/member-qa-chain     <- QA chain & LLM answer work
      └── feature/member-ui-batch     <- Batch pipeline & output work
```

**Team Lead** pushes the base project to GitHub, then assigns each team member their branch.  
**Each member** clones the repo, checks out their assigned branch, and works only on that branch.

---

## For the Team Lead: Push to GitHub

After receiving the project folder:

```bash
# 1. Create a new empty repository on GitHub (no README, no .gitignore)
#    Copy the repository URL e.g. https://github.com/your-org/rag-agent.git

# 2. Link local repo to GitHub and push all branches
git remote add origin https://github.com/your-org/rag-agent.git
git push -u origin main
git push origin develop
git push origin feature/member-ingestion
git push origin feature/member-retriever
git push origin feature/member-qa-chain
git push origin feature/member-ui-batch
```

Then **assign each team member their branch** on GitHub under Settings > Branches.

---

## For Team Members: Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/rag-agent.git
cd rag-agent
```

### 2. Switch to Your Assigned Branch
```bash
# Each member checks out ONLY their assigned branch
git checkout feature/member-ingestion    # if you own ingestion work
# OR
git checkout feature/member-retriever    # if you own retrieval work
# OR
git checkout feature/member-qa-chain     # if you own QA chain work
# OR
git checkout feature/member-ui-batch     # if you own batch pipeline work
```

### 3. Create a Virtual Environment
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
Place your PDF files in the `input_files/` folder.  
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
├── input_files/             # Place source PDFs here (git-ignored)
├── output_files/            # Generated answers (git-ignored)
├── data/                    # Vector DB and chunk cache (git-ignored)
├── logs/                    # Runtime logs (git-ignored)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md                # Full project documentation
```

---

## Team Member Workflow (Day-to-Day)

```bash
# 1. Always pull latest before starting work
git pull origin develop

# 2. Make your changes in src/ files

# 3. Stage and commit
git add .
git commit -m "feat: describe your change clearly"

# 4. Push your branch to GitHub
git push origin feature/member-ingestion   # use your branch name

# 5. Open a Pull Request on GitHub: your branch -> develop
#    Team Lead reviews and merges
```

> The Team Lead periodically merges `develop` into `main` when a stable version is ready.

---

## Branching Strategy

| Branch | Owner | Purpose |
|--------|-------|---------|
| `main` | Team Lead | Stable, production-ready code only |
| `develop` | Team Lead | Integration — all features merge here |
| `feature/member-ingestion` | Member 1 | `ingestion.py`, `embeddings.py`, `vector_store.py` |
| `feature/member-retriever` | Member 2 | `retriever.py` |
| `feature/member-qa-chain` | Member 3 | `qa_chain.py`, `fallback.py` |
| `feature/member-ui-batch` | Member 4 | `app.py`, `batch_questions.py` |

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
- **Do not commit PDFs** in `input_files/` or `output_files/`.
- **Do not commit `data/vectordb/`** — it is large and rebuild-able.
- Always run `python -B app.py ingest` after pulling changes to rebuild the vector DB.
- Use Python 3.10+ for compatibility.
- **Only open Pull Requests to `develop`**, never directly to `main`.
