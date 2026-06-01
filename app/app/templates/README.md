# LLM Resume Screener & Job Match Scorer

An AI-powered resume screening tool that uses **GPT-4 + RAG (Retrieval-Augmented Generation)** to score how well a candidate's resume matches a job description.

## Features

- 📄 Upload any PDF resume
- 🔍 RAG pipeline extracts the most relevant resume sections using FAISS + OpenAI Embeddings
- 🤖 GPT-4 scores the match (0–100) with detailed reasoning
- ✅ Returns strengths, gaps, missing skills, recommendations, and keyword matches
- ⚡ FastAPI backend + clean responsive frontend

## Architecture

```
PDF Resume → Text Extraction (pypdf)
           → Chunking (LangChain RecursiveCharacterTextSplitter)
           → Embeddings (OpenAI Embeddings)
           → Vector Store (FAISS)
           → Similarity Search against Job Description
           → Top chunks → GPT-4 prompt
           → Structured JSON response
```

## Tech Stack

- **Backend:** Python, FastAPI, LangChain, LangChain-OpenAI
- **LLM:** OpenAI GPT-4
- **RAG:** FAISS vector store, OpenAI Embeddings
- **PDF Processing:** pypdf
- **Frontend:** HTML, CSS, JavaScript

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/jaswanth-ganasala/llm-resume-screener.git
cd llm-resume-screener
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 5. Run the server
```bash
uvicorn app.main:app --reload
```

### 6. Open in browser
```
http://localhost:8000
```

## Usage

1. Upload your resume as a PDF
2. Paste the job description
3. Click **Analyze Match**
4. Get your match score, strengths, gaps, and recommendations instantly

## API

### POST `/screen`

**Form Data:**
- `resume` — PDF file
- `job_description` — string

**Response:**
```json
{
  "match_score": 82,
  "verdict": "Strong Match",
  "summary": "...",
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "missing_skills": ["...", "..."],
  "recommendations": ["...", "..."],
  "keyword_matches": ["...", "..."]
}
```

## Author

**Jaswanth Kumar Ganasala** — Generative AI & ML Engineer  
[linkedin.com/in/ganasala-jaswanth-k](https://linkedin.com/in/ganasala-jaswanth-k) · [github.com/jaswanth-ganasala](https://github.com/jaswanth-ganasala)
