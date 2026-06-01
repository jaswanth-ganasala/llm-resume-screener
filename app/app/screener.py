import os
import io
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import re


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def build_rag_context(resume_text: str, job_description: str) -> str:
    """
    Use RAG to retrieve most relevant chunks from resume
    based on the job description query.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(resume_text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Retrieve top 6 most relevant chunks
    docs = vectorstore.similarity_search(job_description, k=6)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def screen_resume(pdf_bytes: bytes, job_description: str) -> dict:
    """
    Main screening function.
    Extracts resume text, builds RAG context, and uses GPT-4 to score the match.
    """
    # Step 1: Extract text
    resume_text = extract_text_from_pdf(pdf_bytes)
    if not resume_text:
        raise ValueError("Could not extract text from the uploaded PDF.")

    # Step 2: RAG - retrieve relevant resume sections
    rag_context = build_rag_context(resume_text, job_description)

    # Step 3: LLM scoring
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    system_prompt = """You are an expert technical recruiter and AI engineer.
Your job is to analyze a candidate's resume against a job description and provide a detailed match analysis.

You must respond ONLY with a valid JSON object — no preamble, no explanation, no markdown.

The JSON must follow this exact structure:
{
  "match_score": <integer 0-100>,
  "verdict": "<Strong Match | Good Match | Partial Match | Weak Match>",
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "gaps": ["<gap 1>", "<gap 2>", "<gap 3>"],
  "missing_skills": ["<skill 1>", "<skill 2>", "<skill 3>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"],
  "keyword_matches": ["<matched keyword 1>", "<matched keyword 2>", "<matched keyword 3>", "<matched keyword 4>", "<matched keyword 5>"]
}"""

    user_prompt = f"""JOB DESCRIPTION:
{job_description}

RELEVANT RESUME SECTIONS (retrieved via RAG):
{rag_context}

Analyze the candidate's fit for this role and return the JSON analysis."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    raw = response.content.strip()

    # Clean any accidental markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()

    result = json.loads(raw)
    return result
