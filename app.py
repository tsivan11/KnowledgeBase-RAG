"""
FastAPI Web Interface for Multi-Domain RAG System
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
from loaders import load_file

load_dotenv()

app = FastAPI(title="KnowledgeBase RAG", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base paths
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "kb"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
KB_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Models
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 8


class QueryRequest(BaseModel):
    domain: str
    question: str
    conversation_history: List[dict] = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class DomainInfo(BaseModel):
    name: str
    file_count: int
    indexed: bool
    last_updated: Optional[str] = None


# Helper functions
def validate_domain_name(domain_name: str) -> bool:
    """Validate domain name to prevent path traversal attacks."""
    import re
    # Allow only alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', domain_name):
        return False
    # Prevent path traversal
    if '..' in domain_name or '/' in domain_name or '\\' in domain_name:
        return False
    return True


def validate_file_size(file_size: int, max_mb: int = 100) -> bool:
    """Validate file size is within limits."""
    return file_size <= max_mb * 1024 * 1024


def get_domains() -> List[str]:
    """Get list of all domains."""
    if not KB_DIR.exists():
        return []
    return [d.name for d in KB_DIR.iterdir() if d.is_dir()]


def get_domain_files(domain: str) -> List[Path]:
    """Get all files in a domain."""
    domain_path = KB_DIR / domain
    if not domain_path.exists():
        return []
    
    supported = {".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".csv", 
                 ".xlsx", ".xls", ".pptx", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
                 ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    return [f for f in domain_path.glob("**/*") if f.is_file() and f.suffix.lower() in supported]


def domain_is_indexed(domain: str) -> bool:
    """Check if domain has been indexed."""
    index_path = DATA_DIR / domain / "faiss.index"
    meta_path = DATA_DIR / domain / "chunks_meta.json"
    return index_path.exists() and meta_path.exists()


def process_domain(domain: str):
    """Run the full processing pipeline for a domain."""
    try:
        # Run ingestion
        subprocess.run([
            sys.executable, "src/ingest_pdfs.py", "--domain", domain
        ], check=True, cwd=BASE_DIR)
        
        # Run chunking
        subprocess.run([
            sys.executable, "src/chunk_pages.py", "--domain", domain
        ], check=True, cwd=BASE_DIR)
        
        # Build index
        subprocess.run([
            sys.executable, "src/build_index.py", "--domain", domain
        ], check=True, cwd=BASE_DIR)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Processing failed: {e}")
        return False


def classify_intent(client: OpenAI, question: str, conversation_history: List[dict]) -> str:
    """Classify if the question needs RAG or can be answered conversationally."""
    prompt = (
        f"User's message: \"{question}\"\n\n"
        "Classify this message into ONE of these categories:\n\n"
        "- greeting: Hello, hi, hey, good morning, etc.\n"
        "- thanks: Thank you, thanks, appreciate it, etc.\n"
        "- chitchat: General conversation not requiring document lookup\n"
        "- meta: ONLY questions asking what files/documents exist (e.g., 'what files do you have?', 'list the documents', 'show me all PDFs')\n"
        "- document_query: Everything else - questions about content, summaries, reformatting, follow-ups, etc.\n\n"
        "CRITICAL RULES:\n"
        "- If it asks about CONTENT (even reformatting/summarizing), it's document_query\n"
        "- If it asks about WHAT FILES EXIST, it's meta\n"
        "- When in doubt, choose document_query\n\n"
        "Output ONLY one word: greeting, thanks, chitchat, meta, or document_query"
    )
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    intent = resp.choices[0].message.content.strip().lower()
    print(f"[Intent Classification] '{question}' -> {intent}")
    return intent


def handle_conversational(client: OpenAI, question: str, domain: str, conversation_history: List[dict]) -> str:
    """Handle non-document questions conversationally."""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Build conversation context
    system_prompt = (
        f"You are a helpful, friendly assistant working with the '{domain}' knowledge domain. "
        f"Today's date is {current_date}. "
        f"Be warm and conversational, but NEVER use emojis. "
        f"You can answer questions about what domain you're in, "
        f"but for specific factual questions about {domain}, direct users to ask document-based questions."
    )
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent history
    for exchange in conversation_history[-3:]:
        messages.append({"role": "user", "content": exchange.get('question', '')})
        messages.append({"role": "assistant", "content": exchange.get('answer', '')})
    
    messages.append({"role": "user", "content": question})
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.7,
        messages=messages
    )
    
    return resp.choices[0].message.content


def handle_meta_query(domain: str) -> str:
    """Handle questions about what documents are available."""
    domain_path = KB_DIR / domain
    
    if not domain_path.exists():
        return f"The '{domain}' domain exists but has no documents yet."
    
    # Get all files
    files = get_domain_files(domain)
    
    if not files:
        return f"The '{domain}' domain exists but has no documents yet."
    
    # Group by file type
    file_types = {}
    for f in files:
        ext = f.suffix.lower()
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(f.name)
    
    # Build response
    response = f"I have access to {len(files)} document(s) in the '{domain}' domain:\n\n"
    
    for ext, names in sorted(file_types.items()):
        response += f"{ext.upper()} files ({len(names)}):\n"
        for name in sorted(names)[:10]:  # Limit to 10 per type
            response += f"  - {name}\n"
        if len(names) > 10:
            response += f"  ... and {len(names) - 10} more\n"
        response += "\n"
    
    response += "Feel free to ask me anything about these documents!"
    return response


def reformulate_query(client: OpenAI, question: str, conversation_history: List[dict]) -> str:
    """Reformulate follow-up questions to include context from conversation history."""
    if not conversation_history:
        return question
    
    # Build context from last 2 exchanges
    context_items = []
    for exchange in conversation_history[-2:]:
        q = exchange.get('question', '')
        a = exchange.get('answer', '')
        if q and a:
            context_items.append(f"Q: {q}\nA: {a}")
    
    if not context_items:
        return question
    
    context = "\n\n".join(context_items)
    
    prompt = (
        f"CONVERSATION HISTORY:\n{context}\n\n"
        f"NEW QUESTION: {question}\n\n"
        "If the new question contains pronouns (it, that, this, they, etc.) or references "
        "to something mentioned in the conversation history, rewrite it as a standalone question. "
        "Replace ALL pronouns and implicit references with the actual entities from the history.\n\n"
        "If the question is already standalone, return it unchanged.\n\n"
        "Output ONLY the reformulated question, nothing else."
    )
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    reformulated = resp.choices[0].message.content.strip()
    
    if reformulated != question:
        print(f"[Query Reformulation] '{question}' -> '{reformulated}'")
    
    return reformulated


def query_domain(domain: str, question: str, conversation_history: List[dict] = None) -> dict:
    """Query a domain and return answer with sources."""
    if conversation_history is None:
        conversation_history = []
    
    client = OpenAI()
    
    # Classify intent - does this need RAG?
    intent = classify_intent(client, question, conversation_history)
    
    if intent == 'meta':
        print(f"[DEBUG] Meta query - listing documents")
        answer = handle_meta_query(domain)
        return {"answer": answer, "sources": []}
    
    if intent in ['greeting', 'thanks', 'chitchat']:
        print(f"[DEBUG] Non-document query detected - responding conversationally")
        answer = handle_conversational(client, question, domain, conversation_history)
        return {"answer": answer, "sources": []}
    
    # Document query - reformulate if needed
    reformulated_question = reformulate_query(client, question, conversation_history)
    
    # Document query - proceed with RAG
    index_path = DATA_DIR / domain / "faiss.index"
    meta_path = DATA_DIR / domain / "chunks_meta.json"
    
    if not index_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not indexed")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    # Load index and metadata
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # Embed query (use reformulated question)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[reformulated_question])
    qvec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qvec)
    
    # Search
    scores, idxs = index.search(qvec, TOP_K)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()
    
    hits = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0:
            continue
        rec = meta[i]
        hits.append((rank, s, rec))
    
    # Build context
    context_blocks = []
    sources = []
    for rank, score, rec in hits:
        source_info = f"{rec['source']} ({rec.get('source_type', 'unknown')})"
        if rec.get('page'):
            source_info += f" p.{rec['page']}"
        if rec.get('section'):
            source_info += f" s.{rec['section']}"
        
        context_blocks.append(f"[{rank}] {source_info}\n{rec['text']}")
        sources.append({
            "rank": rank,
            "source": rec['source'],
            "source_type": rec.get('source_type', 'unknown'),
            "page": rec.get('page'),
            "section": rec.get('section'),
            "score": float(score),
            "text": rec['text']
        })
    
    context = "\n\n".join(context_blocks)
    
    # Generate answer
    system = (
        "You are a strict, citation-bound assistant. "
        "Answer ONLY using the provided context blocks. If the context is insufficient, "
        "say: 'I don't know based on the provided documents.' "
        "Cite sources using bracket numbers like [1], [2]."
    )
    
    # Build conversation history for context (if available)
    history_text = ""
    if conversation_history:
        history_items = []
        for exchange in conversation_history[-3:]:  # Last 3 exchanges
            q = exchange.get('question', '')
            a = exchange.get('answer', '')
            if q and a:
                history_items.append(f"Q: {q}\nA: {a}")
        if history_items:
            history_text = "PREVIOUS CONVERSATION:\n" + "\n\n".join(history_items) + "\n\n"
    
    user = (
        f"{history_text}"
        f"CONTEXT BLOCKS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Write a concise answer with citations. If the question asks to reformat or restructure "
        "a previous answer, use the previous conversation to understand what to reformat."
    )
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    
    answer = resp.choices[0].message.content
    
    return {"answer": answer, "sources": sources}


# API Routes
@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/domains", response_model=List[DomainInfo])
async def list_domains():
    """List all available domains."""
    domains = get_domains()
    result = []
    
    for domain in domains:
        files = get_domain_files(domain)
        indexed = domain_is_indexed(domain)
        
        # Get last updated time from index file if it exists
        index_path = DATA_DIR / domain / "faiss.index"
        last_updated = None
        if index_path.exists():
            mtime = index_path.stat().st_mtime
            last_updated = datetime.fromtimestamp(mtime).isoformat()
        
        result.append(DomainInfo(
            name=domain,
            file_count=len(files),
            indexed=indexed,
            last_updated=last_updated
        ))
    
    return result


@app.post("/api/domains/{domain_name}")
async def create_domain(domain_name: str):
    """Create a new domain."""
    if not validate_domain_name(domain_name):
        raise HTTPException(
            status_code=400, 
            detail="Invalid domain name. Use only letters, numbers, underscores, and hyphens."
        )
    
    domain_path = KB_DIR / domain_name
    
    if domain_path.exists():
        raise HTTPException(status_code=400, detail="Domain already exists")
    
    domain_path.mkdir(parents=True)
    return {"message": f"Domain '{domain_name}' created successfully"}


@app.post("/api/upload/{domain_name}")
async def upload_files(
    domain_name: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload files to a domain and trigger processing."""
    if not validate_domain_name(domain_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid domain name. Use only letters, numbers, underscores, and hyphens."
        )
    
    domain_path = KB_DIR / domain_name
    
    if not domain_path.exists():
        domain_path.mkdir(parents=True)
    
    uploaded_files = []
    for file in files:
        # Validate filename
        if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filename: {file.filename}"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size (100MB limit)
        if not validate_file_size(len(content)):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds 100MB limit"
            )
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        allowed_extensions = {
            ".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".csv",
            ".xlsx", ".xls", ".pptx", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
            ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"
        }
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}"
            )
        
        file_path = domain_path / file.filename
        
        with file_path.open("wb") as f:
            f.write(content)
        
        uploaded_files.append(file.filename)
    
    # Trigger background processing immediately
    background_tasks.add_task(process_domain, domain_name)
    
    return {
        "message": f"Uploaded {len(uploaded_files)} file(s) to '{domain_name}'",
        "files": uploaded_files,
        "processing": "started"
    }


@app.post("/api/process/{domain_name}")
async def trigger_processing(domain_name: str, background_tasks: BackgroundTasks):
    """Manually trigger processing for a domain."""
    if not validate_domain_name(domain_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid domain name"
        )
    
    domain_path = KB_DIR / domain_name
    
    if not domain_path.exists():
        raise HTTPException(status_code=404, detail="Domain not found")
    
    background_tasks.add_task(process_domain, domain_name)
    
    return {"message": f"Processing started for '{domain_name}'"}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query a domain with a question."""
    result = query_domain(request.domain, request.question, request.conversation_history)
    return QueryResponse(**result)


@app.delete("/api/domains/{domain_name}")
async def delete_domain(domain_name: str):
    """Delete a domain and its data."""
    if not validate_domain_name(domain_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid domain name"
        )
    
    domain_path = KB_DIR / domain_name
    data_path = DATA_DIR / domain_name
    
    if domain_path.exists():
        shutil.rmtree(domain_path)
    if data_path.exists():
        shutil.rmtree(data_path)
    
    return {"message": f"Domain '{domain_name}' deleted successfully"}


@app.get("/api/documents/{domain}/{filepath:path}")
async def serve_document(domain: str, filepath: str):
    """Serve a document file from a domain's kb folder."""
    from urllib.parse import unquote
    
    # URL decode the filepath
    filepath = unquote(filepath)
    
    # Security: prevent path traversal
    domain = domain.replace("..", "").replace("/", "").replace("\\", "")
    
    file_path = KB_DIR / domain / filepath
    
    # If file doesn't exist at direct path, search recursively in subdirectories
    if not file_path.exists():
        domain_dir = KB_DIR / domain
        if domain_dir.exists():
            # Search for the file recursively
            for found_file in domain_dir.glob(f"**/{filepath}"):
                if found_file.is_file():
                    file_path = found_file
                    break
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Document not found: {filepath}")
    
    # Security check: ensure file is within domain folder
    try:
        file_path.resolve().relative_to((KB_DIR / domain).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Determine media type - force common types
    extension = file_path.suffix.lower()
    media_type_map = {
        '.txt': 'text/plain',
        '.md': 'text/plain',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
    }
    
    media_type = media_type_map.get(extension)
    if not media_type:
        import mimetypes
        media_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    
    # Read file content
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Return response without Content-Disposition for viewable files
    return Response(
        content=content,
        media_type=media_type
    )



@app.get("/api/open-file/{domain}/{filepath:path}")
async def open_local_file(domain: str, filepath: str):
    """Open a file from the kb folder using the system's default application."""
    from urllib.parse import unquote
    import subprocess
    import platform
    
    # URL decode the filepath
    filepath = unquote(filepath)
    
    # Security: prevent path traversal
    domain = domain.replace("..", "").replace("/", "").replace("\\", "")
    
    file_path = KB_DIR / domain / filepath
    
    # If file doesn't exist at direct path, search recursively
    if not file_path.exists():
        domain_dir = KB_DIR / domain
        if domain_dir.exists():
            for found_file in domain_dir.glob(f"**/{filepath}"):
                if found_file.is_file():
                    file_path = found_file
                    break
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Document not found: {filepath}")
    
    # Security check
    try:
        file_path.resolve().relative_to((KB_DIR / domain).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Open file with default application
    try:
        if platform.system() == 'Windows':
            os.startfile(str(file_path))
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(file_path)])
        else:  # Linux
            subprocess.run(['xdg-open', str(file_path)])
        
        return {"message": f"Opened {file_path.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error opening file: {str(e)}")


# Mount static files last
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    
    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    
    print("Starting KnowledgeBase RAG Server...")
    print("Navigate to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
