"""
FastAPI server for Sanskrity RAG system - FULLY LOCAL version.
No external database dependencies. Uses in-memory storage for user contexts.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import io
import tempfile
from pathlib import Path
from datetime import datetime
import PyPDF2
from docx import Document
try:
    import whisper
except ImportError:
    whisper = None

from config import Config
from logger import get_logger
from models import setup_local_llm, load_embedding_model
from rag_chain import initialize_vector_store, get_retriever
from data_loader import load_and_chunk_itihasa_data
from utils import keep_sanskrit_english_chars, assert_english_or_sanskrit_only

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sanskrity RAG API",
    description="Sanskrit Scripture RAG system - Fully Local",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# IN-MEMORY STORAGE FOR USER CONTEXTS
# ============================================================================

class LocalContextStore:
    """In-memory storage for user-uploaded contexts."""
    
    def __init__(self):
        self.contexts: Dict[str, Dict[str, Any]] = {}  # filename -> context data
        self.chunks: List[Dict[str, Any]] = []  # All chunks with embeddings
    
    def add_context(self, filename: str, file_type: str, chunks: List[str], 
                    embeddings: List[List[float]]) -> Dict[str, Any]:
        """Store a new context with chunks and embeddings."""
        context_data = {
            "filename": filename,
            "file_type": file_type,
            "upload_date": datetime.utcnow().isoformat(),
            "chunk_count": len(chunks),
        }
        self.contexts[filename] = context_data
        
        # Store chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.chunks.append({
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding
            })
        
        return context_data
    
    def list_contexts(self) -> List[Dict[str, Any]]:
        """List all stored contexts."""
        return list(self.contexts.values())
    
    def delete_context(self, filename: str) -> int:
        """Delete a context and its chunks."""
        if filename not in self.contexts:
            return 0
        
        # Count and remove chunks
        original_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c["filename"] != filename]
        deleted_count = original_count - len(self.chunks)
        
        # Remove context
        del self.contexts[filename]
        
        return deleted_count
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity."""
        import numpy as np
        
        if not self.chunks:
            return []
        
        query_vec = np.array(query_embedding)
        
        results = []
        for chunk in self.chunks:
            chunk_vec = np.array(chunk["embedding"])
            # Cosine similarity
            similarity = np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
            )
            results.append({
                "text": chunk["text"],
                "filename": chunk["filename"],
                "score": float(similarity)
            })
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


# Global state
context_store = LocalContextStore()
llm = None
embedding_model = None
retriever = None
whisper_model = None
demo_vectorizer = None
demo_matrix = None
demo_rows: List[Dict[str, Any]] = []


# Pydantic models
class TextContextRequest(BaseModel):
    text: str
    title: str = "Untitled"


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    shloka: Optional[str] = None
    citation: Optional[str] = None
    translation: Optional[str] = None
    scripture_context: List[dict]
    user_context: List[dict]
    processing_time_ms: float


# ============================================================================
# TEXT PROCESSING HELPERS
# ============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes."""
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file bytes."""
    try:
        docx_file = io.BytesIO(file_bytes)
        doc = Document(docx_file)
        
        text_parts = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + len(chunk)
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start <= end - chunk_size:
            start = end
    
    return chunks


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global llm, embedding_model, retriever, whisper_model, demo_vectorizer, demo_matrix, demo_rows
    
    logger.info("=" * 70)
    logger.info("STARTING SANSKRITY RAG SERVER (LOCAL MODE)")
    logger.info("=" * 70)
    
    try:
        # --------------------------------------------------------------------
        # DEMO MODE: offline retrieval-only (no HuggingFace downloads)
        # --------------------------------------------------------------------
        if Config.DEMO_MODE:
            logger.info("DEMO_MODE enabled: starting offline retrieval-only server.")

            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            df = pd.read_csv(Config.DATA_PATH)
            # Ensure required columns exist (repo uses: book, chapter, shloka, sn, en)
            for col in ["sn", "en", "book", "chapter", "shloka"]:
                if col not in df.columns:
                    raise RuntimeError(f"DEMO_MODE requires column '{col}' in {Config.DATA_PATH}")

            df["sn"] = df["sn"].fillna("").astype(str)
            df["en"] = df["en"].fillna("").astype(str)
            df["book"] = df["book"].fillna("").astype(str)
            df["chapter"] = df["chapter"].fillna("").astype(str)
            df["shloka"] = df["shloka"].fillna("").astype(str)

            demo_rows = df.to_dict(orient="records")
            demo_texts = [(r["sn"] + " " + r["en"]).strip() for r in demo_rows]

            # Character n-grams work better across mixed scripts/transliterations.
            demo_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
            )
            demo_matrix = demo_vectorizer.fit_transform(demo_texts)

            # Offline demo: no LLM explanation, no uploads (no embeddings).
            llm = None
            embedding_model = None
            retriever = None
            whisper_model = None

            logger.info("✓ DEMO retrieval index built (TF-IDF over demo CSV).")
            logger.info(f"✓ SERVER READY ON http://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
            logger.info("=" * 70)
            return

        # Load Whisper model for STT (optional — missing on some Mac setups)
        if whisper is None:
            logger.warning("openai-whisper is not installed; voice input disabled")
            whisper_model = None
        else:
            try:
                logger.info("Loading Whisper STT model (tiny)...")
                whisper_model = whisper.load_model("tiny")
                logger.info("✓ Whisper STT model loaded")
            except Exception as whisper_error:
                logger.warning(f"Whisper skipped: {whisper_error}")
                whisper_model = None
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_model = load_embedding_model(Config.EMBEDDINGS_MODEL)
        logger.info("✓ Embedding model loaded")

        # Initialize Local FAISS Vector Store
        logger.info("Initializing Local FAISS Vector Store...")
        if not os.path.exists(Config.FAISS_INDEX_PATH):
            logger.info(f"Index not found at {Config.FAISS_INDEX_PATH}. Loading CSV data...")
            chunks = load_and_chunk_itihasa_data(Config.DATA_PATH)
            vector_store = initialize_vector_store(chunks)
        else:
            logger.info(f"Loading existing index from {Config.FAISS_INDEX_PATH}...")
            vector_store = initialize_vector_store()
            
        if vector_store:
            retriever = get_retriever(vector_store)
            logger.info("✓ Local FAISS Retriever initialized")
        else:
            logger.error("Failed to initialize FAISS Vector Store")
        
        # Load LLM
        logger.info("Loading LLM...")
        llm = setup_local_llm()
        logger.info("✓ LLM loaded")
        
        logger.info("=" * 70)
        logger.info(f"✓ SERVER READY ON http://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Server shutting down...")


# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    
    if frontend_path.exists():
        return frontend_path.read_text()
    else:
        return """
        <html>
            <head><title>Sanskrity RAG</title></head>
            <body>
                <h1>Sanskrity RAG API</h1>
                <p>Frontend not found. API is running at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """


# Mount static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ============================================================================
# CONTEXT MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/context/upload")
async def upload_context_file(file: UploadFile = File(...)):
    """Upload a PDF, DOCX, or TXT file and store locally."""
    try:
        if Config.DEMO_MODE:
            raise HTTPException(status_code=503, detail="Context uploads disabled in demo mode")

        # Validate file type
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
            )
        
        # Read file content
        file_bytes = await file.read()
        
        # Check file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {Config.MAX_FILE_SIZE_MB}MB"
            )
        
        logger.info(f"Received file upload: {file.filename} ({file_ext}, {file_size_mb:.2f}MB)")
        
        # Extract text
        if file_ext == "pdf":
            text = extract_text_from_pdf(file_bytes)
        elif file_ext == "docx":
            text = extract_text_from_docx(file_bytes)
        elif file_ext == "txt":
            text = file_bytes.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Extracted text is too short or empty")
        
        # Chunk and embed
        chunks = chunk_text(text)
        embeddings = embedding_model.embed_documents(chunks)
        
        # Store locally
        result = context_store.add_context(file.filename, file_ext, chunks, embeddings)
        
        return {
            "success": True,
            "message": "File processed successfully",
            "chunks_stored": result["chunk_count"],
            "filename": result["filename"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/context/text")
async def add_text_context(request: TextContextRequest):
    """Add text context directly."""
    try:
        if Config.DEMO_MODE:
            raise HTTPException(status_code=503, detail="Context uploads disabled in demo mode")

        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Text is too short (minimum 10 characters)"
            )
        
        logger.info(f"Received text context: {request.title} ({len(request.text)} chars)")
        
        # Chunk and embed
        chunks = chunk_text(request.text)
        embeddings = embedding_model.embed_documents(chunks)
        
        # Store locally
        filename = f"{request.title}.txt"
        result = context_store.add_context(filename, "text", chunks, embeddings)
        
        return {
            "success": True,
            "message": "Text context stored successfully",
            "chunks_stored": result["chunk_count"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text context storage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/context/list")
async def get_context_list():
    """Get list of all uploaded context files."""
    try:
        contexts = context_store.list_contexts()
        return {
            "success": True,
            "contexts": contexts,
            "total": len(contexts)
        }
    except Exception as e:
        logger.error(f"Failed to list contexts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/context/{filename}")
async def delete_context(filename: str):
    """Delete all chunks associated with a filename."""
    try:
        deleted_count = context_store.delete_context(filename)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Context not found")
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} chunks",
            "deleted_count": deleted_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RAG QUERY ENDPOINT
# ============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a RAG query with scripture and user context."""
    import time
    start_time = time.time()
    
    try:
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query is too short (minimum 3 characters)"
            )

        try:
            assert_english_or_sanskrit_only(request.query)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        logger.info(f"Processing query: {request.query}")

        # --------------------------------------------------------------------
        # DEMO MODE: return retrieval results only (no LLM)
        # --------------------------------------------------------------------
        if Config.DEMO_MODE:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            if demo_vectorizer is None or demo_matrix is None or not demo_rows:
                raise HTTPException(status_code=500, detail="Demo index is not initialized")

            q_vec = demo_vectorizer.transform([request.query])
            sims = cosine_similarity(q_vec, demo_matrix).flatten()

            top_k = max(1, int(Config.RETRIEVER_K))
            top_idx = np.argsort(sims)[::-1][:top_k]

            scripture_context_list: List[dict] = []
            top_shloka = None
            top_citation = None
            top_translation = None

            for rank_i, idx in enumerate(top_idx):
                row = demo_rows[int(idx)]
                book = row.get("book", "")
                chapter = row.get("chapter", "")
                shloka_num = row.get("shloka", "")
                sn = row.get("sn", "")
                en = row.get("en", "")

                scripture_context_list.append({
                    "text": sn,
                    "source": "Demo CSV",
                    "english_text": en,
                    "book": book,
                    "chapter": chapter,
                    "shloka_num": shloka_num,
                    "score": float(sims[int(idx)]),
                })

                if rank_i == 0:
                    top_shloka = sn
                    top_citation = f"{book} {chapter}.{shloka_num}"
                    top_translation = keep_sanskrit_english_chars(en) if en else None

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"DEMO query processed in {processing_time:.2f}ms")

            return QueryResponse(
                query=request.query,
                answer="Demo mode: the server returned the closest retrieved verses and their English translations. Install the HuggingFace models for full LLM explanations.",
                shloka=top_shloka,
                citation=top_citation,
                translation=top_translation,
                scripture_context=scripture_context_list,
                user_context=[],
                processing_time_ms=processing_time,
            )
        
        # 1. Search Local FAISS Index (scripture)
        scripture_docs = []
        if retriever:
            scripture_docs = retriever.invoke(request.query)
        else:
            logger.warning("Local retriever not initialized!")
            
        scripture_context_list = []
        for doc in scripture_docs:
            meta = doc.metadata
            scripture_context_list.append({
                "text": doc.page_content,
                "source": meta.get("source_file", "Local CSV"),
                "english_text": meta.get("english_text", "Translation not available"),
                "book": meta.get("book", ""),
                "chapter": meta.get("chapter", ""),
                "shloka_num": meta.get("shloka_num", ""),
                "score": 0.0
            })
        
        # Extract top scripture details
        top_shloka = None
        top_citation = None
        top_translation = None
        
        if scripture_context_list:
            top_result = scripture_context_list[0]
            top_shloka = top_result["text"]
            book = top_result.get('book', 'Scripture')
            chapter = top_result.get('chapter', '?')
            shloka = top_result.get('shloka_num', '?')
            top_citation = f"{book} {chapter}.{shloka}"
            top_translation = top_result.get("english_text")
            if top_translation:
                top_translation = keep_sanskrit_english_chars(top_translation)
        
        # 2. Search user context (local in-memory)
        user_context_list = []
        if context_store.chunks:
            query_embedding = embedding_model.embed_query(request.query)
            user_results = context_store.search(query_embedding, k=3)
            user_context_list = user_results
        
        # 3. Build prompt
        scripture_text = "\n\n".join([r["text"] for r in scripture_context_list])
        user_text = "\n\n".join([r["text"] for r in user_context_list])
        
        if user_text:
            prompt_text = f"""### Human:
You are a Sanskrit scripture scholar.

SCRIPTURE CONTEXT:
{scripture_text}

ADDITIONAL USER CONTEXT:
{user_text}

QUESTION: {request.query}

Based on the scripture and additional user context provided, please provide a comprehensive summary and explanation. 
Do not just repeat the translation. Analyze the meaning and connect it to the user's question.
Provide the response in English.

### Assistant:
"""
        else:
            prompt_text = f"""### Human:
You are a Sanskrit scripture scholar.

CONTEXT:
{scripture_text}

QUESTION: {request.query}

Based on the scripture context provided, please provide a comprehensive summary and explanation.
Do not just repeat the translation. Analyze the meaning and connect it to the user's question.
Provide the response in English.

### Assistant:
"""
        
        logger.info(f"Prompt sent to LLM:\n{prompt_text[:500]}...")

        # 4. Generate LLM response
        logger.info("Generating LLM summary...")
        answer = llm.invoke(prompt_text)
        
        if isinstance(answer, dict):
            answer = answer.get('text', str(answer))
        answer = answer.strip()
        
        logger.info(f"LLM Generated Summary: {answer[:100]}...")
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Query processed in {processing_time:.2f}ms")
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            shloka=top_shloka,
            citation=top_citation,
            translation=top_translation,
            scripture_context=scripture_context_list,
            user_context=user_context_list,
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "local",
        "llm": "loaded" if llm else "not loaded",
        "embedding_model": "loaded" if embedding_model else "not loaded",
        "retriever": "ready" if retriever else "not ready",
        "whisper": "loaded" if whisper_model else "not loaded",
        "user_contexts": len(context_store.contexts)
    }


# ============================================================================
# SPEECH-TO-TEXT ENDPOINT
# ============================================================================

@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Transcribe audio to text using Whisper.
    Accepts WAV, MP3, WEBM, OGG audio files.
    """
    try:
        if whisper_model is None:
            raise HTTPException(status_code=503, detail="Whisper model not loaded")
        
        # Read audio file
        audio_bytes = await audio.read()
        
        # Get file extension
        ext = audio.filename.split('.')[-1].lower() if audio.filename else 'webm'
        if ext not in ['wav', 'mp3', 'webm', 'ogg', 'm4a', 'flac']:
            ext = 'webm'  # Default for browser recordings
        
        # Save to temp file (Whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # Transcribe
            logger.info(f"Transcribing audio ({len(audio_bytes)} bytes)...")
            result = whisper_model.transcribe(tmp_path, language='en')
            text = result.get('text', '').strip()
            logger.info(f"Transcription: {text[:100]}...")
            
            return {
                "success": True,
                "text": text,
                "language": result.get('language', 'en')
            }
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host=Config.SERVER_HOST,
        port=Config.SERVER_PORT,
        reload=True,
        log_level="info"
    )
