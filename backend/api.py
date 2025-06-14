import os
import asyncio
import logging
import time
import uuid
import numpy as np

from contextlib import asynccontextmanager
from typing import List, Optional
import logging
from fastapi import FastAPI, HTTPException, Query, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.init import Timeout, AdditionalConfig

import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import httpx

logging.basicConfig(level=logging.INFO)


# --- MODELLI PYDANTIC ---
class RAGChunk(BaseModel):
    text: str
    source: Optional[str]
    page: Optional[int]
    images: List[str] = []

class RAGImageInfo(BaseModel):
    filename: str
    url: str
    description: str

class ChatMessage(BaseModel):
    role: str  # 'user' o 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    chunks: List[RAGChunk]
    images: List[RAGImageInfo]


# --- LOGGING CONFIG ---
logging.basicConfig(level=logging.INFO)

# --- LIFESPAN HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Inizializzazione Weaviate
    try:
        logging.info("🔗 Connessione a Weaviate locale")
        app.weaviate_client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host=os.getenv("BREADBOT_WEAVIATE_HOST", "weaviate-service"),
                http_port=int(os.getenv("BREADBOT_WEAVIATE_PORT", 8080)),
                http_secure=False,
                grpc_host=os.getenv("BREADBOT_WEAVIATE_HOST", "weaviate-service"),
                grpc_port=int(os.getenv("BREADBOT_WEAVIATE_GRPC_PORT", 50051)),
                grpc_secure=False,
            ),
            additional_config=AdditionalConfig(timeout=Timeout(query=45))
        )
        app.weaviate_client.connect()
        logging.info("✅ Connessione Weaviate stabilita")

    except Exception as e:
        logging.error(f"❌ Errore connessione Weaviate: {str(e)}")
        raise RuntimeError("Impossibile connettersi a Weaviate") from e

    # Inizializzazione modelli embedding
    try:
        logging.info("🔄 Caricamento modelli di embedding")
        app.text_encoder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        app.clip_processor = CLIPProcessor.from_pretrained(os.getenv("EMBEDDING_MODEL_IMAGE", "openai/clip-vit-base-patch32"))
        app.clip_model = CLIPModel.from_pretrained(os.getenv("EMBEDDING_MODEL_IMAGE", "openai/clip-vit-base-patch16"))
    except Exception as e:
        logging.error(f"❌ Errore caricamento modelli: {str(e)}")
        raise

    # Inizializzazione Redis
    try:
        logging.info("🔗 Connessione a Redis")
        app.redis_client = redis.from_url(os.getenv("BREADBOT_REDIS_URL", "redis://redis-service:6379"))
        await FastAPILimiter.init(app.redis_client)

    except Exception as e:
        logging.error(f"❌ Errore connessione Redis: {str(e)}")
        raise

    # Inizializzazione sessioni
    app.sessions = {}
    asyncio.create_task(cleanup_sessions(app.sessions))
    yield
    logging.info("🚪 Chiusura connessioni")
    await app.redis_client.aclose()
    app.weaviate_client.close()


# --- APP FASTAPI ---
app = FastAPI(
    title="BreadBot AI API",
    description="API avanzata per sistema RAG con gestione sessioni",
    version="4.0.0",
    lifespan=lifespan
)

# --- CORS CONFIG ---
origins = [origin.strip() for origin in os.getenv("BREADBOT_ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# --- FUNZIONI DI SVILUPPO ---

# --- SESSION CLEANUP ---
async def cleanup_sessions(sessions):
    while True:
        now = time.time()
        expired = [k for k, v in sessions.items() if now - v["last_active"] > 1800]
        for k in expired:
            del sessions[k]
            logging.info(f"🧹 Sessione {k} rimossa")
        await asyncio.sleep(60)

# --- FUNZIONE DI BUILD ---
def build_context(chunks, images):
    context = "## Fonti tecniche:\n"
    context += "\n".join([f"[{chunk.properties.get('source', 'N/D')}] {chunk.properties.get('text', '')}" for chunk in chunks])
    context += "\n\n## Immagini rilevanti:\n"
    context += "\n".join([f"- {img.properties.get('description', '')} ({img.properties.get('url', '')})" for img in images])
    return context

# --- GENERA RISPOSTA ---
async def generate_answer(context: str, question: str) -> str:
    api_key = os.getenv('OPENROUTER_API_KEY')
    logging.info(f"OpenRouter API Key: {'Set' if api_key else 'Not_set'}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": os.getenv("SITE_URL", "https://breadbot-ai.com"),
                    "X-Title": "BreadBot AI"
                },
                json={
                    "model": os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free"),
                    "messages": [{
                        "role": "user",
                        "content": f"""
                    Sei un esperto di elettronica, ti occupi di assistenza allo studio delle principali materie di indirizzo delle scuole superiori italiane.
                    Devi rispondere a domande tecniche su argomenti di elettronica e realizzazione di circuiti elettronici su BreadBoard e PCB.
                    Le tue risposte devono essere basate su fonti tecniche aggiornate per usare anche i moderni sistemi di controllo ed elaborazione dati per sistemi automatici.
                    Queste includono: elettronica, elettrotecnica, sistemi automatici, tpsee e fisica.
                    Rispondi con un italiano tecnico preciso, accurato e comprensibile da tutti (bambini inclusi) usando risorse online accademiche (possibilmente non Wikipedia), ma soprattutto e prevalentemente le fonti (hanno la priorità rispetto a fonti online) del seguente contesto:

                    Contesto: {context}

                    Domanda: {question}

                    Risposta:
                    """
                    }],
                    "temperature": 0.3,
                    "max_tokens": 1300
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        logging.error(f"❌ Errore HTTP OpenRouter: {e.response.status_code} - {e.response.text}")
        return f"Errore di autenticazione con OpenRouter: {e.response.text}"
    except Exception as e:
        logging.error(f"❌ Errore generazione risposta: {str(e)}")
        return "Impossibile generare una risposta al momento."

# --- RAG QUERY CORE ---
async def execute_rag_query(query: str, k: int = 5):
    """Funzione RAG per eseguire query ibride testuali e immagini"""
    try:
        # TEXT EMBEDDING (sempre 1D)
        text_vector = app.text_encoder.encode(query)
        if isinstance(text_vector, list):
            text_vector = np.array(text_vector)
        if hasattr(text_vector, "shape") and len(text_vector.shape) == 2:
            text_vector = text_vector[0]
        text_vector = text_vector.tolist()

        # IMAGE EMBEDDING
        inputs = app.clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            image_vector = app.clip_model.get_text_features(**inputs)
        if hasattr(image_vector, "detach"):
            image_vector = image_vector.detach().cpu().numpy()
        if len(image_vector.shape) == 2:
            image_vector = image_vector[0]
        image_vector = image_vector.tolist()

        # QUERY WEAVIATE (v4)
        chunks = app.weaviate_client.collections.get("Chunks").query.hybrid(
            query=query,
            vector=text_vector,
            limit=k,
            return_properties=["text", "source", "page", "images"]
        ).objects

        images = app.weaviate_client.collections.get("Images").query.hybrid(
            query=query,
            vector=image_vector,
            limit=k,
            return_properties=["filename", "url", "description"]
        ).objects

        # LOGGING e RISULTATI ( valutare se rimuovere in produzione )
        logging.info(f"Recuperati {len(chunks)} chunks da Weaviate")
        for i, chunk in enumerate(chunks):
            logging.info(f"Chunk {i}: {chunk.properties}")
        logging.info(f"Recuperate {len(images)} immagini da Weaviate")
        for i, img in enumerate(images):
            logging.info(f"Immagine {i}: {img.properties}")

        # BUILD CONTEXT E GENERA RISPOSTA
        context = build_context(chunks, images)
        answer = await generate_answer(context, query)
        return {
            "answer": answer,
            "chunks": [RAGChunk(**obj.properties) for obj in chunks],
            "images": [RAGImageInfo(**img.properties) for img in images]
        }
    except weaviate.exceptions.WeaviateQueryException as e:
        logging.error(f"❌ Errore query Weaviate: {str(e)}")
        raise HTTPException(500, detail="Errore durante la query al database vettoriale")
    except Exception as e:
        logging.error(f"❌ Errore generico in RAG: {str(e)}")
        raise HTTPException(500, detail="Errore interno del server durante l'esecuzione della query RAG")

# --- FINE FUNZIONI DI SUPPORTO ---



# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "BreadBot AI API is running",
        "version": "4.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "weaviate": app.weaviate_client.is_ready(),
        "redis": await app.redis_client.ping(),
        "sessions_active": len(app.sessions),
        "version": "4.0.0"
    }

@app.get("/query")
async def query_get(q: str = Query(..., min_length=3, max_length=500)):
    """Endpoint GET per domande singole (senza memoria di sessione)."""
    try:
        rag_response = await execute_rag_query(q)
        return {
            "answer": rag_response["answer"],
            "chunks": [chunk.model_dump() for chunk in rag_response["chunks"]],
            "images": [img.model_dump() for img in rag_response["images"]]
        }
    except Exception as e:
        logging.error(f"❌ Errore nella GET /query: {str(e)}")
        raise HTTPException(500, detail="Errore interno del server")

@app.post("/chat", response_model=ChatResponse)
async def chat_session(
    request: ChatRequest,
    session_id: Optional[str] = Cookie(None),
    response: Response = None
    ):
    try:
        current_session = session_id if session_id in app.sessions else None
        if not current_session:
            session_id = str(uuid.uuid4())
            app.sessions[session_id] = {"messages": [], "last_active": time.time()}
        response.set_cookie("session_id", session_id, httponly=True, max_age=1800)
        app.sessions[session_id]["last_active"] = time.time()
        last_message = next((m for m in reversed(request.messages) if m.role == "user"), None)
        if not last_message:
            raise HTTPException(400, "Nessuna domanda valida nella richiesta")
        rag_response = await execute_rag_query(last_message.content)
        return ChatResponse(
            session_id=session_id,
            reply=rag_response["answer"],
            chunks=rag_response["chunks"],
            images=rag_response["images"]
        ).model_dump()
    except Exception as e:
        logging.error(f"❌ Errore durante la chat: {str(e)}")
        raise HTTPException(500, detail=str(e))

