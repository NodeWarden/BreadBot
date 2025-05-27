# backend/api.py
import os
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import weaviate
from weaviate.classes.query import Filter
from langchain_huggingface import HuggingFaceEmbeddings
import httpx
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import aioredis

# Configurazione ambiente
load_dotenv()

# Modelli di dati
class RAGChunk(BaseModel):
    text: str
    source: str
    page: int
    images: List[str]

class RAGResponse(BaseModel):
    answer: str
    chunks: List[RAGChunk]
    images: List[str]

# Inizializzazione FastAPI
app = FastAPI(
    title="BreadBot AI",
    description="API per Retrieval-Augmented Generation come tutor allo studio di Elettronica",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None
)

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"]
)

# Sicurezza API
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Configurazione Weaviate
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "http://localhost:8080")
WEAVIATE_CLOUD = os.getenv("WEAVIATE_CLOUD", "false").lower() == "true"

def get_weaviate_client():
    if WEAVIATE_CLOUD:
        return weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_HOST,
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
    else:
        return weaviate.connect_to_local(host=WEAVIATE_HOST)

# Configurazione embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)

# Configurazione LLM
LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

async def rate_limiter():
    return await RateLimiter(times=100, seconds=60)

@app.on_event("startup")
async def startup():
    # Rate limiting con Redis
    redis = await aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    await FastAPILimiter.init(redis)

@app.get("/rag", response_model=RAGResponse, dependencies=[Depends(rate_limiter)])
async def rag_query(
    q: str = Query(..., min_length=3, max_length=500),
    api_key: str = Depends(api_key_header),
    k: int = Query(5, ge=1, le=10)
):
    """Endpoint principale per query RAG"""
    # Verifica API key
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 1. Embedding della query
        query_vector = embeddings.embed_query(q)
        
        # 2. Hybrid search su Weaviate
        with get_weaviate_client() as client:
            collection = client.collections.get("Chunks")
            
            results = collection.query.hybrid(
                query=q,
                vector=query_vector,
                limit=k,
                return_properties=["text", "source", "page", "images"],
                alpha=0.5  # Bilancia tra ricerca vettoriale e testuale
            )

            chunks = [
                RAGChunk(
                    text=obj.properties["text"],
                    source=obj.properties["source"],
                    page=obj.properties["page"],
                    images=obj.properties.get("images", [])
                ) for obj in results.objects
            ]

            # 3. Estrai immagini uniche
            all_images = list(set(img for chunk in chunks for img in chunk.images))

            # 4. Generazione risposta con LLM
            context = "\n\n".join([f"[Fonte: {c.source}, Pag. {c.page}]\n{c.text}" for c in chunks])
            answer = await call_llm(context, q, api_key)

            return RAGResponse(
                answer=answer,
                chunks=chunks,
                images=all_images
            )

    except weaviate.exceptions.WeaviateBaseError as e:
        raise HTTPException(status_code=500, detail=f"Errore Weaviate: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

async def call_llm(context: str, question: str, api_key: str) -> str:
    """Chiama l'LLM su OpenRouter con contesto e domanda"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:3000"),
            "X-Title": os.getenv("SITE_NAME", "RAG System")
        }

        prompt = f"""Rispondi in italiano alla domanda usando solo il contesto fornito.
Se la risposta non è nel contesto, rispondi 'Non ho informazioni sufficienti'.

Contesto:
{context}

Domanda: {question}
Risposta:"""

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                LLM_API_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                headers=headers
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Errore LLM API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE")
    )
