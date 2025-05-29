import os
import json
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.init import Timeout, AdditionalConfig

import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from langchain_huggingface import HuggingFaceEmbeddings
import httpx

from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

# --- MODELLI DATI ---
class RAGChunk(BaseModel):
    text: str
    source: str
    page: int
    images: List[str]

class RAGResponse(BaseModel):
    answer: str
    chunks: List[RAGChunk]
    images: List[str]

# --- CONFIGURAZIONE ---
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
#API_KEY = os.getenv("API_KEY")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
SITE_URL = os.getenv("SITE_URL", "http://localhost:3000")
SITE_NAME = os.getenv("SITE_NAME", "RAG System")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# --- FASTAPI E CORS ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connessione Weaviate v4
    connection_params = ConnectionParams.from_params(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
    )
    app.weaviate_client = weaviate.WeaviateClient(
        connection_params=connection_params,
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
    )
    app.weaviate_client.connect()
    # Redis & FastAPI Limiter
    app.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    await FastAPILimiter.init(app.redis_client)
    yield
    await app.redis_client.aclose()
    app.weaviate_client.close()

app = FastAPI(
    title="BreadBot AI",
    description="API per Retrieval-Augmented Generation come tutor allo studio di Elettronica",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# --- ENDPOINT PRINCIPALE ---
@app.get("/rag", response_model=RAGResponse, dependencies=[Depends(RateLimiter(times=100, seconds=60))])
async def rag_query(
    q: str = Query(..., min_length=3, max_length=500),
    api_key: str = Depends(api_key_header),
    k: int = Query(5, ge=1, le=10)
):
    # Verifica API key
    # if API_KEY and api_key != API_KEY:
    #     raise HTTPException(status_code=401, detail="Invalid API Key")

    # 1. Embedding della query
    query_vector = embeddings.embed_query(q)

    # 2. Hybrid search su Weaviate v4
    try:
        collection = app.weaviate_client.collections.get("Chunks")
        results = collection.query.hybrid(
            query=q,
            vector=query_vector,
            limit=k,
            return_properties=["text", "source", "page", "images"],
            alpha=0.5
        )
        chunks = [
            RAGChunk(
                text=obj.properties["text"],
                source=obj.properties["source"],
                page=obj.properties["page"],
                images=obj.properties.get("images", [])
            ) for obj in results.objects
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore Weaviate: {str(e)}")

    # 3. Estrai immagini uniche
    all_images = list(set(img for chunk in chunks for img in chunk.images))

    # 4. Generazione risposta con LLM
    context = "\n\n".join([f"[Fonte: {c.source}, Pag. {c.page}]\n{c.text}" for c in chunks])
    answer = await call_llm(context, q)

    return RAGResponse(
        answer=answer,
        chunks=chunks,
        images=all_images
    )

async def call_llm(context: str, question: str) -> str:
    """Chiama l'LLM su OpenRouter con contesto e domanda"""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore chiamata LLM: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "weaviate": app.weaviate_client.is_ready(),
        "redis": await app.redis_client.ping(),
        "version": "1.0.0"
    }
