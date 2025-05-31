# backend/api.py
import os
import json
import asyncio
import logging
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.init import Timeout, AdditionalConfig, Auth

import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import httpx

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Modelli Pydantic
class RAGChunk(BaseModel):
    text: str
    source: str
    page: int
    images: List[str]

class RAGImageInfo(BaseModel):
    filename: str
    url: str
    description: str

class RAGResponse(BaseModel):
    answer: str
    chunks: List[RAGChunk]
    images: List[RAGImageInfo]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inizializzazione Weaviate
    try:
        if os.getenv("WEAVIATE_CLOUD", "false").lower() in ("true", "1", "yes"):
            logging.info("🔗 Connessione a Weaviate Cloud")
            app.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.environ["WEAVIATE_URL"],
                grpc_url=os.environ.get("WEAVIATE_GRPC_URL"),
                grpc_port=int(os.environ.get("WEAVIATE_GRPC_CLOUD_PORT", 443)),
                auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=30, query=60, insert=120)
                )
            )
        else:
            logging.info("🔗 Connessione a Weaviate locale")
            app.weaviate_client = weaviate.WeaviateClient(
                connection_params=ConnectionParams.from_params(
                    http_host=os.getenv("WEAVIATE_HOST", "weaviate"),
                    http_port=int(os.getenv("WEAVIATE_PORT", 8080)),
                    http_secure=False,
                    grpc_host=os.getenv("WEAVIATE_HOST", "weaviate"),
                    grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", 50051)),
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
        app.clip_model = CLIPModel.from_pretrained(os.getenv("EMBEDDING_MODEL_IMAGE", "openai/clip-vit-base-patch32"))
    except Exception as e:
        logging.error(f"❌ Errore caricamento modelli: {str(e)}")
        raise

    # Inizializzazione Redis
    try:
        logging.info("🔗 Connessione a Redis")
        app.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
        await FastAPILimiter.init(app.redis_client)
    except Exception as e:
        logging.error(f"❌ Errore connessione Redis: {str(e)}")
        raise

    yield

    # Cleanup
    logging.info("🚪 Chiusura connessioni")
    await app.redis_client.aclose()
    app.weaviate_client.close()

app = FastAPI(
    title="BreadBot AI API",
    description="API per il sistema RAG di BreadBot AI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/query", response_model=RAGResponse)
async def rag_query(
    q: str = Query(..., min_length=3, max_length=500),
    k: int = Query(5, ge=1, le=10)
):
    try:
        # Embedding testo
        text_vector = app.text_encoder.encode(q).tolist()
        
        # Embedding immagine (CLIP)
        inputs = app.clip_processor(text=[q], return_tensors="pt", padding=True)
        with torch.no_grad():
            image_vector = app.clip_model.get_text_features(**inputs)[0].tolist()

        # Ricerca ibrida testuale
        chunks = app.weaviate_client.collections.get("Chunks").query.hybrid(
            query=q,
            vector=text_vector,
            limit=k,
            return_properties=["text", "source", "page", "images"]
        ).objects

        # Ricerca ibrida immagini
        images = app.weaviate_client.collections.get("Images").query.hybrid(
            query=q,
            vector=image_vector,
            limit=k,
            return_properties=["filename", "url", "description"]
        ).objects

        # Costruzione contesto
        context = build_context(chunks, images)

        # Generazione risposta
        answer = await generate_answer(context, q)

        return RAGResponse(
            answer=answer,
            chunks=[RAGChunk(**obj.properties) for obj in chunks],
            images=[RAGImageInfo(**img.properties) for img in images]
        )

    except Exception as e:
        logging.error(f"❌ Errore durante la query: {str(e)}")
        raise HTTPException(500, detail=str(e))

def build_context(chunks, images):
    context = "## Fonti tecniche:\n"
    for i, chunk in enumerate(chunks):
        context += f"\n### Fonte {i+1} ({chunk.properties['source']}, pag. {chunk.properties.get('page', 'N/A')})\n"
        context += f"{chunk.properties['text']}\n"
    
    context += "\n## Immagini rilevanti:\n"
    for img in images:
        context += f"- {img.properties['description']} ({img.properties['url']})\n"
    
    return context

async def generate_answer(context: str, question: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "HTTP-Referer": os.getenv("SITE_URL", "https://breadbot-ai.com"),
                    "X-Title": "BreadBot AI"
                },
                json={
                    "model": os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free"),
                    "messages": [{
                        "role": "user",
                        "content": f"""
                        Sei un esperto di elettronica. Fornisci una risposta dettagliata basata SOLO su queste fonti:
                        
                        {context}
                        
                        Domanda: {question}
                        Risposta:"""
                    }],
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            )
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"❌ Errore generazione risposta: {str(e)}")
        return "Impossibile generare una risposta al momento."

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "weaviate": app.weaviate_client.is_ready(),
        "redis": await app.redis_client.ping(),
        "version": "1.0.0"
    }
