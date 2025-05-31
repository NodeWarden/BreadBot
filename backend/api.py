import os
import json
import asyncio
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.init import Timeout, AdditionalConfig

import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import httpx

from dotenv import load_dotenv

# --- Configurazione ambiente ---
load_dotenv()

class RAGChunk(BaseModel):
    text: str
    source: str
    page: int
    images: List[str]

class RAGResponse(BaseModel):
    answer: str
    chunks: List[RAGChunk]
    images: List[str]
    diagrams: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Attendi che Weaviate sia pronto (max 60 secondi)
    for _ in range(30):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://{os.getenv('WEAVIATE_HOST', 'weaviate')}:8080/v1/.well-known/ready", timeout=2)
                if r.status_code == 200:
                    break
        except Exception:
            pass
        await asyncio.sleep(2)
    else:
        raise RuntimeError("Weaviate non raggiungibile")

    # Connessione Weaviate v4 (corretta)
    app.weaviate = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host=os.getenv("WEAVIATE_HOST", "weaviate"),
            http_port=8080,
            http_secure=False,
            grpc_host=os.getenv("WEAVIATE_HOST", "weaviate"),
            grpc_port=50051,
            grpc_secure=False
        ),
        additional_config=AdditionalConfig(timeout=Timeout(query=45))
    )
    app.weaviate.connect()

    # Modelli embedding
    app.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    app.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    app.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Redis
    app.redis = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
    await FastAPILimiter.init(app.redis)

    yield

    await app.redis.aclose()
    app.weaviate.close()

app = FastAPI(
    title="Electronics RAG API",
    lifespan=lifespan,
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/query", response_model=RAGResponse)
async def hybrid_search(
    q: str = Query(..., min_length=3, max_length=500),
    top_k: int = Query(5, ge=1, le=10)
):
    try:
        # Embedding testo
        text_vector = app.text_encoder.encode(q).tolist()

        # Embedding immagine (CLIP)
        inputs = app.clip_processor(text=[q], return_tensors="pt", padding=True)
        with torch.no_grad():
            image_vector = app.clip_model.get_text_features(**inputs)[0].tolist()

        # Hybrid search testuale
        text_results = app.weaviate.collections.get("Chunks").query.hybrid(
            query=q,
            vector=text_vector,
            limit=top_k,
            return_properties=["text", "source", "page", "images"]
        ).objects

        # Hybrid search immagini
        image_results = app.weaviate.collections.get("Images").query.hybrid(
            query=q,
            vector=image_vector,
            limit=top_k,
            return_properties=["url", "description"]
        ).objects

        # Estrazione risultati
        chunks = [
            RAGChunk(
                text=obj.properties["text"],
                source=obj.properties["source"],
                page=obj.properties.get("page", 0),
                images=obj.properties.get("images", [])
            ) for obj in text_results
        ]

        diagrams = [img.properties["url"] for img in image_results if "url" in img.properties]

        # Costruzione contesto tecnico
        context = "Fonti tecniche:\n"
        context += "\n\n".join(
            f"[{c.source}, pag.{c.page}] {c.text}"
            for c in chunks
        )
        context += "\n\nDiagrammi rilevanti:\n"
        context += "\n".join(f"- {url}" for url in diagrams)

        # Generazione risposta LLM
        answer = await generate_answer(context, q)

        return RAGResponse(
            answer=answer,
            chunks=chunks,
            images=[img for c in chunks for img in c.images],
            diagrams=diagrams
        )

    except Exception as e:
        raise HTTPException(500, detail=f"Errore: {str(e)}")

async def generate_answer(context: str, question: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:3000")
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{
                    "role": "user",
                    "content": f"""
                    Sei un esperto di elettronica, ti occupi di assistenza allo studio delle principali materie di indirizzo delle scuole superiori italiane.
                    Devi rispondere a domande tecniche su argomenti di elettronica e realizzazione di circuiti elettronici su BreadBoard e PCB.
                    Le tue risposte devono essere basate su fonti tecniche aggiornate per usare anche i moderni sistemi di controllo ed elaborazione dati per sistemi automatici.
                    Queste includono: elettronica, elettrotecnica, sistemi automatici, tpsee, informatica e fisica. 
                    Rispondi con un italiano , preciso, accurato e comprensibile da tutti (bambini inclusi) usando queste fonti e risorse online accademiche:

                    {context}

                    Domanda: {question}
                    Risposta:"""
                }],
                "temperature": 0.3,
                "max_tokens": 1200
            }
        )
        return response.json()["choices"][0]["message"]["content"].strip()

@app.get("/health")
async def health_check():
    return {
        "weaviate": app.weaviate.is_ready(),
        "redis": await app.redis.ping()
    }
