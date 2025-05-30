# backend/api.py
import os
import json
import datetime
import weaviate
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import redis.asyncio as redis
from dotenv import load_dotenv

# Configurazione ambiente
load_dotenv()

app = FastAPI(title="RAG System", version="1.0")

# Configurazione Weaviate
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
WEAVIATE_CLASS_NAME = "Chunks"
client = weaviate.Client(WEAVIATE_URL)

# Configurazione Redis per caching e feedback
REDIS_URL = "redis://redis:6379"
redis_client = redis.from_url(REDIS_URL)

# Configurazione OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.7

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    feedback_key: str

class FeedbackRequest(BaseModel):
    feedback_key: str
    score: int  # 1-5
    comment: Optional[str]

@app.on_event("startup")
async def startup():
    if not client.is_ready():
        raise RuntimeError("Weaviate non raggiungibile")
    await redis_client.ping()

@app.on_event("shutdown")
async def shutdown():
    await redis_client.close()

async def hybrid_search(query: str, top_k: int) -> List[dict]:
    """Ricerca ibrida vettoriale + keyword su Weaviate"""
    try:
        result = (
            client.query
            .get(WEAVIATE_CLASS_NAME, ["text", "source", "page", "images"])
            .with_hybrid(query=query, alpha=0.5)
            .with_limit(top_k)
            .do()
        )
        return result.get("data", {}).get("Get", {}).get(WEAVIATE_CLASS_NAME, [])
    except Exception as e:
        raise HTTPException(500, f"Errore ricerca Weaviate: {str(e)}")

async def generate_answer(context: str, question: str) -> str:
    """Generazione risposta con LLM e caching"""
    cache_key = f"response:{hash(context+question)}"
    cached = await redis_client.get(cache_key)
    if cached:
        return cached.decode()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/NodeWarden/BreadBot",
        "X-Title": "RAG System"
    }

    prompt = f"""Sei un esperto di elettronica. Rispondi in italiano usando SOLO queste fonti:

{context}

Domanda: {question}
Risposta dettagliata:"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500
            },
            timeout=30
        )
        response.raise_for_status()
        
        answer = response.json()["choices"][0]["message"]["content"].strip()
        await redis_client.setex(cache_key, 3600, answer)  # Cache 1h
        return answer
    except Exception as e:
        raise HTTPException(500, f"Errore generazione risposta: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    # Ricerca contesto
    chunks = await hybrid_search(request.question, request.top_k)
    
    if not chunks:
        return QueryResponse(
            answer="Nessuna informazione rilevante trovata.",
            sources=[],
            feedback_key=""
        )

    # Costruzione contesto
    context = "\n\n".join(
        f"Fonte {i+1} ({c['source']} - Pag. {c['page']}):\n{c['text']}\n"
        f"Immagini: {', '.join(c['images']) if c['images'] else 'Nessuna'}"
        for i, c in enumerate(chunks)
    )

    # Generazione risposta
    answer = await generate_answer(context, request.question)
    
    # Salva feedback key
    feedback_key = f"feedback:{hash(context+answer)}"
    await redis_client.hset(feedback_key, mapping={
        "question": request.question,
        "answer": answer,
        "sources": json.dumps([c["_additional"] for c in chunks])
    })

    return QueryResponse(
        answer=answer,
        sources=[{"source": c["source"], "page": c["page"]} for c in chunks],
        feedback_key=feedback_key
    )

@app.post("/feedback")
async def handle_feedback(request: FeedbackRequest):
    """Salva feedback per miglioramento futuro"""
    feedback_data = {
        "score": request.score,
        "comment": request.comment,
        "timestamp": datetime.now().isoformat()
    }
    await redis_client.hset(request.feedback_key, mapping=feedback_data)
    return {"status": "Feedback salvato"}
