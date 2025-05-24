# BreadBotAI


# Assistente Elettronico per Studenti di Istituti Tecnici

## Obiettivo
Sviluppare un assistente intelligente basato su RAG (Retrieval-Augmented Generation) per supportare studenti di elettronica nell’identificazione di componenti, uso della breadboard e supporto in laboratorio.

## Tecnologie
- LangChain + ChromaDB
- FastAPI (backend)
- Gradio (frontend MVP)
- Vosk (Speech-to-Text)
- OpenRouter/Groq (LLM cloud-based)

## Funzionalità principali
- Chat su base documentale tecnica
- Riconoscimento vocale + risposta automatica
- Caricamento immagini per riconoscimento componenti (da implementare)

## Come usare
1. Clonare il progetto
2. Attivare l’ambiente virtuale
3. Inserire le chiavi nel file `.env`
4. Eseguire `run.py`

## To-do
- [ ] Integrazione STT
- [ ] Integrazione Gradio
- [ ] Connessione a LLM (via API)
- [ ] Estensione RAG a immagini
