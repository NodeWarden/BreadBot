import os
import weaviate
import requests
import json
from dotenv import load_dotenv

# Carica variabili d'ambiente da .env
load_dotenv()

# --- Configurazione ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS_NAME = "ElectronicsDocPage" # Deve corrispondere a quello dell'indexer

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "mistralai/mistral-7b-instruct-v0.2")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# --- Inizializzazione Client Weaviate ---
try:
    client = weaviate.Client(url=WEAVIATE_URL)
    if not client.is_ready():
        print(f"ERRORE: Weaviate a {WEAVIATE_URL} non è pronto.")
        exit()
    print(f"Connesso a Weaviate ({WEAVIATE_URL}) e pronto.")
except Exception as e:
    print(f"ERRORE CRITICO: Impossibile connettersi a Weaviate: {e}")
    exit()

if not OPENROUTER_API_KEY:
    print("ERRORE CRITICO: La variabile d'ambiente OPENROUTER_API_KEY non è impostata.")
    print("Vai su https://openrouter.ai per ottenerne una e aggiungila al tuo file .env.")
    exit()

def query_openrouter_llm(prompt_messages, site_url=None, app_name=None, max_tokens=1500, temperature=0.5):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    # OpenRouter consiglia di inviare questi header per un routing migliore e debug
    if site_url: headers["HTTP-Referer"] = site_url
    if app_name: headers["X-Title"] = app_name

    data = {
        "model": OPENROUTER_MODEL_NAME,
        "messages": prompt_messages, # Lista di messaggi {role: "system/user/assistant", content: "..."}
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    print(f"\n--- Invio richiesta a OpenRouter (Modello: {OPENROUTER_MODEL_NAME}) ---")
    # print(f"Prompt (ultimo messaggio utente): {prompt_messages[-1]['content'][:300]}...") # Debug
    
    try:
        response = requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=data, timeout=180) # Timeout lungo
        response.raise_for_status() 
        response_json = response.json()
        
        if response_json.get("choices") and response_json["choices"][0].get("message"):
            content = response_json["choices"][0]["message"]["content"]
            print("--- Risposta ricevuta da OpenRouter ---")
            return content.strip()
        else:
            error_detail = response_json.get("error", {}).get("message", "Formato risposta inatteso.")
            print(f"ERRORE: Risposta LLM non valida o vuota da OpenRouter: {error_detail}")
            print(f"Dettagli risposta completa: {response_json}")
            return f"Errore: L'LLM non ha fornito una risposta valida ({error_detail})."
            
    except requests.exceptions.HTTPError as e:
        print(f"ERRORE HTTP da OpenRouter: {e.response.status_code} - {e.response.text}")
        return f"Errore HTTP {e.response.status_code} dall'LLM. Dettagli: {e.response.text[:200]}"
    except requests.exceptions.RequestException as e:
        print(f"ERRORE di comunicazione con OpenRouter: {e}")
        return f"Errore di comunicazione con l'LLM: {e}"
    except Exception as e:
        print(f"ERRORE generico durante chiamata LLM: {e}")
        return f"Errore generico durante la generazione della risposta LLM: {e}"

def perform_rag_query(user_query, top_k=3):
    print(f"\nDomanda dell'utente: \"{user_query}\"")

    # 1. Recupero da Weaviate
    print(f"\n--- Ricerca in Weaviate (top_k={top_k}) ---")
    try:
        near_text_filter = {"concepts": [user_query]}
        
        query_builder = (
            client.query
            .get(WEAVIATE_CLASS_NAME, [
                "pdf_filename", 
                "page_number", 
                "text_content", 
                # "page_image_base64" # Non passiamo l'immagine base64 all'LLM per non appesantire troppo il prompt
                                      # ma sappiamo che esiste e possiamo riferirci ad essa.
                "_additional {certainty distance id}" 
            ])
            .with_near_text(near_text_filter)
            .with_limit(top_k)
        )
        query_result = query_builder.do()

        retrieved_chunks = query_result.get("data", {}).get("Get", {}).get(WEAVIATE_CLASS_NAME, [])

        if not retrieved_chunks:
            print("Nessun chunk rilevante trovato in Weaviate.")
            return "Mi dispiace, non ho trovato informazioni specifiche nei miei documenti per rispondere alla tua domanda.", []
        
        print(f"Recuperati {len(retrieved_chunks)} chunk(s) da Weaviate:")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  {i+1}. PDF: {chunk['pdf_filename']} (Pag. {chunk['page_number']}), "
                  f"Certezza: {chunk['_additional']['certainty']:.4f}, ID: {chunk['_additional']['id']}")
            # print(f"     Testo: {chunk['text_content'][:100].replace('\n', ' ')}...") # Anteprima

    except Exception as e:
        print(f"ERRORE durante la query a Weaviate: {e}")
        return "Si è verificato un errore tecnico durante la ricerca delle informazioni.", []

    # 2. Preparazione Contesto per LLM
    context_for_llm_str = "Basandoti ESCLUSIVAMENTE sulle seguenti informazioni estratte dai documenti, rispondi alla domanda dell'utente.\n"
    context_for_llm_str += "Ogni blocco di informazione proviene da una pagina specifica di un PDF e potrebbe contenere sia testo che un'immagine (a cui mi riferirò come '[IMMAGINE PRESENTE SU QUESTA PAGINA]').\n\n"

    for i, chunk in enumerate(retrieved_chunks):
        context_for_llm_str += f"--- Contesto dalla Fonte {i+1} ---\n"
        context_for_llm_str += f"Documento: {chunk['pdf_filename']}, Pagina: {chunk['page_number']}\n"
        context_for_llm_str += "[IMMAGINE PRESENTE SU QUESTA PAGINA]\n" # Assumiamo che ogni pagina indicizzata abbia un'immagine
        context_for_llm_str += f"Testo estratto dalla pagina:\n'''\n{chunk['text_content']}\n'''\n"
        context_for_llm_str += "--------------------------------\n\n"
    
    # 3. Generazione Risposta con LLM (OpenRouter)
    system_message = ("Sei un assistente AI specializzato in elettronica ed elettrotecnica, progettato per aiutare studenti delle scuole superiori italiane. "
                      "Il tuo obiettivo è fornire risposte chiare, utili e stimolanti per lo studio e la realizzazione pratica di circuiti. "
                      "Rispondi ALLA DOMANDA DELL'UTENTE basandoti SOLO ED ESCLUSIVAMENTE sul contesto fornito. "
                      "Se nel contesto viene menzionata '[IMMAGINE PRESENTE SU QUESTA PAGINA]', riconosci che la pagina originale contiene elementi visivi (come schemi, figure, grafici) che potrebbero essere cruciali. Invita l'utente a consultare l'immagine sulla pagina indicata se rilevante. "
                      "Non inventare informazioni non presenti nel contesto. Se il contesto non è sufficiente per una risposta completa, spiegalo chiaramente. "
                      "Sii preciso e, se possibile, offri spunti di riflessione o suggerimenti pratici RELATIVI al contesto fornito.")

    # Costruiamo la lista di messaggi per l'API chat/completions
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{context_for_llm_str}Domanda dell'utente: {user_query}\n\nRisposta:"}
    ]
    
    generated_answer = query_openrouter_llm(prompt_messages, app_name="AssistenteElettronicaRAG")

    # Allega i chunk recuperati alla risposta per riferimento (senza il base64 dell'immagine per brevità)
    sources_for_display = [{
        "pdf_filename": c["pdf_filename"], 
        "page_number": c["page_number"],
        "id": c["_additional"]["id"] # Utile per debug o futuri link diretti se l'immagine fosse servita da qualche parte
        } for c in retrieved_chunks]

    return generated_answer, sources_for_display


if __name__ == "__main__":
    print("Assistente Elettronica RAG (usa 'exit' o 'quit' per uscire)")
    while True:
        user_input = input("\nCosa vorresti chiedermi sull'elettronica?\n> ")
        if user_input.lower() in ['exit', 'quit', 'esci', 'basta']:
            print("Alla prossima! Spero di esserti stato utile.")
            break
        if not user_input.strip():
            continue

        final_answer, sources = perform_rag_query(user_input, top_k=3) # Puoi cambiare top_k
        
        print("\n\n💡 RISPOSTA DELL'ASSISTENTE:\n")
        print(final_answer)
        print("------------------------------------")

        if sources:
            print("\nℹ️ Fonti consultate per questa risposta (testo e immagine della pagina):")
            for i, source in enumerate(sources):
                print(f"  [{i+1}] PDF: {source['pdf_filename']}, Pagina: {source['page_number']} (ID: {source['id']})")
        else:
            print("\nℹ️ Nessuna fonte specifica è stata utilizzata o trovata per questa risposta.")