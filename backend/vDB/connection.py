import weaviate

client = weaviate.Client("http://localhost:8080")

#Creazione della classe dei Chunks:
class_obj = {
    "class": "Chunks",
    "vectorizer": "none",
    "properties": [
        {"name": "text", "dataType": ["text"]},
        {"name": "images", "dataType": ["text[]"]},
        {"name": "source", "dataType": ["text"]},
        {"name": "page", "dataType": ["int"]},
        {"name": "metadata", "dataType": ["text"]},
    ]
}

if not client.schema.exists("Chunks"):
    client.schema.create_class(class_obj)
    print("✅ Classe 'Chunks' creata con successo.")
else:
    print("⚠️ Classe 'Chunks' già esistente. Nessuna azione necessaria.")

if client.is_ready():
    print("✅ Connessione a Weaviate riuscita correttamente!")
else:
    print("❌ Errore di connessione a Weaviate. Verifica che il server sia in esecuzione.")