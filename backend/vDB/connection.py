import os
import weaviate
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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


base_dir =  "/out"
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    embeddings_dir = os.path.join(folder_path, "embeddings")
    if not (os.path.isdir(embeddings_dir) and os.path.exists(os.path.join(embeddings_dir, "index.faiss"))):
        continue
    print(f"⏳ Caricamento da: {folder_name}")

    # Carica il database vettoriale FAISS
    vector_store = FAISS.load_local(embeddings_dir, embeddings)
    doc_ids = list(vector_store.index_to_docstore_id.values())
    docs = [vector_store.docstore.__dict__[doc_id] for doc_id in doc_ids]
    # Estrai i vettori uno per doc
    vectors = [vector_store.index.reconstruct(i) for i in range(vector_store.index.ntotal)]


    # Carica batch su Weaviate
    with client.batch(batch_size=100) as batch:
        for i, doc in enumerate(docs):
            # Metadati: immagini, source, pagina
            data_object = {
                "text": doc.page_content,
                "images": doc.metadata.get("images", []),
                "source": folder_name,
                "page": doc.metadata.get("page", -1),
                "metadata": str(doc.metadata)
            }
            batch.add_data_object(
                data_object=data_object,
                class_name="Chunks",
                vector=vectors[i].tolist()  # Converti numpy array in lista
            )
    print(f"✅ Caricamento completato di {len(docs)} chunks da: {folder_name}")

print("🎉 Tutti i chunks sono stati caricati su Weaviate con successo!")