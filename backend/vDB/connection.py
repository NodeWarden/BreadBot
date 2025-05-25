import os
import weaviate
import weaviate.classes as wvc
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # Connessione Weaviate v4
    client = weaviate.connect_to_local()
    
    try:
        if client.is_ready():
            print("✅ Connessione a Weaviate riuscita!")
        else:
            print("❌ Errore di connessione")
            return

        # Configurazione collection
        collection_name = "Chunks"
        if collection_name not in client.collections.list_all():
            client.collections.create(
                name=collection_name,
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="images", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="page", data_type=wvc.config.DataType.INT)
                ],
                vectorizer_config=wvc.config.Configure.Vectorizer.none()
            )
            print(f"✅ Collection '{collection_name}' creata")
        else:
            print(f"⚠️ Collection '{collection_name}' esistente")

        # Configura embeddings
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Processa cartelle
        base_dir = "./out"
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            embeddings_dir = os.path.join(folder_path, "embeddings")
            
            if not os.path.exists(os.path.join(embeddings_dir, "index.faiss")):
                continue

            print(f"⏳ Processo: {folder_name}")
            
            try:
                # Carica FAISS con deserializzazione esplicita
                vector_store = FAISS.load_local(
                    embeddings_dir, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Estrai dati
                docs = [vector_store.docstore._dict[doc_id] 
                        for doc_id in vector_store.index_to_docstore_id.values()]
                
                vectors = [vector_store.index.reconstruct(i) 
                          for i in range(vector_store.index.ntotal)]

                # Batch import con configurazione fixed_size
                with client.batch.fixed_size(batch_size=100) as batch:
                    for i, doc in enumerate(docs):
                        batch.add_object(
                            collection=collection_name,
                            properties={
                                "text": doc.page_content,
                                "images": doc.metadata.get("images", []),
                                "source": folder_name,
                                "page": doc.metadata.get("page", -1)
                            },
                            vector=vectors[i].tolist()
                        )
                
                print(f"✅ {len(docs)} chunk caricati")

            except Exception as e:
                print(f"❌ Errore in {folder_name}: {str(e)}")
                continue

        print("🎉 Operazione completata!")

    finally:
        client.close()
        print("🔒 Connessione chiusa")

if __name__ == "__main__":
    main()
