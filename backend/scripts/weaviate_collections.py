import weaviate
import weaviate.classes as wvc

def reset_weaviate_collections():
    client = weaviate.connect_to_local()
    try:
        # Elimina le collection esistenti (ATTENZIONE: questa operazione è distruttiva!)
        for cname in client.collections.list_all():
            client.collections.delete("Chunks")
            print(f"❌ Collection '{cname}' eliminata.")

        # Crea la nuova collection per i chunk di testo (come prima)
        client.collections.create(
            name="Chunks",
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="images", data_type=wvc.config.DataType.TEXT_ARRAY),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="page", data_type=wvc.config.DataType.INT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )
        print("✅ Collection 'Chunks' creata.")

    finally:
        client.close()
        print("🔒 Connessione a Weaviate chiusa.")

if __name__ == "__main__":
    reset_weaviate_collections()
