import os
import weaviate
import weaviate.classes as wvc
from langchain_huggingface import HuggingFaceEmbeddings

client = weaviate.connect_to_local()

try:
    collection = client.collections.get("Chunks")

    # Conta tutti i chunk
    total = collection.aggregate.over_all(total_count=True)
    print(f"Totale chunk nel database: {total.total_count}")

    # Visualizza i primi 3 chunk di una source specifica
    result = collection.query.fetch_objects(
        limit=3,
        filters=wvc.query.Filter.by_property("source").equal("Lorenza-Corti-Elettrotecnica-per-gestionali")
    )
    for obj in result.objects:
        print(obj.properties)
finally:
    client.close()
