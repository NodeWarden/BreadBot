import os
import weaviate
import requests

client = weaviate.connect_to_local(
    host=os.getenv("WEAVIATE_HOST", "localhost"),
    port=int(os.getenv("BREADBOT_WEAVIATE_PORT", 8080)),
    grpc_port=int(os.getenv("BREADBOT_WEAVIATE_GRPC_PORT", 50051)),
    skip_init_checks=True
)
chunks = client.collections.get("Chunks").query.fetch_objects(limit=10)
for chunk in chunks.objects:
    print(chunk.properties)
client.close()