import os
import weaviate
import weaviate.classes as wvc

WEAVIATE_HOST = os.getenv("BREADBOT_WEAVIATE_HOST", "weaviate-service")
WEAVIATE_PORT = os.getenv("BREADBOT_WEAVIATE_PORT", "8080")

def count_objects_in_collections():
    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=int(WEAVIATE_PORT),
        http_secure=False
    )
    try:
        collections_to_check = ["Chunks", "Images"]
        counts = {}
        for cname in collections_to_check:
            if cname in client.collections.list_all():
                collection = client.collections.get(cname)
                count = collection.aggregate.over_all(total_count=True).total_count
                counts[cname] = count
            else:
                counts[cname] = 0
        return counts
    finally:
        client.close()

if __name__ == "__main__":
    counts = count_objects_in_collections()
    for cname, count in counts.items():
        print(f"{cname}: {count} oggetti")
