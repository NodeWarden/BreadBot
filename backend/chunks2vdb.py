import os
import weaviate
import weaviate.classes as wvc

OUT_DIR = "./out"

def connect_weaviate():
    client = weaviate.connect_to_local()
    print("✅ Connessione a Weaviate stabilita.")
    return client

def create_chunks_collection(client):
    if "Chunks" not in client.collections.list_all():
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
    else:
        print("ℹ️ Collection 'Chunks' già esistente.")

def parse_chunks_txt(txt_path):
    chunks = []
    with open(txt_path, "r") as f:
        content = f.read().split("=== Chunk ")
        for chunk_block in content[1:]:
            lines = chunk_block.strip().split("\n")
            header = lines[0]
            page = None
            if "(Pagina" in header:
                page = int(header.split("(Pagina")[1].split(")")[0].strip())
            images_line = lines[1]
            images = []
            if "Immagini:" in images_line:
                images = [img.strip() for img in images_line.replace("Immagini:", "").split(",") if img.strip() and img.strip() != "Nessuna"]
            text = "\n".join(lines[2:]).strip()
            chunks.append({
                "text": text,
                "images": images,
                "page": page if page else -1
            })
    return chunks

def main():
    client = connect_weaviate()
    try:
        create_chunks_collection(client)
        collection = client.collections.get("Chunks")

        for folder_name in os.listdir(OUT_DIR):
            folder_path = os.path.join(OUT_DIR, folder_name)
            txt_path = os.path.join(folder_path, "text_chunks.txt")
            if not os.path.exists(txt_path):
                continue
            print(f"📄 Indicizzo chunk da: {txt_path}")
            chunks = parse_chunks_txt(txt_path)
            for chunk in chunks:
                chunk["source"] = folder_name
                # NB: Qui NON carichiamo embedding, solo testo e metadati
                collection.data.insert(
                    properties=chunk
                )
            print(f"✅ {len(chunks)} chunk caricati da {folder_name}")

    finally:
        client.close()
        print("🔒 Connessione a Weaviate chiusa.")

if __name__ == "__main__":
    main()
