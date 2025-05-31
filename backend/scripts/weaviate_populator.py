import os
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Configurazione
STRUCTURED_DATA_DIR = "./out_structured"
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST_LOCAL", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT_LOCAL", 8080))
TEXT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
IMAGE_EMBEDDING_MODEL = 'openai/clip-vit-base-patch32'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("weaviate_populator.log"), logging.StreamHandler()]
)

def connect_weaviate():
    print(f" Tentativo di connessione a Weaviate: host={WEAVIATE_HOST}, port={WEAVIATE_PORT}")
    try:
        client = weaviate.connect_to_custom(
            http_host=WEAVIATE_HOST,
            http_port=WEAVIATE_PORT,
            http_secure=False,
            grpc_host=WEAVIATE_HOST,
            grpc_port=50051,
            grpc_secure=False,
        )
        client.is_ready()
        print("✅ Connesso a Weaviate.")
        return client
    except Exception as e:
        print(f"❌ Errore di connessione a Weaviate: {e}")
        return None

def check_and_create_collection(client, name, properties):
    if client.collections.exists(name):
        resp = input(f"⚠️ La collection '{name}' esiste già. Vuoi sovrascriverla? (s/N): ")
        if resp.strip().lower() == 's':
            client.collections.delete(name)
            print(f"🗑️ Collection '{name}' eliminata.")
        else:
            print(f"⏩ Mantengo la collection '{name}'.")
            return
    client.collections.create(
        name=name,
        properties=properties,
        vectorizer_config=wvc.config.Configure.Vectorizer.none()
    )
    print(f"✅ Collection '{name}' creata.")

def main():
    client = connect_weaviate()
    if not client:
        return

    # Definizione schema
    chunk_props = [
        wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="page", data_type=wvc.config.DataType.INT),
        wvc.config.Property(name="images", data_type=wvc.config.DataType.TEXT_ARRAY)
    ]
    img_props = [
        wvc.config.Property(name="filename", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="component", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT)
    ]
    check_and_create_collection(client, "Chunks", chunk_props)
    check_and_create_collection(client, "Images", img_props)

    chunks_collection = client.collections.get("Chunks")
    images_collection = client.collections.get("Images")

    # Inizializza modelli di embedding
    print("🔄 Caricamento modelli di embedding...")
    text_embedder = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
    clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
    print("✅ Modelli di embedding pronti.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    for pdf_name in os.listdir(STRUCTURED_DATA_DIR):
        pdf_dir_path = os.path.join(STRUCTURED_DATA_DIR, pdf_name)
        if not os.path.isdir(pdf_dir_path):
            continue
        print(f"\n⏳ Inizio indicizzazione per PDF: {pdf_name}")
        text_pages_dir = os.path.join(pdf_dir_path, "text_pages")
        pdf_images_dir = os.path.join(pdf_dir_path, "images_extracted")

        # --- Indicizzazione Testo ---
        if os.path.exists(text_pages_dir):
            page_files = sorted(os.listdir(text_pages_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
            with chunks_collection.batch.fixed_size(batch_size=64, concurrent_requests=2) as batch:
                for page_filename in page_files:
                    page_num = int(page_filename.split('_')[1].split('.')[0])
                    with open(os.path.join(text_pages_dir, page_filename), "r", encoding="utf-8") as f:
                        page_text = f.read()
                    if not page_text.strip():
                        print(f" ⚠️ Pagina {page_num} di {pdf_name} è vuota, salto chunking.")
                        continue
                    # Trova immagini associate a questa pagina
                    associated_image_filenames = []
                    if os.path.exists(pdf_images_dir):
                        for img_file in os.listdir(pdf_images_dir):
                            if img_file.startswith(f"page_{page_num}_img_"):
                                associated_image_filenames.append(f"{pdf_name}/images_extracted/{img_file}")
                    page_chunks = text_splitter.split_text(page_text)
                    for i, chunk_text in enumerate(page_chunks):
                        text_vector = text_embedder.encode(chunk_text).tolist()
                        properties = {
                            "text": chunk_text,
                            "source": pdf_name,
                            "page": page_num,
                            "images": associated_image_filenames
                        }
                        batch.add_object(properties=properties, vector=text_vector)
                        print(f" ➕ Aggiunto chunk {i+1} da pagina {page_num} di {pdf_name}")
            print(f" ✅ Batch chunk di testo per {pdf_name} completato.")

        # --- Indicizzazione Immagini ---
        if os.path.exists(pdf_images_dir):
            with images_collection.batch.fixed_size(batch_size=32, concurrent_requests=2) as batch:
                for image_filename in os.listdir(pdf_images_dir):
                    image_path = os.path.join(pdf_images_dir, image_filename)
                    try:
                        pil_image = Image.open(image_path).convert("RGB")
                        inputs = clip_processor(images=pil_image, return_tensors="pt")
                        with torch.no_grad():
                            image_vector = clip_model.get_image_features(**inputs).squeeze().cpu().numpy().tolist()
                        page_num_str = image_filename.split('_')[1]
                        properties = {
                            "filename": image_filename,
                            "url": f"/static/data/{pdf_name}/images_extracted/{image_filename}",
                            "component": f"PDF: {pdf_name}, Pagina: {page_num_str}",
                            "description": f"Immagine '{image_filename}' estratta da pagina {page_num_str} del PDF '{pdf_name}'."
                        }
                        batch.add_object(properties=properties, vector=image_vector)
                        print(f" 🖼️ Aggiunta immagine: {image_filename} da {pdf_name}")
                    except Exception as e_img_idx:
                        print(f" ❌ Errore indicizzazione immagine {image_path}: {e_img_idx}")
            print(f" ✅ Batch immagini per {pdf_name} completato.")
        print(f"🏁 Indicizzazione per PDF {pdf_name} completata.")

    client.close()
    print("\n🎉 Tutte le indicizzazioni completate. Connessione a Weaviate chiusa.")

if __name__ == "__main__":
    main()
