# weaviate_populator.py
import os
import weaviate
import weaviate.classes as wvc # Weaviate Python client v4
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import time

# --- CONFIGURAZIONE ---
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST_LOCAL", "localhost") # Usare 'weaviate' se eseguito DENTRO un container Docker che vede il servizio Weaviate
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT_LOCAL", 8080))
TEXT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
IMAGE_EMBEDDING_MODEL = 'openai/clip-vit-base-patch32' # Assicurati sia lo stesso usato nel backend
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- FUNZIONI HELPER ---
def connect_weaviate():
    print(f" Tentativo di connessione a Weaviate: host={WEAVIATE_HOST}, port={WEAVIATE_PORT}")
    try:
        client = weaviate.connect_to_custom(
            http_host=WEAVIATE_HOST,
            http_port=WEAVIATE_PORT,
            http_secure=False,
            grpc_host=WEAVIATE_HOST,
            grpc_port=50051, # Porta gRPC di default
            grpc_secure=False,
            # Puoi aggiungere headers per API key se Weaviate è protetto
        )
        client.is_ready() # Verifica connessione
        print("✅ Connesso a Weaviate.")
        return client
    except Exception as e:
        print(f"❌ Errore di connessione a Weaviate: {e}")
        print("  Assicurati che Weaviate sia in esecuzione e accessibile.")
        print(f"  Host cercato: {WEAVIATE_HOST}, Porta: {WEAVIATE_PORT}")
        return None

def create_collections_if_not_exist(client):
    # Collection per i chunk di testo
    chunks_collection_name = "Chunks"
    if not client.collections.exists(chunks_collection_name):
        client.collections.create(
            name=chunks_collection_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT), # Nome del PDF
                wvc.config.Property(name="page", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="associated_images", data_type=wvc.config.DataType.TEXT_ARRAY) # Nomi file delle immagini su quella pagina
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none() # Forniremo i vettori manualmente
        )
        print(f"✅ Collection '{chunks_collection_name}' creata.")
    else:
        print(f"ℹ️ Collection '{chunks_collection_name}' già esistente.")

    # Collection per le immagini
    images_collection_name = "Images"
    if not client.collections.exists(images_collection_name):
        client.collections.create(
            name=images_collection_name,
            properties=[
                wvc.config.Property(name="filename", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT), # Path relativo per il frontend
                wvc.config.Property(name="component", data_type=wvc.config.DataType.TEXT), # Es: "PDF: nome.pdf, Pagina: 1"
                wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT) # Descrizione breve
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none() # Forniremo i vettori manualmente
        )
        print(f"✅ Collection '{images_collection_name}' creata.")
    else:
        print(f"ℹ️ Collection '{images_collection_name}' già esistente.")

# --- MAIN ---
def main(structured_data_dir):
    client = connect_weaviate()
    if not client:
        return

    create_collections_if_not_exist(client)
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

    for pdf_name in os.listdir(structured_data_dir):
        pdf_dir_path = os.path.join(structured_data_dir, pdf_name)
        if not os.path.isdir(pdf_dir_path):
            continue

        print(f"\n⏳ Inizio indicizzazione per PDF: {pdf_name}")

        # --- Indicizzazione Testo ---
        text_pages_dir = os.path.join(pdf_dir_path, "text_pages")
        pdf_images_dir = os.path.join(pdf_dir_path, "images_extracted") # Immagini specifiche di questo PDF

        if os.path.exists(text_pages_dir):
            page_files = sorted(os.listdir(text_pages_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            with chunks_collection.batch.dynamic() as batch:
                for page_filename in page_files:
                    page_num = int(page_filename.split('_')[1].split('.')[0])
                    
                    with open(os.path.join(text_pages_dir, page_filename), "r", encoding="utf-8") as f:
                        page_text = f.read()

                    if not page_text.strip():
                        print(f"  ⚠️ Pagina {page_num} di {pdf_name} è vuota, salto chunking.")
                        continue
                    
                    # Trova immagini associate a questa pagina
                    associated_image_filenames = []
                    if os.path.exists(pdf_images_dir):
                        for img_file in os.listdir(pdf_images_dir):
                            if img_file.startswith(f"page_{page_num}_img_"):
                                associated_image_filenames.append(f"{pdf_name}/images_extracted/{img_file}") # Path relativo

                    page_chunks = text_splitter.split_text(page_text)
                    for i, chunk_text in enumerate(page_chunks):
                        text_vector = text_embedder.encode(chunk_text).tolist()
                        properties = {
                            "text": chunk_text,
                            "source": pdf_name,
                            "page": page_num,
                            "associated_images": associated_image_filenames # Salva i nomi file
                        }
                        batch.add_object(properties=properties, vector=text_vector)
                        print(f"  ➕ Aggiunto chunk {i+1} da pagina {page_num} di {pdf_name}")
            print(f"  ✅ Batch chunk di testo per {pdf_name} completato.")


        # --- Indicizzazione Immagini ---
        if os.path.exists(pdf_images_dir):
            with images_collection.batch.dynamic() as batch:
                for image_filename in os.listdir(pdf_images_dir):
                    image_path = os.path.join(pdf_images_dir, image_filename)
                    try:
                        pil_image = Image.open(image_path).convert("RGB")
                        inputs = clip_processor(images=pil_image, return_tensors="pt")
                        with torch.no_grad():
                            image_vector = clip_model.get_image_features(**inputs).squeeze().cpu().numpy().tolist()
                        
                        page_num_str = image_filename.split('_')[1] # page_N_img_M.ext -> N
                        
                        properties = {
                            "filename": image_filename,
                            "url": f"/static/data/{pdf_name}/images_extracted/{image_filename}", # Path servito da FastAPI
                            "component": f"PDF: {pdf_name}, Pagina: {page_num_str}",
                            "description": f"Immagine '{image_filename}' estratta da pagina {page_num_str} del PDF '{pdf_name}'."
                        }
                        batch.add_object(properties=properties, vector=image_vector)
                        print(f"  🖼️ Aggiunta immagine: {image_filename} da {pdf_name}")
                    except Exception as e_img_idx:
                        print(f"  ❌ Errore indicizzazione immagine {image_path}: {e_img_idx}")
            print(f"  ✅ Batch immagini per {pdf_name} completato.")
        
        print(f"🏁 Indicizzazione per PDF {pdf_name} completata.")

    client.close()
    print("\n🎉 Tutte le indicizzazioni completate. Connessione a Weaviate chiusa.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indicizza testo e immagini estratti in Weaviate.")
    parser.add_argument("structured_data_dir", help="Cartella base contenente i dati strutturati per PDF (output di pdf_extractor.py).")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.structured_data_dir):
        print(f"❌ La cartella specificata non esiste: {args.structured_data_dir}")
    else:
        main(args.structured_data_dir)

