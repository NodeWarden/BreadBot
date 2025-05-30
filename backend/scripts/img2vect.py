import os
import time
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import weaviate

# Configura percorso base immagini (adatta se necessario)
OUT_DIR = "./out"

# Inizializza CLIP una sola volta
print("🔄 Caricamento modello CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
print("✅ Modello CLIP pronto.")

def connect_weaviate():
    """Crea una connessione a Weaviate e la restituisce."""
    try:
        client = weaviate.connect_to_local()
        print("✅ Connessione a Weaviate stabilita.")
        return client
    except Exception as e:
        print(f"❌ Errore di connessione a Weaviate: {e}")
        return None

def main():
    client = connect_weaviate()
    if client is None:
        return

    try:
        # Ottieni o crea la collection Images
        if "Images" in client.collections.list_all():
            print("ℹ️ Collection 'Images' già esistente.")
            client.collections.delete("Images")
        else:
            client.collections.create(
                name="Images",
                properties=[
                    weaviate.classes.config.Property(name="filename", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="url", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="component", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="description", data_type=weaviate.classes.config.DataType.TEXT),
                ],
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none()
            )
            print("✅ Collection 'Images' creata.")

        collection = client.collections.get("Images")

        # Scorri tutte le sottocartelle in OUT_DIR
        for pdf_folder in os.listdir(OUT_DIR):
            images_dir = os.path.join(OUT_DIR, pdf_folder, "images")
            if not os.path.isdir(images_dir):
                continue
            print(f"🔍 Indicizzazione immagini in: {images_dir}")

            for img_name in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        embedding = model.get_image_features(**inputs).squeeze().cpu().numpy()

                    # Costruisci URL o path accessibile dal frontend/backend
                    url = f"/static/images/{pdf_folder}/{img_name}"

                    # Inserisci in Weaviate
                    collection.data.insert(
                        properties={
                            "filename": img_name,
                            "url": url,
                            "component": "",  # Aggiungi tag se vuoi
                            "description": f"Screenshot pagina da {pdf_folder}"
                        },
                        vector=embedding.tolist()
                    )
                    print(f"✅ Indicizzata: {img_name}")

                except KeyboardInterrupt:
                    print("\n⏹️ Interrotto da tastiera. Chiusura sicura...")
                    client.close()
                    print("🔒 Connessione a Weaviate chiusa.")
                    return
                except Exception as e:
                    print(f"❌ Errore su {img_name}: {e}")
                    # Se errore di connessione, tenta la riconnessione
                    if "connection" in str(e).lower():
                        print("🔄 Tentativo di riconnessione a Weaviate...")
                        client = connect_weaviate()
                        if client is None:
                            print("❌ Impossibile riconnettersi. Esco.")
                            return
                        collection = client.collections.get("Images")
                    continue

    except KeyboardInterrupt:
        print("\n⏹️ Interrotto da tastiera. Chiusura sicura...")
    finally:
        if client is not None:
            client.close()
            print("🔒 Connessione a Weaviate chiusa.")

if __name__ == "__main__":
    main()
