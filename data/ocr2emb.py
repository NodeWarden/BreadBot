import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_ocr_txt(txt_path, output_dir, images_dir, chunk_size=1000, chunk_overlap=200):
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    output_path = os.path.join(output_dir, base_name)
    os.makedirs(output_path, exist_ok=True)

    # Leggi il file OCR
    with open(txt_path, 'r') as f:
        full_text = f.read()

    # Split per pagina (assumendo formato: === Pagina N ===\nTesto)
    pagine = full_text.split('=== Pagina ')
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', '']
    )

    for page in pagine[1:]:
        header, *content = page.split('\n', 1)
        page_num = header.strip().split()[0]
        page_text = content[0].strip() if content else ''
        sub_chunks = text_splitter.split_text(page_text)
        for sc in sub_chunks:
            # Collega l'immagine della pagina se esiste
            img_name = f"pagina_{int(page_num):03d}.jpg"
            img_path = os.path.join(images_dir, img_name)
            images = [img_name] if os.path.exists(img_path) else []
            chunks.append({
                "page": int(page_num),
                "text": sc,
                "images": images
            })

    # Crea embeddings e salva
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [f"Page {c['page']}: {c['text']}" for c in chunks]
    db = FAISS.from_texts(docs, embeddings, metadatas=chunks)
    db.save_local(os.path.join(output_path, "embeddings"))

    # Salva chunk con riferimenti alle immagini
    with open(os.path.join(output_path, "text_chunks.txt"), "w") as f:
        for i, c in enumerate(chunks):
            f.write(f"=== Chunk {i+1} (Pagina {c['page']}) ===\n")
            f.write(f"Immagini: {', '.join(c['images']) if c['images'] else 'Nessuna'}\n")
            f.write(c['text'] + "\n\n")

    print(f"✅ File OCR processato: {base_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processa file OCR .txt e collega immagini per RAG')
    parser.add_argument('--txt', required=True, help='Percorso file .txt OCR')
    parser.add_argument('--images', required=True, help='Cartella immagini pagine')
    parser.add_argument('--output', default='./out', help='Cartella di output')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Dimensione chunk')
    args = parser.parse_args()

    process_ocr_txt(
        txt_path=args.txt,
        output_dir=args.output,
        images_dir=args.images,
        chunk_size=args.chunk_size
    )
