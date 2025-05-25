import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import argparse

def process_pdf(pdf_path, output_dir, chunk_size=1000, chunk_overlap=200):
    """Elabora un singolo PDF e salva gli embedding"""
    try:
        # Caricamento e splitting del documento
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Configurazione dello splitter avanzato
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=['\n\n', '\n', ' ', '']
        )
        
        chunks = text_splitter.split_documents(pages)
        
        # Creazione embeddings con modello open source
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Creazione e salvataggio database vettoriale
        db = FAISS.from_documents(chunks, embeddings)
        
        # Creazione struttura cartelle
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, base_name)
        os.makedirs(output_path, exist_ok=True)
        
        # Salvataggio degli embedding e metadati
        db.save_local(os.path.join(output_path, "embeddings"))
        
        # Salvataggio dei chunk di testo grezzo
        with open(os.path.join(output_path, "text_chunks.txt"), "w") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== Chunk {i+1} ===\n")
                f.write(chunk.page_content + "\n\n")
        
        print(f"✅ Successo: {base_name}")
        return True
    
    except Exception as e:
        print(f"❌ Errore su {pdf_path}: {str(e)}")
        return False

def process_folder(input_dir, output_dir, chunk_size=1000):
    """Elabora tutti i PDF in una cartella"""
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("Nessun PDF trovato nella cartella specificata")
        return
    
    print(f"🍿 Inizio elaborazione di {len(pdf_files)} file...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        process_pdf(pdf_path, output_dir, chunk_size)
    
    print("\n🎉 Elaborazione completata!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Elaborazione PDF per RAG System')
    parser.add_argument('--input', type=str, required=True, help='Cartella contenente i PDF')
    parser.add_argument('--output', type=str, default='output_embeddings', help='Cartella di output')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Dimensione dei chunk di testo')
    
    args = parser.parse_args()
    
    # Verifica e creazione cartelle
    os.makedirs(args.output, exist_ok=True)
    
    # Avvio elaborazione
    process_folder(args.input, args.output, args.chunk_size)
