import os
import sys
import argparse
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter

def extract_real_images_from_pdf(pdf_path, output_dir, chunk_name="all", page_range=None):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_count = 0
    for page_num in range(len(doc)):
        if page_range and (page_num < page_range[0] or page_num > page_range[1]):
            continue
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(output_dir, f"{chunk_name}_page{page_num+1}_{img_index+1}.{image_ext}")
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_count += 1
    print(f"Totale immagini estratte da {pdf_path} ({chunk_name}): {image_count}")

def split_pdf_by_max_pages(pdf_path, output_dir, max_pages):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    os.makedirs(output_dir, exist_ok=True)
    chunk_ranges = []
    for i in range(0, total_pages, max_pages):
        start = i
        end = min(i + max_pages, total_pages)
        writer = PdfWriter()
        for j in range(start, end):
            writer.add_page(reader.pages[j])
        chunk_name = f"chunk_{(i // max_pages) + 1}_{start+1}-{end}"
        chunk_path = os.path.join(output_dir, f"{chunk_name}.pdf")
        with open(chunk_path, 'wb') as f:
            writer.write(f)
        chunk_ranges.append((chunk_name, start, end-1, chunk_path))
        print(f"Salvato: {chunk_path}")
    return chunk_ranges

def ask_user_for_action(pdf_path, total_pages):
    print(f"\nIl PDF '{pdf_path}' ha solo {total_pages} pagine.")
    print("Il numero massimo di pagine impostato è superiore al numero di pagine del PDF.")
    print("Cosa vuoi fare?")
    print("1. Dimezzare la lunghezza massima e riprovare")
    print("2. Inserire manualmente un nuovo valore")
    print("3. Estrarre solo le immagini effettive dal PDF")
    print("4. Uscire")
    scelta = input("Seleziona opzione (1/2/3/4): ").strip()
    return scelta

def main():
    parser = argparse.ArgumentParser(description="Split PDF in chunk e salva solo immagini effettive.")
    parser.add_argument('pdf_path', help="Percorso del file PDF da processare")
    parser.add_argument('--max_pages', type=int, default=250, help="Numero massimo di pagine per chunk PDF (default: 250)")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    max_pages = args.max_pages
    if not os.path.isfile(pdf_path):
        print(f"File non trovato: {pdf_path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(os.getcwd(), base_name)
    images_dir = os.path.join(output_dir, "images")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    while max_pages >= total_pages:
        scelta = ask_user_for_action(pdf_path, total_pages)
        if scelta == "1":
            max_pages = max(1, max_pages // 2)
            print(f"Nuovo valore max_pages: {max_pages}")
        elif scelta == "2":
            nuovo = input("Inserisci nuovo valore max_pages: ").strip()
            if nuovo.isdigit() and int(nuovo) > 0:
                max_pages = int(nuovo)
            else:
                print("Valore non valido. Riprovo.")
        elif scelta == "3":
            print("Estrazione solo immagini effettive...")
            extract_real_images_from_pdf(pdf_path, os.path.join(images_dir, "all"), "all")
            print("Fatto.")
            sys.exit(0)
        else:
            print("Uscita.")
            sys.exit(0)

    # Split PDF e salva immagini per ogni chunk
    chunk_ranges = split_pdf_by_max_pages(pdf_path, output_dir, max_pages)
    for chunk_name, start, end, chunk_path in chunk_ranges:
        chunk_images_dir = os.path.join(images_dir, chunk_name)
        extract_real_images_from_pdf(pdf_path, chunk_images_dir, chunk_name, page_range=(start, end))

    print("\nOperazione completata!")
    print(f"Tutti i file sono nella cartella: {output_dir}")

if __name__ == "__main__":
    main()
