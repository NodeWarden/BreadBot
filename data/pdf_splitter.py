import sys
import argparse
from PyPDF2 import PdfReader, PdfWriter

# Funzione aggiornata di estrazione immagini con log e tipo immagine nel nome file
import fitz  # PyMuPDF
import os

def get_image_type(base_image):
    cs_name = base_image.get('cs-name', '').lower() if 'cs-name' in base_image else ''
    bpc = base_image.get('bpc', 0)
    width = base_image.get('width', 0)
    height = base_image.get('height', 0)
    if bpc <= 8 and ('gray' in cs_name or 'alpha' in cs_name or bpc == 1):
        return 'grafico'
    if width < 200 or height < 200:
        return 'grafico'
    return 'foto'

def extract_filtered_images_with_log_and_type(pdf_path, output_dir, chunk_name="all", page_range=None, min_width=100, min_height=100, log_file_path=None):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_count = 0
    filtered_out_count = 0
    log_lines = []
    for page_num in range(len(doc)):
        if page_range and (page_num < page_range[0] or page_num > page_range[1]):
            continue
        page = doc[page_num]
        images = page.get_images(full=True)
        if not images:
            continue
        page_image_index = 1
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            # Filtra immagini troppo piccole
            if width < min_width or height < min_height:
                filtered_out_count += 1
                continue
            # Determina tipo immagine
            img_type = get_image_type(base_image)
            # Nome file con tipo immagine
            image_path = os.path.join(
                output_dir,
                f"{chunk_name}_page_{page_num+1:03d}_img_{page_image_index}_{img_type}.{image_ext}"
            )
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_count += 1
            # Aggiungo riga al log: nome file, pagina, dimensioni, tipo
            log_lines.append(f"{os.path.basename(image_path)}\tPage: {page_num+1}\tWidth: {width}\tHeight: {height}\tType: {img_type}")
            page_image_index += 1
    # Scrivo il file di log se specificato
    if log_file_path:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write("Filename\tPage\tWidth\tHeight\tType\n")
            log_file.write("\n".join(log_lines))
    print(f"Totale immagini estratte da {pdf_path} ({chunk_name}): {image_count}")
    print(f"Immagini scartate per dimensioni troppo piccole: {filtered_out_count}")

# Funzione di split PDF

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

# Funzione interattiva per scelta utente

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

# Funzione main

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split PDF in chunk e salva solo immagini effettive con log e tipo immagine nel nome file.")
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
            extract_filtered_images_with_log_and_type(pdf_path, os.path.join(images_dir, "all"), "all", log_file_path=os.path.join(images_dir, "all", "extraction_log.txt"))
            print("Fatto.")
            sys.exit(0)
        else:
            print("Uscita.")
            sys.exit(0)

    # Split PDF e salva immagini per ogni chunk
    chunk_ranges = split_pdf_by_max_pages(pdf_path, output_dir, max_pages)
    for chunk_name, start, end, chunk_path in chunk_ranges:
        chunk_images_dir = os.path.join(images_dir, chunk_name)
        log_path = os.path.join(chunk_images_dir, "extraction_log.txt")
        extract_filtered_images_with_log_and_type(pdf_path, chunk_images_dir, chunk_name, page_range=(start, end), log_file_path=log_path)

    print("\nOperazione completata!")
    print(f"Tutti i file sono nella cartella: {output_dir}")

if __name__ == "__main__":
    main()