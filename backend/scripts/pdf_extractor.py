# pdf_extractor.py v 0.0
import fitz  # PyMuPDF
import os
from PIL import Image
import io
import argparse

def extract_data_from_pdf(pdf_path, base_output_dir):
    """
    Estrae testo per pagina e immagini da un file PDF.
    Salva il testo in file .txt per pagina e le immagini in sottocartelle.
    """
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(base_output_dir, pdf_filename)
    text_output_dir = os.path.join(pdf_output_dir, "text_pages")
    images_output_dir = os.path.join(pdf_output_dir, "images_extracted")

    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    print(f"📄 Inizio elaborazione PDF: {pdf_filename}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Errore nell'apertura del PDF {pdf_path}: {e}")
        return

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Estrazione testo
        text = page.get_text("text")
        if not text.strip(): # Prova con OCR se il testo è vuoto (scansionato)
            try:
                import pytesseract
                print(f"  ⓘ Tentativo OCR per pagina {page_num + 1} (potrebbe richiedere tempo)...")
                pix = page.get_pixmap(dpi=300)
                img_for_ocr = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img_for_ocr, lang='ita') # Assicurati di avere 'ita' lang pack per Tesseract
            except ImportError:
                print("  ⚠️ pytesseract non trovato. Salto OCR per pagine vuote.")
            except Exception as ocr_e:
                print(f"  ❌ Errore OCR per pagina {page_num + 1}: {ocr_e}")
        
        with open(os.path.join(text_output_dir, f"page_{page_num + 1}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  📄 Testo estratto per pagina {page_num + 1}")

        # Estrazione immagini
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                
                # Salva l'immagine
                img_out_path = os.path.join(images_output_dir, image_filename)
                with open(img_out_path, "wb") as img_file:
                    img_file.write(image_bytes)
                print(f"  🖼️ Immagine estratta: {image_filename}")
            except Exception as e_img:
                print(f"  ❌ Errore estrazione immagine {xref} da pagina {page_num + 1}: {e_img}")

    doc.close()
    print(f"✅ Elaborazione PDF {pdf_filename} completata.")
    print(f"   Output testo: {text_output_dir}")
    print(f"   Output immagini: {images_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrae testo e immagini da file PDF.")
    parser.add_argument("pdf_files", nargs="+", help="Percorso/i dei file PDF da processare.")
    parser.add_argument("--output_dir", default="./out_structured", help="Cartella base per l'output strutturato.")
    
    args = parser.parse_args()

    # Opzionale: installare pytesseract e il language pack italiano se si vogliono gestire PDF scansionati
    # sudo apt-get install tesseract-ocr tesseract-ocr-ita (su Debian/Ubuntu)
    # pip install pytesseract

    for pdf_file_path in args.pdf_files:
        if os.path.exists(pdf_file_path):
            extract_data_from_pdf(pdf_file_path, args.output_dir)
        else:
            print(f"⚠️ File PDF non trovato: {pdf_file_path}")
