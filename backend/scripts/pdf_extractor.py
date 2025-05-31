import fitz  # PyMuPDF
import os
from PIL import Image
import argparse
import logging

# Configurazione cartelle
INPUT_PDF_DIR = "./data"
OUTPUT_BASE_DIR = "./out_structured"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_extractor.log"), logging.StreamHandler()]
)

def extract_data_from_pdf(pdf_path, base_output_dir):
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(base_output_dir, pdf_filename)
    text_output_dir = os.path.join(pdf_output_dir, "text_pages")
    images_output_dir = os.path.join(pdf_output_dir, "images_extracted")
    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    logging.info(f"📄 Inizio elaborazione PDF: {pdf_filename}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"❌ Errore nell'apertura del PDF {pdf_path}: {e}")
        return

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Estrazione testo
        text = page.get_text("text")
        if not text.strip():
            try:
                import pytesseract
                logging.info(f"  ⓘ Tentativo OCR per pagina {page_num + 1}...")
                pix = page.get_pixmap(dpi=300)
                img_for_ocr = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img_for_ocr, lang='ita')
            except ImportError:
                logging.warning("  ⚠️ pytesseract non trovato. Salto OCR per pagine vuote.")
            except Exception as ocr_e:
                logging.error(f"  ❌ Errore OCR per pagina {page_num + 1}: {ocr_e}")
        with open(os.path.join(text_output_dir, f"page_{page_num + 1}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"  📄 Testo estratto per pagina {page_num + 1}")

        # Estrazione immagini
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                img_out_path = os.path.join(images_output_dir, image_filename)
                with open(img_out_path, "wb") as img_file:
                    img_file.write(image_bytes)
                logging.info(f"  🖼️ Immagine estratta: {image_filename}")
            except Exception as e_img:
                logging.error(f"  ❌ Errore estrazione immagine {xref} da pagina {page_num + 1}: {e_img}")

    doc.close()
    logging.info(f"✅ Elaborazione PDF {pdf_filename} completata.")

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    pdf_files = [os.path.join(INPUT_PDF_DIR, f) for f in os.listdir(INPUT_PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.warning(f"Nessun PDF trovato in {INPUT_PDF_DIR}")
        return
    logging.info(f"🎯 Trovati {len(pdf_files)} PDF da processare")
    for pdf_file_path in pdf_files:
        extract_data_from_pdf(pdf_file_path, OUTPUT_BASE_DIR)
    logging.info("🎉 Estrazione completata per tutti i PDF")

if __name__ == "__main__":
    main()
