import os
from pdf2image import convert_from_path
import pytesseract

def pdf_to_ocr_text(pdf_path, output_txt):
    images = convert_from_path(pdf_path)
    with open(output_txt, "w") as f:
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang='ita')
            f.write(f"=== Pagina {i+1} ===\n{text}\n\n")
            print(f"Pagina {i+1} processata")
    print(f"Testo OCR salvato in {output_txt}")

# ESEMPIO DI USO:
# Sostituisci 'tuofile.pdf' e 'output_ocr.txt' con i tuoi percorsi
pdf_to_ocr_text("disp_attiviEnon.pdf", "disp_attiviEnon.txt")
