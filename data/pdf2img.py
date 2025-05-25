from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path)
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"pagina_{i+1:03d}.jpg")
        img.save(img_path, "JPEG")
        print(f"Salvata {img_path}")

# Esempio d'uso:
# pip install pdf2image
# brew install poppler  # (su Mac)
pdf_to_images("Lorenza-Corti-Elettrotecnica-per-gestionali.pdf", "../out/Lorenza-Corti-Elettrotecnica-per-gestionali/images")
