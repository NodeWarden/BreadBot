import os
import argparse
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_pdf, output_dir, max_pages=100):
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]

    print(f"Splitting '{input_pdf}' ({total_pages} pagine) in chunk da massimo {max_pages} pagine...")

    chunk_num = 1
    for start in range(0, total_pages, max_pages):
        writer = PdfWriter()
        end = min(start + max_pages, total_pages)
        for page in range(start, end):
            writer.add_page(reader.pages[page])
        output_path = os.path.join(output_dir, f"{base_name}_part{chunk_num}_{start+1}-{end}.pdf")
        with open(output_path, "wb") as out_file:
            writer.write(out_file)
        print(f"Creato: {output_path}")
        chunk_num += 1

    print("✅ Split completato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitta un PDF in più file da N pagine")
    parser.add_argument("input_pdf", help="Percorso del PDF da splittare")
    parser.add_argument("--output_dir", default="splitted_pdf", help="Cartella di output")
    parser.add_argument("--max_pages", type=int, default=100, help="Numero massimo di pagine per file")
    args = parser.parse_args()

    split_pdf(args.input_pdf, args.output_dir, args.max_pages)
