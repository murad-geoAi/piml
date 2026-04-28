import pypandoc
import os

def convert():
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("Pandoc not found. Downloading...")
        pypandoc.download_pandoc()

    input_file = "APMCE_2026_Manuscript.md"
    output_file = "APMCE_2026_Manuscript.docx"
    
    print(f"Converting {input_file} to {output_file}...")
    pypandoc.convert_file(
        input_file,
        'docx',
        outputfile=output_file
    )
    print("Conversion complete.")

if __name__ == "__main__":
    convert()
