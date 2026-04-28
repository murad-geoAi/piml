import pypandoc
import os

print("Downloading pandoc...")
pypandoc.download_pandoc()

input_file = r'f:\PIML-Conferance\piml\APMCE_2026_Manuscript.md'
output_file = r'f:\PIML-Conferance\piml\APMCE_2026_Manuscript.docx'

print("Converting file...")
pypandoc.convert_file(input_file, 'docx', outputfile=output_file)
print("Conversion complete!")
