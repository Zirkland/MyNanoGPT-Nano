import os
from pdfdocx import read_pdf

# Initialize an empty string to hold the combined text
p_text = ""

# Directory containing the PDF files
directory = '.'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(directory, filename)
        # Read the PDF content and append it to p_text
        p_text += read_pdf(file_path) + "\n"

# Write the combined content to a text file
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(p_text)

print("All PDF contents have been written to 'output.txt'")