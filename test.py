from pypdf import PdfReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Function to extract text from a PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Path to your PDF
pdf_path = "/teamspace/studios/this_studio/ProjectGenAI/BOE-A-1978-31229-consolidado.pdf"

# Extract text from the PDF
text = extract_text_from_pdf(pdf_path)

# Save the extracted text into a text file for indexing
with open("extracted_text.txt", "w") as file:
    file.write(text)

# Read the text file using SimpleDirectoryReader
documents = SimpleDirectoryReader(input_dir=".").load_data()

# Create a vector store index
index = VectorStoreIndex.from_documents(documents)

# Define your query
#query = "Your query text here"

# Query the index
#response = index.query(query)

# Print the response
#print(response)
