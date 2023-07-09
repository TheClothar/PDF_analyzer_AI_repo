import re
import fitz
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Processes text to make it easier to read.
def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

# Location of the PDF file
file_path = "/Users/noah/Documents/Atlas invest documents/NYC Finance - Printable Page.pdf"

# Changes the PDF to readable text
def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        page = doc.load_page(i)
        text = page.get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Read the PDF file and convert it to text
texts = pdf_to_text(file_path)

# Create a vector store using FAISS
doc_search = FAISS.from_texts(texts, embeddings)

# Test question
query = "Are there any late payments, or payments still due that should have been payed? The date is 07/06/2023 American date"

# Perform similarity search
docs = doc_search.similarity_search(query, k=5)

# Load the question-answering chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Print similarity search results
print(chain.run(input_documents=docs, question=query))