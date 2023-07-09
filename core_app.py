from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Location of the PDF file
file_path = "/Users/noah/Downloads/992-998 Amsterdam Avenue Offering Memorandum.pdf"

# Read the PDF file using PdfReader
doc_reader = PdfReader(file_path)

# Extract text from each page of the PDF
raw_text = ""
for page in doc_reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text

# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=400,
    chunk_overlap=70,
    length_function=len
)



texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Create a vector store using FAISS
doc_search = FAISS.from_texts(texts, embeddings)


# Test question
query = "what is the LTC?"

# Perform similarity search
docs = doc_search.similarity_search(query, k=20)



# Load the question-answering chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Print similarity search results
print(chain.run(input_documents=docs, question=query))