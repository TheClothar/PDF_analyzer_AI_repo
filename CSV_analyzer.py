import re
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sklearn.neighbors import NearestNeighbors
import os
import csv

# Processes text to make it easier to read.
def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

# Read CSV data and extract text
def read_csv(file_path):
    text_list = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row if present
        for row in reader:
            text = ' '.join(preprocess(column) for column in row)
            text_list.append(text)
    return text_list


# Location of the CSV file
file_path = "/Users/noah/Documents/Atlas invest documents/dataset_google-search-scraper_2023-07-05_11-10-14-871.csv"

# Read the CSV file and extract text
texts = read_csv(file_path)

# Create a vector store using FAISS
start_time = time.time()
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
doc_search = FAISS.from_texts(texts, embeddings)
end_time = time.time()

# Calculate the execution time for embeddings
embedding_time = end_time - start_time
print(f"Embeddings took {embedding_time} seconds")

# Test query (name to search)
query = "Zach Ehrlich"

# Perform similarity search
start_time = time.time()
neighbors = doc_search.similarity_search(query, k=5)
end_time = time.time()

# Calculate the execution time for semantic search
search_time = end_time - start_time
print(f"Semantic search took {search_time} seconds")

# Print search results
print(neighbors)