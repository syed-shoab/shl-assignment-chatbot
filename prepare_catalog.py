import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

# Load CSV
df = pd.read_csv("shl_catalog.csv")

# Convert each row into a chunk of text
data = []
for _, row in df.iterrows():
    text = f"""
    Assessment Name: {row['Assessment Name']}
    Duration: {row['Duration']}
    Skills Covered: {row['Skills Covered']}
    Target Role: {row['Target Role']}
    Type: {row['Type']}
    """
    data.append(text.strip())

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = splitter.create_documents(data)

# Load OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"


# Create vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
db.save_local("shl_vector_store")

print("✅ Vector store created and saved!")
