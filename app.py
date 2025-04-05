import os
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS as FAISS_DB

app = Flask(__name__)

# Load and split documents
loader = TextLoader("assessments.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
db = FAISS_DB.from_documents(docs, embeddings)
retriever = db.as_retriever()

# LLM setup (make sure HUGGINGFACEHUB_API_TOKEN is set in Render environment)
llm = HuggingFaceEndpoint(repo_id="google/flan-t5-base", temperature=0.5)

# Create QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question field is required"}), 400

    answer = qa.run(question)
    return jsonify({"answer": answer})

@app.route('/')
def index():
    return "SHL Assessment Recommender is running!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)

