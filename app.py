from flask import Flask, request, jsonify, render_template
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

qa = None  # Will be initialized later


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    global qa
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question field is required"}), 400

    try:
        answer = qa.run(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def initialize_qa():
    """Initializes the QA chain once on startup."""
    loader = TextLoader("assessments.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-base", temperature=0.5)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


if __name__ == "__main__":
    qa = initialize_qa()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
