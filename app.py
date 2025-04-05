from flask import Flask, request, jsonify, render_template
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS as FAISS_DB

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

try:
    print("🔹 Loading documents...")
    loader = TextLoader("assessments.txt")
    documents = loader.load()

    print("🔹 Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print("🔹 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🔹 Creating FAISS vector store...")
    db = FAISS_DB.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    print("🔹 Loading LLM...")
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-base", temperature=0.5)

    print("🔹 Creating RetrievalQA chain...")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("✅ QA pipeline initialized successfully!")

except Exception as e:
    print("❌ Failed to initialize QA pipeline:", str(e))
    qa = None


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question field is required"}), 400

    if qa is None:
        return jsonify({"error": "QA pipeline not initialized on server"}), 500

    answer = qa.run(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
