from flask import Flask, request, jsonify, render_template
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Initialize QA pipeline
qa = None
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
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    print("🔹 Loading smaller LLM...")
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",  # changed from flan-t5-base to reduce memory usage
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    print("🔹 Creating RetrievalQA chain...")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("✅ QA pipeline initialized successfully!")

except Exception as e:
    print("❌ Failed to initialize QA pipeline:", str(e))

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question field is required"}), 400

    if qa is None:
        return jsonify({"error": "QA pipeline not initialized"}), 500

    try:
        answer = qa.invoke({"query": question})  # changed from .run() to .invoke()
        return jsonify({"answer": answer})
    except Exception as e:
        print("❌ Error while running QA:", str(e))
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
