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

def init_qa_pipeline():
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

    print("🔹 Loading LLM...")
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-rw-1b",  # 🔄 try this model or any lighter one
        model_kwargs={"temperature": 0.5}
    )

    print("🔹 Creating RetrievalQA chain...")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'GET':
        return jsonify({"message": "Use POST to send a question."})
    
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question field is required"}), 400

    try:
        qa = init_qa_pipeline()
        answer = qa.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        print("❌ Error while running QA:", str(e))
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
