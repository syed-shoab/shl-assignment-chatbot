import os
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load .env if available (optional)
load_dotenv()

# Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize Flask app
app = Flask(__name__)

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("shl_vector_store", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Load LLM
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)


# QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Ask endpoint
@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    raw_response = qa.run(query)

    # Extracting structured assessments from raw response
    assessments = []
    blocks = raw_response.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        item = {}
        for line in lines:
            if "Assessment Name:" in line:
                item["name"] = line.split(":", 1)[1].strip()
            elif "Duration:" in line:
                item["duration"] = line.split(":", 1)[1].strip()
            elif "Skills Covered:" in line:
                item["skills"] = line.split(":", 1)[1].strip()
            elif "Target Role:" in line:
                item["role"] = line.split(":", 1)[1].strip()
            elif "Type:" in line:
                item["type"] = line.split(":", 1)[1].strip()
        if item:
            assessments.append(item)

    return jsonify({"assessments": assessments})




# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)
