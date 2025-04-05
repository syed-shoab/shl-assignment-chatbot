# SHL Assessment Chatbot

This is an intelligent chatbot(Web App) that recommends relevant SHL assessments based on a natural language query. It uses LangChain, FAISS vector store, and a HuggingFace LLM to process and retrieve assessment information.

---

## 🔧 Tech Stack

- **Flask** – Web framework
- **LangChain** – For building the QA pipeline
- **FAISS** – Vector store for semantic search
- **HuggingFace Hub** – LLM for generating answers
- **Gunicorn** – WSGI server for deployment
- **Render** – Cloud platform for deployment

---

## 🌐 Live Demo

- **Demo URL**: [https://shl-assignment-chatbot.onrender.com](https://shl-assignment-chatbot.onrender.com)
- **API Endpoint**: `POST https://shl-assignment-chatbot.onrender.com/query`

### Example API Usage:
```bash
curl -X POST https://shl-assignment-chatbot.onrender.com/query \
-H "Content-Type: application/json" \
-d '{"question": "What is an SHL assessment?"}'

