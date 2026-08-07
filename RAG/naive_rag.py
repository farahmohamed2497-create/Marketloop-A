from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from groq import Groq
from dotenv import load_dotenv
import os

# -----------------------------
# Groq
# -----------------------------

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
#----------------------------
# Load Vector DB
# -----------------------------
embedder = EmbeddingModel()
store = VectorStore.load("./data/marketloop_vector_db")

# -----------------------------
# User Question
# -----------------------------
query = input("Question: ")

# -----------------------------
# Retrieval
# -----------------------------
query_embedding = embedder.embed(query)

results = store.search(
    query_embedding=query_embedding,
    k=3
)

# -----------------------------
# Build Context
# -----------------------------
context = "\n\n".join(
    r["text"] for r in results
)

# -----------------------------
# Prompt
# -----------------------------
prompt = f"""
You are an assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

# -----------------------------
# Generation
# -----------------------------
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)

print(response.choices[0].message.content)