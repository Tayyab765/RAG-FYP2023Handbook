import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

    
# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
INDEX_PATH = "index/faiss.index"
META_PATH = "index/metadata.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
SIM_THRESHOLD = 0.25


# ---------------------------------------------------------
# LOAD MODEL, INDEX, METADATA
# ---------------------------------------------------------
print("Loading model...")
model = SentenceTransformer(EMBED_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

chunks = metadata["chunks"]        # list of dicts
print(f"Loaded {len(chunks)} chunks.")


# ---------------------------------------------------------
# RETRIEVAL FUNCTION
# ---------------------------------------------------------
def retrieve_chunks(query: str, top_k=TOP_K):
    # Embed + normalize query
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    # Search FAISS index (inner product)
    scores, indices = index.search(q_emb, top_k)

    scores = scores[0]
    indices = indices[0]

    retrieved = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        item = chunks[idx]
        item["similarity"] = float(score)
        retrieved.append(item)

    return retrieved


# ---------------------------------------------------------
# BUILD PROMPT FOR GENERATION
# ---------------------------------------------------------
def build_prompt(user_question: str, retrieved_chunks):
    context_texts = []

    for c in retrieved_chunks:
        context_texts.append(
            f"[Chunk from p.{c['page']} | score={c['similarity']:.3f}]\n{c['text']}\n"
        )

    context_block = "\n\n".join(context_texts)

    prompt = f"""
You are a handbook assistant. Answer ONLY from the context.
Cite page numbers like “(p. X)”. If unsure, say you don’t know.

Question: {user_question}

Context:
{context_block}
    """

    return prompt


# ---------------------------------------------------------
# ANSWER GENERATION (LLM CALL PLACEHOLDER)
# ---------------------------------------------------------

"""
IMPORTANT NOTE:
Your assignment likely requires using ChatGPT/OpenAI locally.

Replace the below function with your LLM of choice, for example:

    import openai
    def call_llm(prompt):
        response = openai.ChatCompletion.create(...)
        return response['choices'][0]['message']['content']

For now, we keep a placeholder that prints the prompt so
you can inspect the RAG pipeline output.
"""

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4.1", "gpt-4o", "o3-mini"
            messages=[
                {"role": "system", "content": "You are a helpful FYP Handbook assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {str(e)}"


# ---------------------------------------------------------
# MAIN QA FUNCTION
# ---------------------------------------------------------
def ask(question: str):
    # Retrieve
    retrieved = retrieve_chunks(question)

    # Check similarity threshold
    if len(retrieved) == 0 or retrieved[0]["similarity"] < SIM_THRESHOLD:
        return "I don’t have that in the handbook.", []

    # Build prompt for LLM
    prompt = build_prompt(question, retrieved)

    # Call model
    answer = call_llm(prompt)

    # Extract page refs
    sources = [f"p.{c['page']} (score={c['similarity']:.3f})" for c in retrieved]

    return answer, sources


# ---------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\nFAST-NUCES FYP Handbook RAG Assistant")
    print("-------------------------------------")

    while True:
        user_q = input("\nAsk a question (or 'exit'): ").strip()
        if user_q.lower() in ["exit", "quit"]:
            break

        answer, sources = ask(user_q)

        print("\nANSWER:")
        print(answer)

        print("\nSources:")
        for s in sources:
            print(" -", s)
