import os
import pickle
from typing import List, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PDF_PATH = "data/handbook.pdf"
INDEX_DIR = "index"
CHUNK_SIZE = 300       # within 250–400
OVERLAP = 100          # 20–40% overlap
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# UTIL: Extract a heading-like section hint
# ---------------------------------------------------------
def extract_section_hint(text: str) -> str:
    """
    Returns the first line that looks like a heading.
    Heuristics:
    - short line (1–8 words)
    - title-like (Starts with uppercase)
    """
    for line in text.split("\n"):
        line_strip = line.strip()
        if not line_strip:
            continue
        words = line_strip.split()
        if 1 <= len(words) <= 8:
            # check if it looks like a heading
            if line_strip[0].isupper():
                return line_strip
    return "Unknown Section"


# ---------------------------------------------------------
# CHUNKER
# ---------------------------------------------------------
def chunk_text(text: str, page: int, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> List[Dict]:
    """
    Split text into word-based chunks with overlap.
    Store metadata: page, section_hint, chunk_id
    """
    words = text.split()
    chunks = []
    start = 0

    section_hint = extract_section_hint(text)

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk_text_str = " ".join(chunk_words)

        chunk_id = f"p{page}_s{start}"

        chunks.append({
            "page": page,
            "text": chunk_text_str,
            "chunk_id": chunk_id,
            "section_hint": section_hint
        })

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# LOAD PDF PAGE-BY-PAGE
# ---------------------------------------------------------
def load_pdf(pdf_path: str) -> List[Dict]:
    print(f"Loading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        pages.append({
            "page": i,
            "text": raw_text
        })

    print(f"Loaded {len(pages)} pages.")
    return pages


# ---------------------------------------------------------
# CREATE INDEX
# ---------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray):
    """
    Build a cosine-similarity FAISS index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product for cosine similarity (after normalization)
    return index


# ---------------------------------------------------------
# MAIN INGEST PIPELINE
# ---------------------------------------------------------
def main():
    # Ensure index folder exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 1. Load PDF pages
    pages = load_pdf(PDF_PATH)

    # 2. Chunk everything
    all_chunks = []
    for p in pages:
        chunks = chunk_text(p["text"], p["page"])
        all_chunks.extend(chunks)

    print(f"Generated {len(all_chunks)} chunks.")

    # 3. Embed chunks
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]

    print("Embedding chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # 4. Build FAISS index
    index = build_faiss_index(embeddings)
    index.add(embeddings)

    # 5. Save index
    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    faiss.write_index(index, faiss_path)
    print(f"FAISS index saved → {faiss_path}")

    # 6. Save metadata
    metadata = {
        "chunks": all_chunks,           # text, page, chunk_id, section_hint
        "embeddings_shape": embeddings.shape,
        "model_name": EMBED_MODEL
    }

    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Metadata saved → {meta_path}")

    print("\n✅ Ingestion complete!")
    print(f"Total chunks stored: {len(all_chunks)}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
