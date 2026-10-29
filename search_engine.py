import io

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    # O modelo lê no máximo ~128 tokens, então os trechos são curtos
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
    return chunks


class SearchEngine:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks: list[dict] = []

    def add_documents(self, docs: dict[str, str]):
        chunks = []
        for source, text in docs.items():
            for piece in chunk_text(text):
                chunks.append({"source": source, "text": piece})
        if not chunks:
            raise ValueError(
                "Não consegui extrair texto. O PDF pode ser escaneado (imagem)."
            )

        embeddings = self.model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        # Embeddings normalizados + produto interno = similaridade de cosseno
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]