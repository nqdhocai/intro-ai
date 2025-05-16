from src.enviroment.embedding import EmbeddingModel
import faiss
import numpy as np
from typing import List, Tuple, Union
import json 
import os

class FaissRetriever:
    def __init__(
        self, 
        embedding_model: EmbeddingModel, 
        index_path: str = None
    ):
        self.embedder = embedding_model
        self.index = None
        self.doc_texts = []
        self.doc_ids = []
        self.index_path = index_path

    def build_index(self, docs: List[str], doc_ids: Union[List[str], List[int]] = None):
        print("[*] Encoding documents...")
        embeddings = self.embedder.encode_docs(docs, batch_size=64).cpu().numpy()

        self.doc_texts = docs
        self.doc_ids = doc_ids or list(range(len(docs)))

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Use inner product for cosine sim (with normalized vectors)
        self.index.add(embeddings)
        print(f"[*] FAISS index built with {len(docs)} documents.")

    def save_index(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        meta = {"doc_ids": self.doc_ids, "doc_texts": self.doc_texts}
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("[*] FAISS index and metadata saved.")

    def load_index(self, load_dir: str):
        print("[*] Loading FAISS index and metadata...")
        self.index = faiss.read_index(os.path.join(load_dir, "index.faiss"))
        with open(os.path.join(load_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.doc_ids = meta["doc_ids"]
        self.doc_texts = meta["doc_texts"]
        print(f"[*] Loaded {len(self.doc_texts)} documents.")

    def retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[str, str, float]]]:
        if top_k == 0:
            return []
        # print("[*] Encoding queries...")
        query_emb = self.embedder.encode_queries(queries).cpu().numpy()
        D, I = self.index.search(query_emb, top_k)  # D: scores, I: indices

        results = []
        for scores, indices in zip(D, I):
            hits = [(i, self.doc_texts[i], float(scores[idx])) for idx, i in enumerate(indices)]
            results.append(hits)
        return results
