from typing import List, Tuple
import gym
import numpy as np
from src.enviroment.retriever import FaissRetriever
from ..processing_data import train_dataset

class RAGTopKEnv(gym.Env):
    def __init__(
        self,
        dataset: List[Tuple[str, List[int]]],
        retriever: FaissRetriever,
        k_candidates: List[int] = [0, 1, 3, 5, 7, 9],
        max_docs: int = 10
    ):
        self.dataset = dataset
        self.retriever = retriever
        self.k_candidates = k_candidates
        self.Kmax = max_docs

        self.action_space = gym.spaces.Discrete(len(k_candidates))
        self.state_dim = retriever.embedder.embedding_size   # query_emb + doc_feats_agg
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.idx = -1
        self.query = None
        self.gold_ids = None
        self.query_emb = None
        self.retrieved_docs = None
        self.state = None
        self.rewards = None

    def reset(self):
        self.idx = (self.idx + 1) % len(self.dataset)
        self.query, self.gold_ids = self.dataset[self.idx]
        self.query_emb = self.retriever.embedder.encode_queries([self.query])[0].cpu().numpy()

        results = self.retriever.retrieve([self.query], top_k=self.Kmax)[0]  # List[(id, doc_text, score)]
        self.retrieved_docs = results
        retrived_doc_ids = [i[0] for i in results]
        self.rewards = self._compute_penalized_rewards(retrived_doc_ids, self.gold_ids, self.k_candidates)

        # Compute features per doc: score, rank, sim
        doc_feats = self._compute_doc_features(results)
        doc_feats_agg = np.mean(doc_feats, axis=0)  # mean over docs

        # self.state = np.concatenate([self.query_emb, doc_feats_agg], axis=0).astype(np.float32)
        self.state = self.query_emb
        return self.state

    def step(self, action_idx):
        k = self.k_candidates[action_idx]
        top_k = self.retrieved_docs[:k]
        predicted_ids = set(str(i) for i, _, _ in top_k)
        gold_ids = set([str(i) for i in self.gold_ids])

        reward = self.rewards[action_idx]
        done = True

        return self.state, reward, done, {"k": k, "gold_ids": gold_ids, "predicted_ids": predicted_ids}

    def _compute_doc_features(self, docs):
        scores = np.array([s for (_, _, s) in docs]).reshape(-1, 1)
        ranks = np.arange(1, len(docs) + 1).reshape(-1, 1) / self.Kmax
        # Cosine sim ~ score nếu FAISS normalize sẵn, có thể bỏ qua hoặc giữ nguyên
        return np.concatenate([scores, ranks], axis=1)  # shape (K, 2)

    def _compute_penalized_rewards(
        self,
        retrieved_ids: List[int],
        gold_ids: List[int],
        k_candidates: List[int],
        penalty_lambda: float = 0.1
    ) -> List[float]:
        """
        Tính reward vector theo công thức:
        - Nếu recall@k == 1.0 và k = k*, reward = 1.0
        - Nếu k > k*, reward = 1.0 - λ * (k - k*)
        - Nếu k < k*, reward = recall@k
    
        Parameters:
            retrieved_ids: List các doc ID truy xuất (đã sắp xếp theo score)
            gold_ids: List doc ID ground-truth
            k_candidates: List các giá trị k cần đánh giá
            penalty_lambda: hệ số phạt khi chọn k > k*
    
        Returns:
            List[float]: reward ứng với từng k trong k_candidates
        """
        gold_set = set(gold_ids)
        rewards = []
        final_rewards = []
        k_star = None  # k* nhỏ nhất đạt full recall

        if len(set(retrieved_ids) & gold_set) == 0:
            final_rewards = [1.0]
            
            for i in range(1, len(k_candidates)):
                final_rewards.append(max(0.0, round(1.0 - penalty_lambda * i, 2)))
            return final_rewards
    
        for k in k_candidates:
            D_k = set(retrieved_ids[:k])
            recall_k = len(D_k & gold_set) / max(1, len(gold_set))
            rewards.append(recall_k)
            if recall_k == 1.0 and k_star is None:
                k_star = k
        if k_star is None:
            k_star = k_candidates[-1]
            
        for i, k in enumerate(k_candidates):
            if k == 0:
                final_rewards.append(0)
                continue 
            if k_star is None:
                final_rewards.append(rewards[i])  # fallback: dùng recall
            elif k == k_star:
                final_rewards.append(1.0)
            elif k > k_star:
                penalty = penalty_lambda * (k - k_star)
                final_rewards.append(max(0.0, round(1.0 - penalty, 2)))
            else:  # k < k*
                final_rewards.append(rewards[i])
    
        return final_rewards

# env = RAGTopKEnv(train_dataset, retrievers["bge-m3"]) # retrievers["e5-small"]
# state = env.reset()
# print(env.rewards)
# state, reward, done, info = env.step(action_idx=5)  # thử k = 5
# reward, done, info