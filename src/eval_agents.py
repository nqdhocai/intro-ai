from tqdm import tqdm
import numpy as np
from rich import print 

def train_dqn(env, dqn_agent, test_dataset, k_candidates):
    retriever = env.retriever
    Kmax = env.Kmax

    dqn_recalls = []
    dqn_k_diff = []
    dqn_full = 0

    for query, gold_ids in tqdm(test_dataset):
        query_emb = retriever.embedder.encode_queries([query])[0].cpu().numpy()
        retrieved_docs = retriever.retrieve([query], top_k=Kmax)[0]
        doc_ids = [doc_id for doc_id, _, _ in retrieved_docs]
        gold_set = set(gold_ids)

        best_k = next((i + 1 for i in range(Kmax) if set(doc_ids[:i + 1]) >= gold_set), None)
        if best_k is None:
            best_k = 0 if not set(doc_ids) & gold_set else Kmax

        dqn_action = dqn_agent.select_action(query_emb)
        dqn_k = k_candidates[dqn_action]
        dqn_recall = len(set(doc_ids[:dqn_k]) & gold_set) / max(1, len(gold_set))
        dqn_recalls.append(dqn_recall)
        dqn_k_diff.append(abs(dqn_k - best_k))
        if dqn_recall == 1.0:
            dqn_full += 1

    print(f"[DQN] Recall@k avg     : {np.mean(dqn_recalls):.4f}")
    print(f"[DQN] Full recall rate : {dqn_full / len(test_dataset):.2%}")
    print(f"[DQN] |k - k*| avg      : {np.mean(dqn_k_diff):.2f}")


def train_ppo(env, ppo_agent, test_dataset, k_candidates):
    retriever = env.retriever
    Kmax = env.Kmax

    ppo_recalls = []
    ppo_k_diff = []
    ppo_full = 0

    for query, gold_ids in tqdm(test_dataset):
        query_emb = retriever.embedder.encode_queries([query])[0].cpu().numpy()
        retrieved_docs = retriever.retrieve([query], top_k=Kmax)[0]
        doc_ids = [doc_id for doc_id, _, _ in retrieved_docs]
        gold_set = set(gold_ids)

        best_k = next((i + 1 for i in range(Kmax) if set(doc_ids[:i + 1]) >= gold_set), None)
        if best_k is None:
            best_k = 0 if not set(doc_ids) & gold_set else Kmax

        ppo_action, _, _ = ppo_agent.select_action(query_emb)
        ppo_k = k_candidates[ppo_action]
        ppo_recall = len(set(doc_ids[:ppo_k]) & gold_set) / max(1, len(gold_set))
        ppo_recalls.append(ppo_recall)
        ppo_k_diff.append(abs(ppo_k - best_k))
        if ppo_recall == 1.0:
            ppo_full += 1

    print(f"[PPO] Recall@k avg     : {np.mean(ppo_recalls):.4f}")
    print(f"[PPO] Full recall rate : {ppo_full / len(test_dataset):.2%}")
    print(f"[PPO] |k - k*| avg      : {np.mean(ppo_k_diff):.2f}")

def eval_baselines(env, test_dataset, fixed_k_values=[1, 3, 5, 10]):
    retriever = env.retriever
    Kmax = env.Kmax
    gold_total = len(test_dataset)

    baseline_results = {k: {"recalls": [], "full": 0} for k in fixed_k_values}

    for query, gold_ids in tqdm(test_dataset):
        retrieved_docs = retriever.retrieve([query], top_k=Kmax)[0]
        doc_ids = [doc_id for doc_id, _, _ in retrieved_docs]
        gold_set = set(gold_ids)

        for k in fixed_k_values:
            recall = len(set(doc_ids[:k]) & gold_set) / max(1, len(gold_set))
            baseline_results[k]["recalls"].append(recall)
            if recall == 1.0:
                baseline_results[k]["full"] += 1

    print(f"\n[bold yellow]Fixed Top-K Baseline Evaluation ({gold_total} queries)[/bold yellow]")
    for k in fixed_k_values:
        avg_recall = np.mean(baseline_results[k]["recalls"])
        full_rate = baseline_results[k]["full"] / gold_total
        print(f"Recall@{k} avg: {avg_recall:.4f} | Full recall rate: {full_rate:.2%}")
