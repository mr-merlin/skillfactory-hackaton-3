import numpy as np


def hit_at_k(predicted_ids: list[int], target_id: int, k: int) -> float:
    return 1.0 if target_id in predicted_ids[:k] else 0.0


def mrr(predicted_ids: list[int], target_id: int) -> float:
    try:
        return 1.0 / (predicted_ids.index(target_id) + 1)
    except ValueError:
        return 0.0


def ndcg_at_k(predicted_ids: list[int], target_id: int, k: int) -> float:
    rels = [1.0 if pid == target_id else 0.0 for pid in predicted_ids[:k]]
    if not any(r > 0 for r in rels):
        return 0.0
    arr = np.asarray(rels, dtype=np.float64)
    dcg = np.sum(arr / np.log2(np.arange(1, len(arr) + 1, dtype=np.float64) + 1))
    idcg = 1.0 / np.log2(2)  # ideal: target at pos 1
    return float(dcg / idcg)


def compute_metrics_for_session(predicted_ids: list[int], target_id: int, k_values: list[int] = None) -> dict:
    if k_values is None:
        k_values = [5, 10]
    out = {"mrr": mrr(predicted_ids, target_id)}
    for k in k_values:
        out[f"hit@{k}"] = hit_at_k(predicted_ids, target_id, k)
        out[f"ndcg@{k}"] = ndcg_at_k(predicted_ids, target_id, k)
    return out


def coverage(all_recommended_ids: list[int], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    return len(set(all_recommended_ids)) / catalog_size


def diversity_intra_list(predicted_ids: list[int], perfume_vectors: dict, note_to_idx: dict) -> float:
    if len(predicted_ids) < 2:
        return 0.0
    vecs = []
    for pid in predicted_ids:
        if pid not in perfume_vectors:
            continue
        v = perfume_vectors[pid]
        nrm = np.linalg.norm(v)
        vecs.append(v / nrm if nrm > 0 else v)
    if len(vecs) < 2:
        return 0.0
    M = np.array(vecs)
    sim = M @ M.T
    np.fill_diagonal(sim, 0)
    n = len(vecs)
    mean_sim = sim.sum() / (n * (n - 1)) if n > 1 else 0
    return float(1.0 - mean_sim)
