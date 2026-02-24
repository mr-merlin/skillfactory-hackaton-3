import numpy as np
import pandas as pd


def baseline_popular(perfumes: pd.DataFrame, top_n: int = 10, popularity_col: str = "allVotes") -> list[tuple[int, float]]:
    if popularity_col not in perfumes.columns:
        raise ValueError(f"Column '{popularity_col}' not in perfumes")
    df = perfumes[["perfume_id", popularity_col]].drop_duplicates(subset=["perfume_id"], keep="first")
    df = df.sort_values(popularity_col, ascending=False).head(top_n)
    return [(int(r["perfume_id"]), float(r[popularity_col])) for _, r in df.iterrows()]


def baseline_overlap(
    user_vec: dict[str, float],
    perfume_vectors: dict[int, np.ndarray],
    note_to_idx: dict[str, int],
    idx_to_note: list[str],
    top_n: int = 10,
    normalize_vectors: bool = True,
) -> list[tuple[int, float]]:
    size = len(note_to_idx)
    u = np.zeros(size, dtype=np.float64)
    for note, val in user_vec.items():
        idx = note_to_idx.get(note.strip().lower())
        if idx is not None:
            u[idx] = val

    if normalize_vectors and np.linalg.norm(u) > 0:
        u = u / np.linalg.norm(u)

    scores = []
    for pid, v in perfume_vectors.items():
        v_n = v / np.linalg.norm(v) if normalize_vectors and np.linalg.norm(v) > 0 else v
        sc = float(np.sum(np.minimum(u, v_n)))
        scores.append((pid, sc))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def baseline_single_note(
    user_vec: dict[str, float],
    perfume_notes: pd.DataFrame,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    if not user_vec:
        return []
    best_note = max(user_vec, key=user_vec.get)
    sub = perfume_notes[perfume_notes["note"].astype(str).str.strip().str.lower() == best_note.strip().lower()]
    if sub.empty:
        return []
    agg = sub.groupby("perfume_id")["votes"].sum().sort_values(ascending=False).head(top_n)
    return [(int(pid), float(agg[pid])) for pid in agg.index]
