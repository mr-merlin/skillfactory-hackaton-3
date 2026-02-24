import sys
import random
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.ranking.data import DataLoader
from app.ranking.profile.build_profile import session_to_user_vector
from app.ranking.scoring.score import build_sku_vectors
from app.ranking.gbm.ranker import extract_features, FEATURE_NAMES


def build_dataset(loader, note_to_idx, idx_to_note, perfume_vectors, popularity, seed=42, neg_per_pos=15):
    sessions = loader.load_organ_sessions()
    shuffled = sessions.sample(frac=1, random_state=seed)
    n_test = max(1, int(len(shuffled) * 0.2))
    train_sessions = shuffled.iloc[:-n_test]

    catalog = sorted(perfume_vectors.keys())
    max_pop = max(popularity.values()) if popularity else 1.0

    X, y, groups = [], [], []

    for _, row in train_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        if target not in perfume_vectors:
            continue

        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if not uv:
            continue

        pos_f = extract_features(uv, perfume_vectors[target], note_to_idx, idx_to_note,
                                 popularity.get(target, 0) / max_pop)
        X.append(pos_f)
        y.append(1)

        negs = [p for p in catalog if p != target]
        chosen = random.sample(negs, min(neg_per_pos, len(negs)))
        for neg_pid in chosen:
            neg_f = extract_features(uv, perfume_vectors[neg_pid], note_to_idx, idx_to_note,
                                     popularity.get(neg_pid, 0) / max_pop)
            X.append(neg_f)
            y.append(0)

        groups.append(1 + len(chosen))

    return np.vstack(X), np.array(y, dtype=np.float32), groups


def train():
    settings = get_settings()
    loader = DataLoader(
        perfume_dir=settings.data_perfume_dir,
        organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
    )

    perfume_notes = loader.load_perfume_notes()
    perfume_vectors, note_to_idx, idx_to_note = build_sku_vectors(perfume_notes)

    perfumes = loader.load_perfumes()
    popularity = {}
    if "allVotes" in perfumes.columns:
        popularity = perfumes.set_index("perfume_id")["allVotes"].to_dict()

    random.seed(42)
    X, y, groups = build_dataset(loader, note_to_idx, idx_to_note, perfume_vectors, popularity)
    print(f"Сэмплов: {len(y)}, позитивных: {int(y.sum())}, групп: {len(groups)}")

    n_val_groups = max(1, int(len(groups) * 0.2))
    train_groups = groups[:-n_val_groups]
    val_groups = groups[-n_val_groups:]

    train_end = sum(train_groups)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:], y[train_end:]

    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=FEATURE_NAMES)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, feature_name=FEATURE_NAMES, reference=train_ds)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    model = lgb.train(
        params, train_ds,
        num_boost_round=300,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(20),
        ],
    )

    save_path = Path(__file__).parent / "models" / "gbm_ranker.pkl"
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nВажность признаков:")
    imp = model.feature_importance(importance_type="gain")
    for name, val in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1]):
        print(f"  {name:20s} {val:.1f}")

    print(f"\nМодель сохранена: {save_path}")


if __name__ == "__main__":
    train()
