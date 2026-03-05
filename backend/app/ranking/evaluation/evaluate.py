import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data import DataLoader
from ..normalize import build_synonym_map
from ..profile.build_profile import session_to_user_vector
from ..scoring.score import build_sku_vectors, score_skus
from ..baseline.baselines import baseline_popular, baseline_overlap, baseline_single_note
from .metrics import (
    compute_metrics_for_session, coverage, diversity_intra_list,
    note_similarity_at_k, weighted_jaccard_at_k,
)

logger = logging.getLogger(__name__)


def _load_nn_scorer(note_to_idx, perfume_vectors):
    model_path = Path(__file__).resolve().parents[3] / "models" / "two_tower_best.pt"
    if not model_path.exists():
        return None
    try:
        from ..nn.two_tower import TwoTowerScorer
        return TwoTowerScorer(model_path, note_to_idx, perfume_vectors)
    except Exception as exc:
        logger.warning("Cannot load two-tower model: %s", exc)
        return None


def _load_sku_meta():
    meta_path = Path(__file__).resolve().parents[3] / "models" / "sku_meta.pkl"
    if not meta_path.exists():
        return None
    try:
        import pickle
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Cannot load sku_meta: %s", exc)
        return None


def _load_gbm_scorer(note_to_idx, idx_to_note, perfume_vectors, popularity,
                     norm_perfume_vectors=None, norm_note_to_idx=None,
                     sku_meta=None):
    model_path = Path(__file__).resolve().parents[3] / "models" / "gbm_ranker.pkl"
    if not model_path.exists():
        return None
    try:
        from ..gbm.ranker import GBMScorer
        return GBMScorer(model_path, note_to_idx, idx_to_note, perfume_vectors, popularity,
                         norm_perfume_vectors=norm_perfume_vectors, norm_note_to_idx=norm_note_to_idx,
                         sku_meta=sku_meta)
    except Exception as exc:
        logger.warning("Cannot load GBM model: %s", exc)
        return None


def _load_cross_encoder_scorer(note_to_idx, perfume_vectors, sku_meta=None):
    model_path = Path(__file__).resolve().parents[3] / "models" / "cross_encoder_best.pt"
    if not model_path.exists():
        return None
    try:
        from ..nn.cross_encoder import CrossEncoderScorer
        return CrossEncoderScorer(model_path, note_to_idx, perfume_vectors, sku_meta=sku_meta)
    except Exception as exc:
        logger.warning("Cannot load cross-encoder model: %s", exc)
        return None


def _load_hybrid_gbm_scorer(note_to_idx, idx_to_note, perfume_vectors, popularity,
                             norm_perfume_vectors=None, norm_note_to_idx=None,
                             sku_meta=None):
    """Load Hybrid GBM: requires both GBM model and synthetic Two-Tower."""
    gbm_path = Path(__file__).resolve().parents[3] / "models" / "hybrid_gbm_ranker.pkl"
    tt_path = Path(__file__).resolve().parents[3] / "models" / "synthetic_two_tower.pt"
    if not gbm_path.exists() or not tt_path.exists():
        return None
    try:
        from ..gbm.hybrid_scorer import HybridGBMScorer
        return HybridGBMScorer(gbm_path, tt_path, note_to_idx, idx_to_note, perfume_vectors,
                                popularity, norm_perfume_vectors=norm_perfume_vectors,
                                norm_note_to_idx=norm_note_to_idx, sku_meta=sku_meta)
    except Exception as exc:
        logger.warning("Cannot load Hybrid GBM: %s", exc)
        return None


def _load_knn_gbm_scorer(note_to_idx, idx_to_note, perfume_vectors, popularity,
                          norm_perfume_vectors=None, norm_note_to_idx=None,
                          sku_meta=None):
    """Load kNN-augmented GBM (best model: 13.5% Hit@10)."""
    model_path = Path(__file__).resolve().parents[3] / "models" / "knn_gbm_ranker.pkl"
    knn_path = Path(__file__).resolve().parents[3] / "models" / "knn_data.pkl"
    if not model_path.exists() or not knn_path.exists():
        return None
    try:
        from ..gbm.knn_scorer import KnnGBMScorer
        return KnnGBMScorer(model_path, knn_path, note_to_idx, idx_to_note, perfume_vectors,
                             popularity, norm_perfume_vectors=norm_perfume_vectors,
                             norm_note_to_idx=norm_note_to_idx, sku_meta=sku_meta)
    except Exception as exc:
        logger.warning("Cannot load kNN-GBM model: %s", exc)
        return None


def run_evaluation(
    loader: DataLoader,
    test_ratio: float = 0.2,
    top_n: int = 10,
    k_values: list[int] = None,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> dict:
    if k_values is None:
        k_values = [5, 10]

    if not loader.has_organ_data():
        return {"error": "Organ data not available (sessions + recipe_components + aroma_notes_map)"}

    sessions = loader.load_organ_sessions()
    perfume_notes = loader.load_perfume_notes()
    perfumes = loader.load_perfumes()
    catalog_size = perfumes["perfume_id"].nunique()

    popularity = None
    if "allVotes" in perfumes.columns:
        popularity = perfumes.set_index("perfume_id")["allVotes"].to_dict()

    sessions_shuffled = sessions.sample(frac=1, random_state=seed)
    n_test = max(1, int(len(sessions_shuffled) * test_ratio))
    test_sessions = sessions_shuffled.iloc[-n_test:]

    perfume_vectors, note_to_idx, idx_to_note = build_sku_vectors(perfume_notes)

    catalog_notes = perfume_notes["note"].astype(str).str.strip().str.lower().unique().tolist()
    syn_map = build_synonym_map(catalog_notes)
    pv_norm, nti_norm, itn_norm = build_sku_vectors(perfume_notes, synonym_map=syn_map)

    pop_recs = [r[0] for r in baseline_popular(perfumes, top_n=top_n)]

    nn_scorer = _load_nn_scorer(nti_norm, pv_norm)
    sku_meta = _load_sku_meta()
    gbm_scorer = _load_gbm_scorer(note_to_idx, idx_to_note, perfume_vectors, popularity,
                                   norm_perfume_vectors=pv_norm, norm_note_to_idx=nti_norm,
                                   sku_meta=sku_meta)
    cross_encoder_scorer = _load_cross_encoder_scorer(note_to_idx, perfume_vectors, sku_meta=sku_meta)
    hybrid_gbm_scorer = _load_hybrid_gbm_scorer(note_to_idx, idx_to_note, perfume_vectors, popularity,
                                                  norm_perfume_vectors=pv_norm, norm_note_to_idx=nti_norm,
                                                  sku_meta=sku_meta)
    knn_gbm_scorer = _load_knn_gbm_scorer(note_to_idx, idx_to_note, perfume_vectors, popularity,
                                            norm_perfume_vectors=pv_norm, norm_note_to_idx=nti_norm,
                                            sku_meta=sku_meta)

    methods = {
        "main": lambda sid: [r[0] for r in score_skus(
            session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
            perfume_vectors, note_to_idx, idx_to_note, top_n=top_n,
            popularity=popularity, retrieval_k=100, blend_alpha=0.5,
        )],
        "popular": lambda sid: pop_recs,
        "overlap": lambda sid: [r[0] for r in baseline_overlap(
            session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
            perfume_vectors, note_to_idx, idx_to_note, top_n=top_n,
        )],
        "single_note": lambda sid: [r[0] for r in baseline_single_note(
            session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
            perfume_notes, top_n=top_n,
        )],
        "cosine_norm": lambda sid: [r[0] for r in score_skus(
            session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
            pv_norm, nti_norm, itn_norm, top_n=top_n,
            popularity=popularity, retrieval_k=100, blend_alpha=0.5,
        )],
    }

    if nn_scorer is not None:
        methods["two_tower"] = lambda sid, sc=nn_scorer: [
            r[0] for r in sc.score(
                session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
                top_n=top_n,
            )
        ]

    if gbm_scorer is not None:
        methods["gbm"] = lambda sid, sc=gbm_scorer: [
            r[0] for r in sc.score(
                session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
                top_n=top_n,
            )
        ]

    if cross_encoder_scorer is not None:
        methods["cross_encoder"] = lambda sid, sc=cross_encoder_scorer: [
            r[0] for r in sc.score(
                session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
                top_n=top_n,
            )
        ]

    if hybrid_gbm_scorer is not None:
        methods["hybrid_gbm"] = lambda sid, sc=hybrid_gbm_scorer: [
            r[0] for r in sc.score(
                session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
                top_n=top_n,
            )
        ]

    if knn_gbm_scorer is not None:
        methods["knn_gbm"] = lambda sid, sc=knn_gbm_scorer: [
            r[0] for r in sc.score(
                session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {},
                top_n=top_n,
            )
        ]

    results = {name: [] for name in methods}
    all_recommended = {name: [] for name in methods}
    diversity_list = {name: [] for name in methods}
    soft_sim = {name: {k: [] for k in k_values} for name in methods}
    soft_jaccard = {name: {k: [] for k in k_values} for name in methods}
    error_counts = {name: 0 for name in methods}

    all_targets = set()
    target_hits = {name: set() for name in methods}

    for _, row in test_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        all_targets.add(target)

        user_vec = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7) or {}

        for name, get_recs in methods.items():
            try:
                pred_ids = get_recs(sid)
            except Exception as exc:
                logger.warning("%s failed on session %d: %s", name, sid, exc)
                error_counts[name] += 1
                pred_ids = []
            results[name].append(compute_metrics_for_session(pred_ids, target, k_values))
            all_recommended[name].extend(pred_ids)
            if target in pred_ids[:max(k_values)]:
                target_hits[name].add(target)
            if pred_ids:
                diversity_list[name].append(diversity_intra_list(pred_ids, perfume_vectors, note_to_idx))
            for k in k_values:
                soft_sim[name][k].append(note_similarity_at_k(user_vec, pred_ids, perfume_vectors, note_to_idx, k))
                soft_jaccard[name][k].append(weighted_jaccard_at_k(user_vec, pred_ids, perfume_vectors, note_to_idx, k))

    summary = {}
    for name, metrics_list in results.items():
        if not metrics_list:
            summary[name] = {}
            continue
        avg = {}
        for key in metrics_list[0]:
            avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        avg["coverage"] = coverage(all_recommended[name], catalog_size)
        avg["diversity"] = (sum(diversity_list[name]) / len(diversity_list[name])) if diversity_list[name] else 0.0
        for k in k_values:
            avg[f"note_sim@{k}"] = float(np.mean(soft_sim[name][k])) if soft_sim[name][k] else 0.0
            avg[f"w_jaccard@{k}"] = float(np.mean(soft_jaccard[name][k])) if soft_jaccard[name][k] else 0.0
        avg["target_coverage"] = len(target_hits[name]) / len(all_targets) if all_targets else 0.0
        avg["errors"] = error_counts[name]
        summary[name] = avg

    summary["n_test_sessions"] = len(test_sessions)
    summary["catalog_size"] = catalog_size
    summary["unique_targets"] = len(all_targets)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
