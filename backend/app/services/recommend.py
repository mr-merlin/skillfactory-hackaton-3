import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..core.config import get_settings
from ..ranking.data import DataLoader
from ..ranking.profile.build_profile import (
    session_to_user_vector,
    recipe_string_to_user_vector,
    _build_channel_to_aromas,
)
from ..ranking.normalize import build_synonym_map
from ..ranking.scoring.score import build_sku_vectors, build_sku_matrix, score_skus, score_skus_with_explanation

logger = logging.getLogger(__name__)


class RecommendService:
    def __init__(self):
        self._loader: Optional[DataLoader] = None
        self._perfume_vectors = None
        self._note_to_idx = None
        self._idx_to_note = None
        self._sku_matrix = None
        self._sku_ids = None
        self._channel_to_aromas = None
        self._popularity = None
        self._nn_scorer = None
        self._gbm_scorer = None
        self._cross_encoder_scorer = None
        self._hybrid_gbm_scorer = None
        self._knn_gbm_scorer = None

    def warmup(self):
        if self._loader is not None:
            return
        settings = get_settings()
        self._loader = DataLoader(
            perfume_dir=settings.data_perfume_dir,
            organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
        )
        perfume_notes = self._loader.load_perfume_notes()
        catalog_notes = perfume_notes["note"].astype(str).str.strip().str.lower().unique().tolist()
        syn_map = build_synonym_map(catalog_notes)
        self._perfume_vectors, self._note_to_idx, self._idx_to_note = build_sku_vectors(perfume_notes, synonym_map=syn_map)
        self._sku_matrix, self._sku_ids = build_sku_matrix(self._perfume_vectors)
        self._channel_to_aromas = _build_channel_to_aromas(self._loader)

        raw_pv, raw_nti, raw_itn = build_sku_vectors(perfume_notes)

        perfumes = self._loader.load_perfumes()
        if "allVotes" in perfumes.columns:
            self._popularity = perfumes.set_index("perfume_id")["allVotes"].to_dict()

        models_dir = Path(settings.project_root) / "backend" / "models"

        nn_path = models_dir / "two_tower_best.pt"
        if nn_path.exists():
            try:
                from ..ranking.nn.two_tower import TwoTowerScorer
                self._nn_scorer = TwoTowerScorer(nn_path, self._note_to_idx, self._perfume_vectors)
                logger.info("Two-tower model loaded")
            except Exception as exc:
                logger.warning("Cannot load two-tower model: %s", exc)

        gbm_path = models_dir / "gbm_ranker.pkl"
        if gbm_path.exists():
            try:
                import pickle
                from ..ranking.gbm.ranker import GBMScorer
                sku_meta = None
                meta_path = models_dir / "sku_meta.pkl"
                if meta_path.exists():
                    with open(meta_path, "rb") as f:
                        sku_meta = pickle.load(f)
                self._gbm_scorer = GBMScorer(gbm_path, raw_nti, raw_itn, raw_pv, self._popularity,
                                             norm_perfume_vectors=self._perfume_vectors,
                                             norm_note_to_idx=self._note_to_idx,
                                             sku_meta=sku_meta)
                logger.info("GBM model loaded")
            except Exception as exc:
                logger.warning("Cannot load GBM model: %s", exc)

        # Cross-Encoder
        ce_path = models_dir / "cross_encoder_best.pt"
        if ce_path.exists():
            try:
                from ..ranking.nn.cross_encoder import CrossEncoderScorer
                self._cross_encoder_scorer = CrossEncoderScorer(
                    ce_path, raw_nti, raw_pv, sku_meta=sku_meta)
                logger.info("Cross-Encoder model loaded")
            except Exception as exc:
                logger.warning("Cannot load Cross-Encoder model: %s", exc)

        # Hybrid GBM
        hybrid_gbm_path = models_dir / "hybrid_gbm_ranker.pkl"
        hybrid_tt_path = models_dir / "synthetic_two_tower.pt"
        if hybrid_gbm_path.exists() and hybrid_tt_path.exists():
            try:
                from ..ranking.gbm.hybrid_scorer import HybridGBMScorer
                self._hybrid_gbm_scorer = HybridGBMScorer(
                    hybrid_gbm_path, hybrid_tt_path, raw_nti, raw_itn, raw_pv,
                    self._popularity,
                    norm_perfume_vectors=self._perfume_vectors,
                    norm_note_to_idx=self._note_to_idx,
                    sku_meta=sku_meta)
                logger.info("Hybrid GBM model loaded")
            except Exception as exc:
                logger.warning("Cannot load Hybrid GBM model: %s", exc)

        # kNN-augmented GBM (best model: 15.0% Hit@10)
        knn_gbm_path = models_dir / "knn_gbm_ranker.pkl"
        knn_data_path = models_dir / "knn_data.pkl"
        if knn_gbm_path.exists() and knn_data_path.exists():
            try:
                from ..ranking.gbm.knn_scorer import KnnGBMScorer
                self._knn_gbm_scorer = KnnGBMScorer(
                    knn_gbm_path, knn_data_path, raw_nti, raw_itn, raw_pv,
                    self._popularity,
                    norm_perfume_vectors=self._perfume_vectors,
                    norm_note_to_idx=self._note_to_idx,
                    sku_meta=sku_meta)
                logger.info("kNN-GBM model loaded")
            except Exception as exc:
                logger.warning("Cannot load kNN-GBM model: %s", exc)

    def recommend_by_session(self, session_id: int, top_n: int = 10, with_explanation: bool = True, method: str = "cosine"):
        self.warmup()
        if not self._loader.has_organ_data():
            raise ValueError("Organ data not available; use recommend_by_recipe")
        user_vec = session_to_user_vector(session_id, self._loader)
        if not user_vec:
            return ([], [], []) if with_explanation else ([], [], None)
        return self._score(user_vec, top_n, with_explanation, method)

    def recommend_by_recipe(self, recipe: str, top_n: int = 10, with_explanation: bool = True, method: str = "cosine"):
        self.warmup()
        aroma_map = self._loader.load_organ_aroma_notes_map()
        ch_map = self._channel_to_aromas or {}

        if aroma_map is None or len(aroma_map) == 0:
            aroma_map = self._fallback_aroma_map()
            ch_map = {}

        user_vec = recipe_string_to_user_vector(recipe, aroma_map, channel_to_aromas=ch_map)
        if not user_vec:
            user_vec = recipe_string_to_user_vector(recipe, self._fallback_aroma_map())
        if not user_vec:
            return ([], [], []) if with_explanation else ([], [], None)
        return self._score(user_vec, top_n, with_explanation, method)

    def _fallback_aroma_map(self, n_channels: int = 6) -> pd.DataFrame:
        notes_df = self._loader.load_perfume_notes()
        top_notes = notes_df.groupby("note")["votes"].sum().sort_values(ascending=False).head(n_channels).index
        return pd.DataFrame([
            {"aroma_id": i, "note": str(n).strip().lower(), "weight": 1.0}
            for i, n in enumerate(top_notes)
        ])

    def _score(self, user_vec: dict, top_n: int, with_explanation: bool, method: str = "cosine"):
        if method == "nn" and self._nn_scorer is not None:
            recs = self._nn_scorer.score(user_vec, top_n=top_n)
            ids = [r[0] for r in recs]
            scores = [r[1] for r in recs]
            return (ids, scores, [[] for _ in ids]) if with_explanation else (ids, scores, None)

        if method == "gbm" and self._gbm_scorer is not None:
            recs = self._gbm_scorer.score(user_vec, top_n=top_n)
            ids = [r[0] for r in recs]
            scores = [r[1] for r in recs]
            return (ids, scores, [[] for _ in ids]) if with_explanation else (ids, scores, None)

        if method == "cross_encoder" and self._cross_encoder_scorer is not None:
            recs = self._cross_encoder_scorer.score(user_vec, top_n=top_n)
            ids = [r[0] for r in recs]
            scores = [r[1] for r in recs]
            return (ids, scores, [[] for _ in ids]) if with_explanation else (ids, scores, None)

        if method == "hybrid_gbm" and self._hybrid_gbm_scorer is not None:
            recs = self._hybrid_gbm_scorer.score(user_vec, top_n=top_n)
            ids = [r[0] for r in recs]
            scores = [r[1] for r in recs]
            return (ids, scores, [[] for _ in ids]) if with_explanation else (ids, scores, None)

        if method == "knn_gbm" and self._knn_gbm_scorer is not None:
            recs = self._knn_gbm_scorer.score(user_vec, top_n=top_n, explain=with_explanation)
            ids = [r[0] for r in recs]
            scores = [r[1] for r in recs]
            if with_explanation:
                explanations = [r[2] if len(r) > 2 else [] for r in recs]
                return ids, scores, explanations
            return ids, scores, None

        kwargs = dict(
            perfume_vectors=self._perfume_vectors,
            note_to_idx=self._note_to_idx,
            idx_to_note=self._idx_to_note,
            top_n=top_n,
            sku_matrix=self._sku_matrix,
            sku_ids=self._sku_ids,
            popularity=self._popularity,
            retrieval_k=100,
            blend_alpha=0.5,
        )
        if with_explanation:
            recs = score_skus_with_explanation(user_vec, **kwargs)
            return [r[0] for r in recs], [r[1] for r in recs], [r[2] for r in recs]
        else:
            recs = score_skus(user_vec, **kwargs)
            return [r[0] for r in recs], [r[1] for r in recs], None
