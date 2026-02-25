"""
Построение нотного профиля пользователя по рецепту и/или нажатиям.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .config import ALPHA_RECIPE, ALPHA_PRESSES, DECAY_LAMBDA


# ──────────────────────────────────────────────────────────────────────────────
# Профиль из рецепта (organ_recipe_components)
# ──────────────────────────────────────────────────────────────────────────────

def profile_from_recipe_components(
    recipe_rows: pd.DataFrame,
    aroma_notes_map: pd.DataFrame,
    aromas: pd.DataFrame,
    note2idx: dict[str, int],
) -> np.ndarray:
    """
    recipe_rows: строки organ_recipe_components для одной сессии.
    Возвращает вектор профиля пользователя размерности len(note2idx).
    
    Логика:
      canal → aroma_id (через aromas по channel_index)
      aroma_id → note (через aroma_notes_map)
      u[note] += (intensity/100) * weight_mapping
    """
    n = len(note2idx)
    vec = np.zeros(n, dtype=np.float32)

    # channel_index -> list of aroma_ids (один канал может содержать несколько аром)
    ch2aromas: dict[int, list[int]] = {}
    for r in aromas.itertuples(index=False):
        ch2aromas.setdefault(r.channel_index, []).append(r.aroma_id)

    for row in recipe_rows.itertuples(index=False):
        aroma_ids = ch2aromas.get(row.channel_index, [])
        if not aroma_ids:
            continue
        # Распределяем вес канала равномерно по всем аромам канала
        per_aroma_weight = (row.intensity / 100.0) / len(aroma_ids)
        for aroma_id in aroma_ids:
            notes_for_aroma = aroma_notes_map[aroma_notes_map["aroma_id"] == aroma_id]
            for nr in notes_for_aroma.itertuples(index=False):
                idx = note2idx.get(nr.note)
                if idx is not None:
                    vec[idx] += per_aroma_weight * float(nr.weight)

    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Профиль из нажатий (organ_presses)
# ──────────────────────────────────────────────────────────────────────────────

def profile_from_presses(
    press_rows: pd.DataFrame,
    aroma_notes_map: pd.DataFrame,
    note2idx: dict[str, int],
    now_ms: int | None = None,
    decay_lambda: float = DECAY_LAMBDA,
) -> np.ndarray:
    """
    press_rows: строки organ_presses для одной сессии.
    now_ms: временная метка «сейчас» в ms; если None — берём max(started_ms) в сессии.
    
    Логика:
      u[note] += intensity/100 * duration_sec * exp(-λ*age_sec) * weight_mapping
    """
    n = len(note2idx)
    vec = np.zeros(n, dtype=np.float32)
    if press_rows.empty:
        return vec

    if now_ms is None:
        now_ms = int(press_rows["started_ms"].max())

    for row in press_rows.itertuples(index=False):
        age_sec = max(0.0, (now_ms - int(row.started_ms)) / 1000.0)
        weight = (row.intensity_end / 100.0) * (row.duration_ms / 1000.0) * math.exp(-decay_lambda * age_sec)
        notes_for_aroma = aroma_notes_map[aroma_notes_map["aroma_id"] == row.aroma_id]
        for nr in notes_for_aroma.itertuples(index=False):
            idx = note2idx.get(nr.note)
            if idx is not None:
                vec[idx] += weight * float(nr.weight)

    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Комбинированный профиль
# ──────────────────────────────────────────────────────────────────────────────

def build_user_profile(
    session_id: int,
    recipe_components: pd.DataFrame,
    presses: pd.DataFrame,
    aroma_notes_map: pd.DataFrame,
    aromas: pd.DataFrame,
    note2idx: dict[str, int],
    alpha: float = ALPHA_RECIPE,
    use_presses: bool = True,
) -> np.ndarray:
    """
    Строит итоговый вектор пользователя: alpha * u_recipe + (1-alpha) * u_presses.
    Если нажатий нет — используется только рецепт (alpha=1).
    """
    rc = recipe_components[recipe_components["session_id"] == session_id]
    u_recipe = profile_from_recipe_components(rc, aroma_notes_map, aromas, note2idx)

    if use_presses:
        pr = presses[presses["session_id"] == session_id]
        u_presses = profile_from_presses(pr, aroma_notes_map, note2idx)
    else:
        u_presses = np.zeros_like(u_recipe)

    # если рецепт есть — комбинируем
    recipe_norm = np.linalg.norm(u_recipe)
    press_norm = np.linalg.norm(u_presses)

    if recipe_norm > 0 and press_norm > 0 and use_presses:
        profile = alpha * u_recipe + (1 - alpha) * u_presses
    elif recipe_norm > 0:
        profile = u_recipe
    else:
        profile = u_presses

    return profile


# ──────────────────────────────────────────────────────────────────────────────
# Профиль из строки рецепта (для API/CLI)
# ──────────────────────────────────────────────────────────────────────────────

def profile_from_recipe_string(
    recipe_str: str,
    aroma_notes_map: pd.DataFrame,
    aromas: pd.DataFrame,
    note2idx: dict[str, int],
) -> np.ndarray:
    """
    Прямо из строки '0:49,1:80,2:50,...' строит вектор профиля.
    """
    from .data_loader import parse_recipe_string
    ch_intensity = parse_recipe_string(recipe_str)

    rows = [{"channel_index": ch, "intensity": val} for ch, val in ch_intensity.items()]
    recipe_df = pd.DataFrame(rows)
    return profile_from_recipe_components(recipe_df, aroma_notes_map, aromas, note2idx)
