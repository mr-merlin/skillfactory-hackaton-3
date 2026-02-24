from typing import Optional

import numpy as np
import pandas as pd


def parse_recipe_string(recipe: str) -> list[tuple[int, int]]:
    pairs = []
    for part in recipe.strip().split(","):
        if ":" not in part:
            continue
        ch, val = part.split(":", 1)
        try:
            pairs.append((int(ch.strip()), int(val.strip())))
        except ValueError:
            continue
    return pairs


def _build_channel_to_aromas(loader) -> dict[int, list[int]]:
    aromas_df = loader.load_organ_aromas()
    if aromas_df is not None and {"channel_index", "aroma_id"} <= set(aromas_df.columns):
        return aromas_df.groupby("channel_index")["aroma_id"].apply(list).to_dict()
    return {}


def _expand_channels_to_aromas(recipe_components: pd.DataFrame, channel_to_aromas: dict[int, list[int]]) -> pd.DataFrame:
    rows = []
    for _, row in recipe_components.iterrows():
        ch = int(row["channel_index"])
        intensity = float(row["intensity"])
        for aid in channel_to_aromas.get(ch, [ch]):
            rows.append({"channel_index": ch, "aroma_id": aid, "intensity": intensity})
    return pd.DataFrame(rows, columns=["channel_index", "aroma_id", "intensity"]) if rows else pd.DataFrame(columns=["channel_index", "aroma_id", "intensity"])


def recipe_to_user_vector(
    recipe_components: pd.DataFrame,
    aroma_notes_map: pd.DataFrame,
    *,
    weight_col: str = "weight",
    channel_to_aromas: Optional[dict[int, list[int]]] = None,
) -> dict[str, float]:
    if weight_col not in aroma_notes_map.columns:
        aroma_notes_map = aroma_notes_map.copy()
        aroma_notes_map[weight_col] = 1.0

    if channel_to_aromas:
        comp = _expand_channels_to_aromas(recipe_components, channel_to_aromas)
    elif "aroma_id" in recipe_components.columns:
        comp = recipe_components[["channel_index", "aroma_id", "intensity"]].copy()
        comp["aroma_id"] = comp["aroma_id"].astype(int)
    else:
        comp = recipe_components[["channel_index", "intensity"]].copy()
        comp["aroma_id"] = comp["channel_index"].astype(int)

    comp["intensity_norm"] = comp["intensity"].astype(float) / 100.0

    am = aroma_notes_map[["aroma_id", "note", weight_col]].copy()
    am["aroma_id"] = am["aroma_id"].astype(int)
    merged = comp.merge(am, on="aroma_id", how="inner")
    if merged.empty:
        return {}

    merged["note_lower"] = merged["note"].astype(str).str.strip().str.lower()
    merged["contribution"] = merged["intensity_norm"] * merged[weight_col]
    agg = merged.groupby("note_lower", as_index=False)["contribution"].sum()
    return dict(zip(agg["note_lower"], agg["contribution"].astype(float)))


def recipe_string_to_user_vector(
    recipe: str,
    aroma_notes_map: pd.DataFrame,
    *,
    weight_col: str = "weight",
    channel_to_aromas: Optional[dict[int, list[int]]] = None,
) -> dict[str, float]:
    pairs = parse_recipe_string(recipe)
    if not pairs:
        return {}
    components = pd.DataFrame(pairs, columns=["channel_index", "intensity"])
    return recipe_to_user_vector(components, aroma_notes_map, weight_col=weight_col, channel_to_aromas=channel_to_aromas)


def session_to_user_vector(
    session_id: int,
    loader,
    *,
    use_recipe: bool = True,
    use_presses: bool = True,
    alpha_recipe: float = 0.7,
) -> dict[str, float]:
    aroma_map = loader.load_organ_aroma_notes_map()
    if aroma_map is None or len(aroma_map) == 0:
        return {}

    channel_to_aromas = _build_channel_to_aromas(loader)

    u_recipe: dict[str, float] = {}
    if use_recipe:
        components = loader.load_organ_recipe_components()
        if components is not None:
            sc = components[components["session_id"] == session_id]
            if len(sc) > 0:
                u_recipe = recipe_to_user_vector(sc, aroma_map, channel_to_aromas=channel_to_aromas)

    if not use_presses:
        return u_recipe

    u_presses: dict[str, float] = {}
    presses_df = loader.load_organ_presses()
    if presses_df is not None:
        sp = presses_df[presses_df["session_id"] == session_id]
        if len(sp) > 0:
            am = aroma_map.copy()
            if "weight" not in am.columns:
                am["weight"] = 1.0
            merged = sp.merge(am[["aroma_id", "note", "weight"]], on="aroma_id", how="inner")
            merged["note_lower"] = merged["note"].astype(str).str.strip().str.lower()
            merged["contribution"] = (merged["intensity_end"] / 100.0) * merged["weight"]
            agg = merged.groupby("note_lower", as_index=False)["contribution"].sum()
            u_presses = dict(zip(agg["note_lower"], agg["contribution"].astype(float)))

    if not u_presses:
        return u_recipe

    all_notes = set(u_recipe) | set(u_presses)
    return {
        note: alpha_recipe * u_recipe.get(note, 0) + (1 - alpha_recipe) * u_presses.get(note, 0)
        for note in all_notes
    }
