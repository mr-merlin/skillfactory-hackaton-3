import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.ranking.data import DataLoader
from app.ranking.profile.build_profile import session_to_user_vector, _build_channel_to_aromas
from app.ranking.scoring.score import build_sku_vectors, score_skus


def main():
    settings = get_settings()
    loader = DataLoader(
        perfume_dir=settings.data_perfume_dir,
        organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
    )

    sessions = loader.load_organ_sessions()
    perfume_notes = loader.load_perfume_notes()
    perfumes = loader.load_perfumes()
    recipe_comp = loader.load_organ_recipe_components()
    aromas = loader.load_organ_aromas()

    perfume_vectors, note_to_idx, idx_to_note = build_sku_vectors(perfume_notes)
    popularity = perfumes.set_index("perfume_id")["allVotes"].to_dict() if "allVotes" in perfumes.columns else None

    # 1. Какие SKU чаще всего в Top-5 / Top-10
    print("=" * 60)
    print("1. SKU чаще всего попадающие в Top-5 / Top-10")
    print("=" * 60)
    top5_counter = Counter()
    top10_counter = Counter()

    for _, row in sessions.iterrows():
        sid = int(row["session_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if not uv:
            continue
        recs = score_skus(uv, perfume_vectors, note_to_idx, idx_to_note,
                         top_n=10, popularity=popularity, retrieval_k=100, blend_alpha=0.5)
        for i, (pid, _) in enumerate(recs):
            top10_counter[pid] += 1
            if i < 5:
                top5_counter[pid] += 1

    name_col = next((c for c in perfumes.columns if c in ("name", "perfumeName")), None)
    brand_col = next((c for c in perfumes.columns if c in ("brand", "brandName")), None)
    name_map = perfumes.set_index("perfume_id")[name_col].to_dict() if name_col else {}
    brand_map = perfumes.set_index("perfume_id")[brand_col].to_dict() if brand_col else {}

    print("\nTop-15 SKU в Top-5 рекомендациях:")
    for pid, cnt in top5_counter.most_common(15):
        name = name_map.get(pid, "?")
        brand = brand_map.get(pid, "?")
        print(f"  perfume_id={pid:6d}  cnt={cnt:4d}  {brand} — {name}")

    print("\nTop-15 SKU в Top-10 рекомендациях:")
    for pid, cnt in top10_counter.most_common(15):
        name = name_map.get(pid, "?")
        brand = brand_map.get(pid, "?")
        print(f"  perfume_id={pid:6d}  cnt={cnt:4d}  {brand} — {name}")

    # 2. Какие ноты доминируют в рецептах
    print("\n" + "=" * 60)
    print("2. Доминирующие ноты в рецептах пользователей")
    print("=" * 60)
    note_counter = Counter()
    note_intensity = Counter()

    for _, row in sessions.iterrows():
        sid = int(row["session_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if not uv:
            continue
        for note, val in uv.items():
            note_counter[note] += 1
            note_intensity[note] += val

    print("\nTop-20 нот по частоте появления в профилях:")
    for note, cnt in note_counter.most_common(20):
        avg_int = note_intensity[note] / cnt
        print(f"  {note:30s}  sessions={cnt:4d}  avg_weight={avg_int:.3f}")

    # 3. Какие каналы/аромы приводят к каким SKU
    print("\n" + "=" * 60)
    print("3. Связь каналов с рекомендуемыми SKU")
    print("=" * 60)

    channel_to_aromas = _build_channel_to_aromas(loader)
    aroma_name_map = {}
    if aromas is not None and "aroma_id" in aromas.columns:
        name_col = next((c for c in aromas.columns if c in ("name", "aroma_name", "aromaName")), None)
        if name_col:
            aroma_name_map = aromas.set_index("aroma_id")[name_col].to_dict()

    if recipe_comp is not None:
        top_channel = recipe_comp.groupby("session_id").apply(
            lambda g: g.loc[g["intensity"].idxmax(), "channel_index"]
        ).reset_index(name="top_channel")
        top_channel.columns = ["session_id", "top_channel"]

        merged = sessions.merge(top_channel, on="session_id")
        print("\nСамый интенсивный канал → целевой SKU (Top-5 связей на канал):")
        for ch in sorted(merged["top_channel"].unique()):
            subset = merged[merged["top_channel"] == ch]
            top_targets = subset["target_perfume_id"].value_counts().head(5)
            aroma_ids = channel_to_aromas.get(int(ch), [int(ch)])
            aroma_names = [aroma_name_map.get(a, str(a)) for a in aroma_ids]
            print(f"\n  Канал {ch} ({', '.join(aroma_names)})  [{len(subset)} сессий]:")
            for pid, cnt in top_targets.items():
                name = name_map.get(pid, "?")
                print(f"    → perfume_id={pid}  cnt={cnt}  {name}")

    # 4. Статистика
    print("\n" + "=" * 60)
    print("4. Общая статистика")
    print("=" * 60)
    print(f"  Сессий: {len(sessions)}")
    print(f"  SKU в каталоге: {perfumes['perfume_id'].nunique()}")
    print(f"  Уникальных нот: {len(note_to_idx)}")
    print(f"  Нот в профилях пользователей: {len(note_counter)}")
    print(f"  Уникальных SKU в Top-10 (по всем сессиям): {len(top10_counter)}")
    print(f"  Покрытие каталога рекомендациями: {len(top10_counter) / perfumes['perfume_id'].nunique():.1%}")


if __name__ == "__main__":
    main()
