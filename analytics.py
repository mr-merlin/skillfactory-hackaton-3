"""
Аналитика по каталогу и сессиям (задание B).

Отвечает на два простых вопроса:
  1. Какие SKU чаще всего оказываются в Top-N?
  2. Какие ноты доминируют в рецептах по всем сессиям?
  3. Какие аромы с какими SKU чаще всего попадают в пару?

Запуск:
    python analytics.py
    python analytics.py --top-n 10 --data-dir ./data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.recommender import PerfumeRecommendationSystem
from src.features import build_user_profile


def main():
    parser = argparse.ArgumentParser(description="Аналитика каталога и сессий")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    print("Загружаю данные...")
    prs = PerfumeRecommendationSystem(data_dir=args.data_dir)
    prs.load()

    # ── 1. Топ SKU в выдаче ────────────────────────────────────────────────────
    print(f"\n=== 1. Самые часто рекомендуемые SKU (Top-{args.top_n}) ===")

    sku_counter: Counter = Counter()
    all_sessions = prs.sessions["session_id"].tolist()
    for sid in all_sessions[:500]:  # берём первые 500 для скорости
        recs = prs.recommend_by_session(int(sid), top_n=args.top_n, explain=False)
        for r in recs:
            sku_counter[r["perfume_id"]] += 1

    top_sku = sku_counter.most_common(20)
    df_top_sku = pd.DataFrame(top_sku, columns=["perfume_id", "count"])
    df_top_sku = df_top_sku.merge(
        prs.perfumes[["perfume_id", "brand", "name", "allVotes"]],
        on="perfume_id", how="left"
    )
    print(df_top_sku.to_string(index=False))

    # ── 2. Доминирующие ноты в рецептах ───────────────────────────────────────
    print("\n=== 2. Самые весомые ноты в профилях пользователей ===")

    note_weights: dict[str, float] = {}
    idx2note = {v: k for k, v in prs.note2idx.items()}

    for sid in all_sessions[:500]:
        vec = build_user_profile(
            session_id=int(sid),
            recipe_components=prs.recipe_components,
            presses=prs.presses,
            aroma_notes_map=prs.aroma_notes_map,
            aromas=prs.aromas,
            note2idx=prs.note2idx,
        )
        for idx, w in enumerate(vec):
            if w > 0 and idx in idx2note:
                note_weights[idx2note[idx]] = note_weights.get(idx2note[idx], 0.0) + float(w)

    df_notes = pd.DataFrame(
        sorted(note_weights.items(), key=lambda x: -x[1])[:20],
        columns=["note", "total_weight"]
    )
    df_notes["total_weight"] = df_notes["total_weight"].round(2)
    print(df_notes.to_string(index=False))

    # ── 3. Каналы/аромы → SKU ─────────────────────────────────────────────────
    print("\n=== 3. Аромы органа → с какими SKU чаще всего сочетаются ===")

    aroma_sku: dict[int, Counter] = {}
    rc = prs.recipe_components.copy()
    # Каждый канал имеет несколько аром — собираем все
    ch2aromas: dict[int, list[int]] = {}
    for r in prs.aromas.itertuples(index=False):
        ch2aromas.setdefault(r.channel_index, []).append(r.aroma_id)
    sessions_with_target = prs.sessions.dropna(subset=["target_perfume_id"])

    for _, sess_row in sessions_with_target.head(500).iterrows():
        sid = int(sess_row["session_id"])
        target = int(sess_row["target_perfume_id"])
        sess_rc = rc[rc["session_id"] == sid]
        for r in sess_rc.itertuples():
            for aroma_id in ch2aromas.get(r.channel_index, []):
                if aroma_id not in aroma_sku:
                    aroma_sku[aroma_id] = Counter()
                aroma_sku[aroma_id][target] += int(r.intensity)

    rows = []
    aromas_info = prs.aromas.set_index("aroma_id")
    for aroma_id, sku_cnt in sorted(aroma_sku.items()):
        top_sku_for_aroma = sku_cnt.most_common(3)
        aroma_name = ""
        if aroma_id in aromas_info.index:
            row = aromas_info.loc[aroma_id]
            # Реальное имя колонки в organ_aromas
            for col in ["base_note", "name", "aroma_name", "note", "label"]:
                if col in row.index and pd.notna(row[col]):
                    aroma_name = str(row[col])
                    break
        rows.append({
            "aroma_id": aroma_id,
            "aroma_name": aroma_name,
            "top_skus": str([s for s, _ in top_sku_for_aroma]),
        })

    df_aroma_sku = pd.DataFrame(rows)
    print(df_aroma_sku.to_string(index=False))

    # ── Сохранение отчёта ─────────────────────────────────────────────────────
    out_path = Path("analytics_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Аналитический отчёт\n\n")
        f.write(f"## Топ SKU в Top-{args.top_n} (по 500 сессиям)\n\n")
        f.write(df_top_sku.to_markdown(index=False))
        f.write("\n\n## Доминирующие ноты в рецептах\n\n")
        f.write(df_notes.to_markdown(index=False))
        f.write("\n\n## Аромы органа → SKU\n\n")
        f.write(df_aroma_sku.to_markdown(index=False))
    print(f"\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
