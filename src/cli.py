"""
CLI-интерфейс для инференса рекомендаций.

Использование:
    python -m src.cli recommend-session --session-id 42
    python -m src.cli recommend-recipe --recipe "0:49,1:80,2:50,3:40,4:63,5:50"
    python -m src.cli list-sessions --limit 10
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(
    name="parfume-recommend",
    help="🌸 Помощник выбора на полке — CLI рекомендаций по 'Парфюмерному органу'",
    add_completion=False,
)
console = Console()

_prs = None


def _get_prs(data_dir: Optional[str] = None):
    global _prs
    if _prs is None:
        from src.recommender import PerfumeRecommendationSystem
        console.print("[cyan]Загружаю данные...[/cyan]")
        _prs = PerfumeRecommendationSystem(data_dir=data_dir)
        _prs.load()
        console.print("[green]✓ Система инициализирована[/green]")
    return _prs


def _print_recs(recs: list[dict], title: str = "Рекомендации"):
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("perfume_id", style="cyan", width=10)
    table.add_column("Бренд", width=20)
    table.add_column("Название", width=30)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Топ-нота (вклад)", width=30)

    for i, r in enumerate(recs, 1):
        exp = r.get("explanation") or {}
        top_note = ""
        if exp:
            best = max(exp, key=lambda k: exp[k])
            top_note = f"{best}: {exp[best]:.4f}"
        table.add_row(
            str(i),
            str(r["perfume_id"]),
            r.get("brand", "")[:20],
            r.get("name", "")[:30],
            f"{r['score']:.4f}",
            top_note,
        )

    console.print(table)
    if recs:
        console.print(f"[dim]Latency: {recs[0].get('latency_ms', '?')} ms[/dim]")


@app.command("recommend-session")
def recommend_session(
    session_id: int = typer.Option(..., "--session-id", "-s", help="ID сессии"),
    top_n: int = typer.Option(10, "--top-n", "-n", help="Количество рекомендаций"),
    alpha: float = typer.Option(0.7, "--alpha", "-a", help="Вес рецепта (0-1)"),
    no_presses: bool = typer.Option(False, "--no-presses", help="Не учитывать нажатия"),
    no_explain: bool = typer.Option(False, "--no-explain", help="Без объяснений"),
    json_out: bool = typer.Option(False, "--json", help="Вывод в JSON"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Путь к данным"),
):
    """Рекомендации по ID сессии из базы Парфюмерного органа."""
    prs = _get_prs(data_dir)
    recs = prs.recommend_by_session(
        session_id=session_id,
        top_n=top_n,
        alpha=alpha,
        use_presses=not no_presses,
        explain=not no_explain,
    )
    if json_out:
        typer.echo(json.dumps(recs, ensure_ascii=False, indent=2))
    else:
        _print_recs(recs, title=f"Top-{top_n} для сессии {session_id}")


@app.command("recommend-recipe")
def recommend_recipe(
    recipe: str = typer.Option(..., "--recipe", "-r", help="Рецепт: '0:49,1:80,2:50,...'"),
    top_n: int = typer.Option(10, "--top-n", "-n"),
    no_explain: bool = typer.Option(False, "--no-explain"),
    json_out: bool = typer.Option(False, "--json"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir"),
):
    """Рекомендации из строки рецепта, без привязки к сессии."""
    prs = _get_prs(data_dir)
    recs = prs.recommend_by_recipe(
        recipe_str=recipe,
        top_n=top_n,
        explain=not no_explain,
    )
    if json_out:
        typer.echo(json.dumps(recs, ensure_ascii=False, indent=2))
    else:
        _print_recs(recs, title=f"Top-{top_n} для рецепта: {recipe}")


@app.command("list-sessions")
def list_sessions(
    limit: int = typer.Option(10, "--limit", "-l"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir"),
):
    """Показывает список сессий из датасета."""
    prs = _get_prs(data_dir)
    df = prs.sessions.head(limit)

    table = Table(title="Сессии органа", box=box.SIMPLE)
    for col in ["session_id", "user_id", "channels_count", "target_perfume_id"]:
        if col in df.columns:
            table.add_column(col)

    for _, row in df.iterrows():
        table.add_row(*[str(row.get(c, "")) for c in ["session_id", "user_id", "channels_count", "target_perfume_id"] if c in df.columns])

    console.print(table)


if __name__ == "__main__":
    app()
