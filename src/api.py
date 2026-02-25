"""
REST API на FastAPI для инференса рекомендаций.

Run:
    python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    POST /recommend/session
    POST /recommend/recipe
    GET  /session/{session_id}/info
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import TOP_N_DEFAULT
from .recommender import PerfumeRecommendationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Глобальный объект системы ────────────────────────────────────────────────
_prs: PerfumeRecommendationSystem | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _prs
    logger.info("Инициализация рекомендательной системы...")
    data_dir = os.environ.get("DATA_DIR", None)
    _prs = PerfumeRecommendationSystem(data_dir=data_dir)
    _prs.load()
    logger.info("Система готова.")
    yield
    _prs = None


app = FastAPI(
    title="Parfumery Organ Recommender",
    description="Рекомендательная система «Помощник выбора на полке»",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Схемы запросов/ответов ───────────────────────────────────────────────────

class SessionRequest(BaseModel):
    session_id: int = Field(..., description="ID сессии из organ_sessions")
    top_n: int = Field(TOP_N_DEFAULT, ge=1, le=50)
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Вес рецепта (0..1)")
    use_presses: bool = Field(True, description="Учитывать нажатия")
    explain: bool = Field(True, description="Включить вклад нот в ответ")


class RecipeRequest(BaseModel):
    recipe: str = Field(
        ...,
        description="Рецепт в формате '0:49,1:80,2:50,3:40,4:63,5:50'",
        example="0:49,1:80,2:50,3:40,4:63,5:50",
    )
    top_n: int = Field(TOP_N_DEFAULT, ge=1, le=50)
    explain: bool = Field(True)


class RecommendationItem(BaseModel):
    perfume_id: int
    brand: str
    name: str
    score: float
    all_votes: Optional[int] = None
    explanation: Optional[dict[str, float]] = None
    latency_ms: Optional[float] = None


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]
    total: int


# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Проверка работоспособности")
def health():
    return {"status": "ok", "loaded": _prs is not None and _prs._loaded}


@app.post("/recommend/session", response_model=RecommendResponse, summary="Рекомендации по session_id")
def recommend_by_session(req: SessionRequest):
    if _prs is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    try:
        recs = _prs.recommend_by_session(
            session_id=req.session_id,
            top_n=req.top_n,
            alpha=req.alpha,
            use_presses=req.use_presses,
            explain=req.explain,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RecommendResponse(
        recommendations=[RecommendationItem(**r) for r in recs],
        total=len(recs),
    )


@app.post("/recommend/recipe", response_model=RecommendResponse, summary="Рекомендации по рецепту")
def recommend_by_recipe(req: RecipeRequest):
    if _prs is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    try:
        recs = _prs.recommend_by_recipe(
            recipe_str=req.recipe,
            top_n=req.top_n,
            explain=req.explain,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RecommendResponse(
        recommendations=[RecommendationItem(**r) for r in recs],
        total=len(recs),
    )


@app.get("/session/{session_id}/info", summary="Информация о сессии")
def session_info(session_id: int):
    if _prs is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    sess = _prs.sessions[_prs.sessions["session_id"] == session_id]
    if sess.empty:
        raise HTTPException(status_code=404, detail=f"Сессия {session_id} не найдена")
    row = sess.iloc[0].to_dict()
    # конвертируем numpy типы
    return {k: (int(v) if hasattr(v, "item") else v) for k, v in row.items()}
