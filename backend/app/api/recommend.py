from fastapi import APIRouter, Depends, HTTPException, Request

from ..models.schemas import RecommendRequest, RecommendResponse, RecommendItem
from ..services.recommend import RecommendService

router = APIRouter(prefix="/recommend", tags=["recommend"])


def get_service(request: Request) -> RecommendService:
    svc = getattr(request.app.state, "recommend_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return svc


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest, svc: RecommendService = Depends(get_service)):
    try:
        if req.session_id is not None:
            ids, scores, explanations = svc.recommend_by_session(req.session_id, top_n=req.top_n, method=req.method)
        else:
            ids, scores, explanations = svc.recommend_by_recipe(req.recipe.strip(), top_n=req.top_n, method=req.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    items = None
    if explanations:
        items = [
            RecommendItem(perfume_id=pid, score=sc, explanation=exp)
            for pid, sc, exp in zip(ids, scores, explanations)
        ]
    return RecommendResponse(perfume_ids=ids, scores=scores, items=items)
