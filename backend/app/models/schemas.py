from typing import Optional
from pydantic import BaseModel, Field, model_validator


class RecommendRequest(BaseModel):
    session_id: Optional[int] = None
    recipe: Optional[str] = None
    top_n: int = Field(default=10, ge=1, le=100)
    method: str = Field(default="cosine", pattern="^(cosine|nn|gbm|knn_gbm)$")

    @model_validator(mode="after")
    def require_session_or_recipe(self):
        if self.session_id is None and not (self.recipe or "").strip():
            raise ValueError("Provide either session_id or recipe")
        return self


class RecommendItem(BaseModel):
    perfume_id: int
    score: float
    reason: Optional[str] = None
    explanation: Optional[list[dict]] = None


class RecommendResponse(BaseModel):
    perfume_ids: list[int]
    scores: list[float]
    items: Optional[list[RecommendItem]] = None
