from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api import recommend_router
from app.services.recommend import RecommendService


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = RecommendService()
    svc.warmup()
    app.state.recommend_service = svc
    yield
    app.state.recommend_service = None


app = FastAPI(title="Shelf Helper API", version="0.1.0", lifespan=lifespan)
app.include_router(recommend_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
