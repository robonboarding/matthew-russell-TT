"""
FastAPI wrapper for the RAG pipeline.

Exposes /query and /health. Swagger auto-docs at /docs.

Authentication: API-key header. Optional at 2-hour scope (if API_KEY env
var is unset, auth is disabled with a startup warning). In production this
would be replaced with OAuth2 / managed identity and RBAC handled at the
Azure AI Search layer below the API.
"""
from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.generate import generate

logger = logging.getLogger("rag.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_API_KEY = os.getenv("API_KEY")
if not _API_KEY:
    logger.warning("API_KEY is not set — /query is unauthenticated. OK for local dev, NOT for production.")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if _API_KEY is None:
        return  # Auth disabled for local dev
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


app = FastAPI(
    title="Rabobank RAG Assessment API",
    description="RAG over Wikipedia (subprime mortgage crisis) via Azure OpenAI",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[str]
    model: str
    latency_ms: float
    request_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest) -> QueryResponse:
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    logger.info("query request_id=%s question_length=%d", request_id, len(req.question))
    try:
        result = generate(req.question)
    except Exception as e:
        msg = str(e).lower()
        if "content_filter" in msg or "responsibleaipolicyviolation" in msg:
            logger.warning("content_filter tripped request_id=%s", request_id)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Request blocked by Azure content-safety filter",
            )
        logger.exception("generation_failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "query response request_id=%s latency_ms=%.1f chunks=%d "
        "input_tokens=%d output_tokens=%d cost_usd=%.5f",
        request_id, latency_ms, len(result.retrieved_chunks),
        result.input_tokens, result.output_tokens, result.cost_usd,
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        retrieved_chunks=result.retrieved_chunks,
        model=result.model,
        latency_ms=latency_ms,
        request_id=request_id,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )
