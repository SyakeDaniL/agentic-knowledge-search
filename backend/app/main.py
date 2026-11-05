"""FastAPI のエントリーポイント。"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .agent import KnowledgeAgent
from .config import Settings, get_settings
from .models import AskRequest, AskResponse
from .rate_limit import DailyRateLimiter


@lru_cache
def get_agent() -> KnowledgeAgent:
    from .llm import AzureOpenAILLM
    from .search import AzureSearchRetriever

    settings = get_settings()
    llm = AzureOpenAILLM(settings)
    retriever = AzureSearchRetriever(settings, llm)
    return KnowledgeAgent(
        llm, retriever, max_iterations=settings.max_iterations, top_k=settings.top_k
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    limiter = DailyRateLimiter(settings.daily_request_limit)

    app = FastAPI(title="Agentic Knowledge Search", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ask", response_model=AskResponse)
    def ask(
        body: AskRequest,
        request: Request,
        agent: KnowledgeAgent = Depends(get_agent),
    ) -> AskResponse:
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        client_ip = client_ip or (request.client.host if request.client else "unknown")
        if not limiter.hit(client_ip):
            raise HTTPException(
                status_code=429, detail="本日の利用上限に達しました。明日またお試しください。"
            )
        return agent.ask(body.question, body.mode)

    return app


app = create_app()
