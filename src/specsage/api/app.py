"""FastAPI service: SSE question answering + static chat UI."""

import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from specsage.agent.pipeline import RagPipeline
from specsage.config import get_settings

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parents[3] / "ui"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="specsage", version="0.1.0")
    app.state.pipeline = pipeline

    def get_pipeline() -> RagPipeline:
        if app.state.pipeline is None:
            app.state.pipeline = RagPipeline.from_settings()
        return app.state.pipeline

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                }
            )
        )
        return response

    @app.get("/health")
    def health() -> dict:
        settings = get_settings()
        return {
            "status": "ok",
            "provider": settings.llm_provider,
            "collection": settings.collection,
        }

    @app.post("/ask")
    async def ask(body: AskRequest) -> StreamingResponse:
        pipe = get_pipeline()

        async def events():
            try:
                async for event in pipe.ask_stream(body.question):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception:
                logger.exception("pipeline error")
                yield f"data: {json.dumps({'type': 'error', 'message': 'internal error'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if UI_DIR.exists():
        app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return app
