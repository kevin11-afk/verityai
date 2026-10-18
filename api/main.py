"""Standalone ASGI app: API at /api, built React SPA at /product.

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from contextlib import asynccontextmanager

from api.app import create_api, start_warmup
from api.settings import Settings


class SPAStaticFiles(StaticFiles):
    """Serves the built SPA; unknown paths fall back to index.html so client-side
    routes (/product/research, /product/reports/<id>) survive a browser refresh."""

    async def get_response(self, path: str, scope: Scope):
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            return await super().get_response("index.html", scope)
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        if path.startswith("assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response


def mount_frontend(app, static_dir: Path) -> None:
    if (static_dir / "index.html").exists():
        app.mount("/product", SPAStaticFiles(directory=static_dir, html=True), name="product")
    else:
        @app.get("/product/{rest:path}", include_in_schema=False)
        def _not_built(rest: str = ""):
            return JSONResponse({"error": {"code": "frontend_not_built",
                                           "message": "The web app has not been built. Run `npm run build` in web/."}},
                                status_code=503)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    api = create_api(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        start_warmup(api)
        yield

    root = FastAPI(title="VerityAI", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    root.mount("/api", api)
    mount_frontend(root, settings.static_dir)

    @root.get("/", include_in_schema=False)
    def _index():
        return RedirectResponse("/product/")

    return root


app = create_app()
