"""Entry point for Streamlit Community Cloud (free hosting):

    streamlit run product/server.py

Streamlit >= 1.53 can run on Starlette and accept extra routes, so one free Cloud app serves
  /api/*      the FastAPI backend (api.app), and
  /product/   the built React app (product/static),
while Streamlit's own page at "/" is only a small landing page. Nothing here changes behaviour:
the same API also runs standalone with `uvicorn api.main:app` (see Dockerfile).
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starlette.routing import Mount  # noqa: E402
from streamlit.starlette import App  # noqa: E402

from api.app import create_api, start_warmup  # noqa: E402
from api.main import SPAStaticFiles  # noqa: E402
from api.settings import Settings  # noqa: E402

settings = Settings.from_env()
api = create_api(settings)


@asynccontextmanager
async def lifespan(_app):
    start_warmup(api)
    yield {}


routes = [Mount("/api", app=api)]
if (settings.static_dir / "index.html").exists():
    routes.append(Mount("/product", app=SPAStaticFiles(directory=settings.static_dir, html=True), name="product"))

app = App(str(Path(__file__).with_name("landing.py")), routes=routes, lifespan=lifespan)
