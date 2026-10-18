"""FastAPI application. Routes are defined relative to the mount point so the
same app can be mounted at /api by api.main (uvicorn/Docker) or by server.py
(Streamlit Community Cloud's st.App)."""
from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import schemas
from api.errors import ApiError
from api.services import analysis
from api.services.answer import compose_answer
from api.services.evaluation import load_evaluation
from api.services.history import HistoryStore
from api.services.knowledge import KnowledgeService
from api.settings import APP_VERSION, Settings
from src import config

log = logging.getLogger("verityai.api")

_ERR = {code: {"model": schemas.ErrorResponse} for code in (400, 403, 404, 409, 413, 422, 503)}


def _rss_mb() -> Optional[float]:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS and KiB on Linux.
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001
        return None


async def _read_limited(file: UploadFile, limit_bytes: int) -> bytes:
    data = await file.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise ApiError(413, "file_too_large", f"The file exceeds the {limit_bytes // (1024 * 1024)} MB limit.")
    return data


def create_api(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    kb = KnowledgeService(settings)
    history = HistoryStore(settings.history_dir, settings.max_history)

    api = FastAPI(title="VerityAI API", version=APP_VERSION, docs_url="/docs", redoc_url=None,
                  openapi_url="/openapi.json")
    api.state.kb = kb
    api.state.history = history
    api.state.settings = settings

    if settings.cors_origins:
        api.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
                           expose_headers=["Content-Disposition"])

    # ---- error handling ------------------------------------------------------
    @api.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @api.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return JSONResponse({"error": {"code": "validation_error",
                                       "message": f"{where}: {first.get('msg', 'invalid request')}".strip(": ")}},
                            status_code=422)

    @api.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": {"code": "http_error", "message": str(exc.detail)}}, status_code=exc.status_code)

    @api.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse({"error": {"code": "internal_error",
                                       "message": "Something went wrong on the server. Please try again."}},
                            status_code=500)

    # ---- health ---------------------------------------------------------------
    @api.get("/health", response_model=schemas.Health, tags=["system"])
    def health() -> schemas.Health:
        from src.detector import image_detector
        from src.rag import embeddings
        docs, chunks = kb.counts()
        detector_loaded = image_detector._load_pipeline.cache_info().currsize > 0
        embedder_loaded = embeddings._load_model.cache_info().currsize > 0
        return schemas.Health(
            status="ok" if (kb.ready or chunks > 0) else "degraded",
            version=APP_VERSION, python=sys.version.split()[0],
            detector=schemas.ComponentStatus(name=config.DETECTOR_MODEL_NAME, loaded=detector_loaded,
                                             detail=None if detector_loaded else "Loads on first analysis"),
            detector_version_label=config.DETECTOR_VERSION_LABEL,
            thresholds=schemas.Thresholds(ai=config.AI_SUSPECTED_THRESHOLD, real=config.REAL_SUSPECTED_THRESHOLD),
            embedder=schemas.ComponentStatus(name=config.EMBEDDING_MODEL_NAME, loaded=embedder_loaded),
            knowledge_base_documents=docs, knowledge_base_chunks=chunks, knowledge_base_ready=kb.ready,
            persistence_mode=settings.persistence, persistence_note=settings.persistence_note,
            limits=schemas.Limits(
                max_image_mb=config.MAX_IMAGE_SIZE_MB, max_video_mb=config.MAX_VIDEO_SIZE_MB,
                max_video_seconds=config.MAX_VIDEO_DURATION_SECONDS, max_video_frames=config.MAX_VIDEO_FRAMES_SAMPLED,
                max_kb_upload_mb=settings.max_kb_upload_mb,
                image_extensions=sorted(config.ALLOWED_IMAGE_EXTENSIONS),
                video_extensions=sorted(config.ALLOWED_VIDEO_EXTENSIONS),
                document_extensions=[".pdf", ".txt", ".md"],
            ),
            rss_mb=_rss_mb(),
        )

    # ---- analysis -------------------------------------------------------------
    @api.post("/analyze/image", response_model=schemas.Report, responses=_ERR, tags=["analysis"])
    async def analyze_image(file: UploadFile = File(...)):
        data = await _read_limited(file, config.MAX_IMAGE_SIZE_MB * 1024 * 1024)
        report = await run_in_threadpool(analysis.analyze_image_bytes, data, file.filename or "upload", kb)
        history.save(report.model_dump(mode="json"))
        return report

    @api.post("/analyze/video", response_model=schemas.Report, responses=_ERR, tags=["analysis"])
    async def analyze_video(file: UploadFile = File(...)):
        data = await _read_limited(file, config.MAX_VIDEO_SIZE_MB * 1024 * 1024)
        report = await run_in_threadpool(analysis.analyze_video_bytes, data, file.filename or "upload", kb)
        history.save(report.model_dump(mode="json"))
        return report

    # ---- reports --------------------------------------------------------------
    @api.get("/reports", response_model=schemas.ReportList, tags=["reports"])
    def list_reports(q: str = "", limit: int = 100):
        limit = max(1, min(limit, 500))
        needle = q.strip().lower()
        items = history.list()
        rows = [
            schemas.ReportSummary(
                id=r["id"], created_at=r["created_at"], kind=r["kind"], file_name=r["file"]["name"],
                size_bytes=r["file"]["size_bytes"], category=r["result"]["category"],
                detector_version=r["detector"]["version_label"],
            )
            for r in items
            if not needle or needle in r["file"]["name"].lower() or needle in r["result"]["category"].lower()
        ]
        return schemas.ReportList(reports=rows[:limit], total=len(rows), persistence_mode=settings.persistence,
                                  persistence_note=settings.persistence_note)

    def _must_get(report_id: str) -> dict:
        report = history.get(report_id)
        if report is None:
            raise ApiError(404, "not_found", "No saved report with that id (it may have been deleted or the server restarted).")
        return report

    @api.get("/reports/{report_id}", response_model=schemas.Report, responses=_ERR, tags=["reports"])
    def get_report(report_id: str):
        return _must_get(report_id)

    @api.get("/reports/{report_id}/download", responses=_ERR, tags=["reports"])
    def download_report(report_id: str, format: str = "json"):
        report = schemas.Report.model_validate(_must_get(report_id))
        stem = f"verityai_report_{report.id[:8]}"
        if format == "md":
            body, media = analysis.report_to_markdown(report), "text/markdown; charset=utf-8"
        elif format == "json":
            body, media = report.model_dump_json(indent=2), "application/json"
        else:
            raise ApiError(400, "bad_format", "format must be 'json' or 'md'.")
        return Response(body, media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="{stem}.{format}"'})

    @api.delete("/reports/{report_id}", response_model=schemas.KBActionResult, responses=_ERR, tags=["reports"])
    def delete_report(report_id: str):
        _must_get(report_id)
        history.delete(report_id)
        return schemas.KBActionResult(message="Report deleted.", chunks=0)

    # ---- research ---------------------------------------------------------------
    def _passages(question: str, top_k: int):
        passages, diag = kb.retrieve(question, top_k)
        out = [
            schemas.Passage(
                citation_id=i, chunk_id=p.chunk_id, title=p.title, source_file=p.source_file,
                page_number=p.page_number, source_url=p.source_url, text=p.text, vector_rank=p.vector_rank,
                bm25_rank=p.bm25_rank, rrf_score=p.rrf_score, vector_distance=p.vector_distance,
            )
            for i, p in enumerate(passages, start=1)
        ]
        return out, schemas.RetrievalDiag(**diag.__dict__)

    @api.post("/research/ask", response_model=schemas.AskResponse, tags=["research"])
    def ask(body: schemas.AskRequest):
        passages, diag = _passages(body.question, body.top_k)
        if not passages:
            return schemas.AskResponse(
                question=body.question, abstained=True, answer=[], passages=[], diagnostics=diag,
                notice=("The knowledge base does not contain a passage relevant enough to answer this question. "
                        "Try rephrasing, or add source documents in Knowledge."),
            )
        claims = [schemas.Claim(text=t, citation_id=c) for t, c in compose_answer(body.question, passages)]
        notice = None if claims else (
            "Relevant passages were retrieved, but no single sentence matched the question terms closely. "
            "Review the passages below directly."
        )
        return schemas.AskResponse(question=body.question, abstained=False, answer=claims, passages=passages,
                                   diagnostics=diag, notice=notice)

    @api.post("/kb/search", response_model=schemas.AskResponse, tags=["knowledge"])
    def search(body: schemas.SearchRequest):
        passages, diag = _passages(body.query, body.top_k)
        return schemas.AskResponse(
            question=body.query, abstained=not passages, answer=[], passages=passages, diagnostics=diag,
            notice=None if passages else diag.abstain_reason,
        )

    # ---- knowledge base -----------------------------------------------------------
    @api.get("/kb/documents", response_model=schemas.KBList, tags=["knowledge"])
    def kb_list():
        docs, total = kb.list_documents()
        return schemas.KBList(documents=docs, total_chunks=total, persistence_mode=settings.persistence,
                              persistence_note=settings.persistence_note)

    @api.post("/kb/documents", response_model=schemas.KBUploadResult, responses=_ERR, tags=["knowledge"])
    async def kb_upload(file: UploadFile = File(...), source_url: Optional[str] = Form(default=None)):
        data = await _read_limited(file, settings.max_kb_upload_mb * 1024 * 1024)
        status, message, added, doc = await run_in_threadpool(kb.add_document, file.filename or "", data, source_url)
        return schemas.KBUploadResult(status=status, message=message, chunks_added=added, document=doc)  # type: ignore[arg-type]

    @api.delete("/kb/documents/{source_file}", response_model=schemas.KBActionResult, responses=_ERR, tags=["knowledge"])
    def kb_remove(source_file: str, confirm: str = ""):
        if confirm != source_file:
            raise ApiError(400, "confirmation_required",
                           "Removal requires confirm=<exact document name> as a query parameter.")
        removed = kb.remove_document(source_file)
        return schemas.KBActionResult(message=f"Removed '{source_file}'.", chunks=removed)

    @api.post("/kb/reindex", response_model=schemas.KBActionResult, responses=_ERR, tags=["knowledge"])
    def kb_reindex(confirm: bool = False):
        if not confirm:
            raise ApiError(400, "confirmation_required", "Re-indexing rebuilds the whole index; pass confirm=true.")
        total = kb.reindex()
        return schemas.KBActionResult(message="Index rebuilt from all documents.", chunks=total)

    # ---- evaluation -----------------------------------------------------------------
    @api.get("/evaluation", response_model=schemas.EvaluationResponse, tags=["evaluation"])
    def evaluation():
        return schemas.EvaluationResponse(**load_evaluation(settings.eval_dir))

    return api


def start_warmup(api: FastAPI) -> None:
    """Open/seed the knowledge-base index in the background so the first user request
    is not slow. Called by the host app: a mounted sub-app does not run its own
    startup hooks, so api.main / server.py call this from their lifespan."""
    threading.Thread(target=api.state.kb.warm, daemon=True, name="kb-warm").start()
