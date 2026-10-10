from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlmodel import Session
from stl_runner.storage import store_from_env

from stl_api.db import get_engine, init_db
from stl_api.dispatch import make_dispatcher
from stl_api.limits import InternalErrorMiddleware, UploadLimitMiddleware
from stl_api.routes import datasets, generators, reports, runs
from stl_api.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(get_engine())
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Synthetic Data Trust Lab API", version="0.1.0", lifespan=lifespan)
    # Starlette: the LAST added middleware is outermost. Order (inner -> outer): error
    # catcher, upload limit, CORS, so every response, including 413/500, carries CORS headers.
    app.add_middleware(InternalErrorMiddleware)
    app.add_middleware(UploadLimitMiddleware, max_mb=settings.max_upload_mb)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.dispatcher = make_dispatcher(settings.dispatch, settings.redis_url)
    app.include_router(datasets.router)
    app.include_router(generators.router)
    app.include_router(runs.router)
    app.include_router(reports.router)

    @app.get("/health")
    def health() -> dict:
        try:
            with Session(get_engine()) as s:
                s.exec(text("SELECT 1"))
            db = True
        except Exception:
            db = False
        try:
            store_from_env().exists("__health__")
            store = True
        except Exception:
            store = False
        return {"status": "ok", "db": db, "store": store}

    return app


app = create_app()
