"""Generator catalog endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from stl_core.catalog import CATALOG

router = APIRouter(prefix="/generators", tags=["generators"])

AUDIT_FAMILY = "nominal_dp_audit"


class GeneratorOut(BaseModel):
    name: str
    family: str
    queue: str
    cpu_ok: bool
    params: dict[str, Any]
    audit: bool


@router.get("", response_model=list[GeneratorOut])
def list_generators() -> list[GeneratorOut]:
    return [GeneratorOut(name=g.name, family=g.family, queue=g.queue, cpu_ok=g.cpu_ok,
                         params=g.params, audit=g.family == AUDIT_FAMILY)
            for g in CATALOG.values()]
