"""FastAPI HTTP microservice for the TraceCraft Vectorization Engine."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import vectorize_image

app = FastAPI(title="TraceCraft Vectorization Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "tracecraft-v3-color9-vtracer"}


@app.post("/vectorize")
async def vectorize_endpoint(
    file: UploadFile = File(...),
    mode: Literal["auto", "binary", "color"] = Form("auto"),
    include_bg: bool = Form(True),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    suffix = os.path.splitext(file.filename or "input.png")[1] or ".png"
    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f_in:
            f_in.write(raw)
            in_path = f_in.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as f_out:
            out_path = f_out.name

        t0 = time.perf_counter()
        stats = vectorize_image(in_path, out_path, mode=mode, include_bg=include_bg)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        with open(out_path, "r", encoding="utf-8") as f:
            svg = f.read()

        return {
            "svg": svg,
            "stats": {**stats, "elapsedMs": elapsed_ms},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
