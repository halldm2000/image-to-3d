#!/usr/bin/env python3
"""API server for browser-based 3D generation.

Provides REST endpoints so the viewer can trigger generation,
list output files, and stream progress.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_DIR = PROJECT_DIR / "input"

app = FastAPI(title="image-to-3d", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict] = {}


@app.get("/api/models")
def api_list_models():
    from models import list_models
    return {"models": list_models()}


@app.get("/api/outputs")
def api_list_outputs():
    """List all generated .glb files with metadata."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []
    for glb in sorted(OUTPUT_DIR.glob("**/*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_path = glb.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        results.append({
            "filename": glb.name,
            "path": str(glb.relative_to(PROJECT_DIR)),
            "size_bytes": glb.stat().st_size,
            "modified": datetime.fromtimestamp(glb.stat().st_mtime).isoformat(),
            "metadata": meta,
        })
    return {"outputs": results}


@app.get("/api/output/{filename}")
def api_get_output(filename: str):
    """Serve a generated .glb file."""
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_relative_to(OUTPUT_DIR):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=filename)


@app.get("/api/output/{subdir}/{filename}")
def api_get_output_subdir(subdir: str, filename: str):
    """Serve a generated .glb file from a subdirectory."""
    path = OUTPUT_DIR / subdir / filename
    if not path.exists() or not path.is_relative_to(OUTPUT_DIR):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=filename)


@app.post("/api/generate")
async def api_generate(
    image: UploadFile = File(...),
    model: str = Form("hunyuan3d"),
    shape_only: bool = Form(False),
    steps: int = Form(30),
    guidance: float = Form(7.5),
    octree_res: int = Form(256),
    seed: Optional[int] = Form(None),
    low_vram: bool = Form(False),
    preprocess: bool = Form(True),
):
    """Upload an image and start 3D generation. Returns a job ID."""
    job_id = str(uuid.uuid4())[:8]

    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    input_path = INPUT_DIR / f"{job_id}_{image.filename}"
    with open(input_path, "wb") as f:
        content = await image.read()
        f.write(content)

    output_path = OUTPUT_DIR / f"{job_id}_{Path(image.filename).stem}.glb"

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "model": model,
        "input": str(input_path),
        "output": str(output_path),
        "created": datetime.now().isoformat(),
        "error": None,
    }

    asyncio.get_event_loop().run_in_executor(
        None, _run_generation, job_id, input_path, output_path,
        model, shape_only, steps, guidance, octree_res, seed, low_vram, preprocess,
    )

    return {"job_id": job_id, "status": "queued"}


def _run_generation(job_id, input_path, output_path,
                    model_name, shape_only, steps, guidance,
                    octree_res, seed, low_vram, do_preprocess):
    """Run generation in a background thread."""
    import time

    jobs[job_id]["status"] = "running"
    jobs[job_id]["started"] = datetime.now().isoformat()

    try:
        from models import get_model
        from models.base import GenerationConfig

        config = GenerationConfig(
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            octree_resolution=octree_res,
            texture=not shape_only,
            low_vram=low_vram,
            remove_background=do_preprocess,
        )

        if do_preprocess:
            try:
                from preprocess import preprocess as pp
                input_path = pp(input_path)
            except ImportError:
                pass

        m = get_model(model_name)
        m.load(low_vram=low_vram)
        result = m.generate(input_path, output_path, config)
        m.unload()

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["completed"] = datetime.now().isoformat()
        jobs[job_id]["result"] = {
            "elapsed_seconds": result.elapsed_seconds,
            "vertex_count": result.vertex_count,
            "face_count": result.face_count,
            "has_texture": result.has_texture,
            "output_file": output_path.name,
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/jobs")
def api_list_jobs():
    return {"jobs": list(jobs.values())}


# Serve the viewer at /
app.mount("/", StaticFiles(directory=str(PROJECT_DIR / "viewer"), html=True), name="viewer")


if __name__ == "__main__":
    import uvicorn
    print(f"\n  image-to-3d server")
    print(f"  Viewer:  http://localhost:8090")
    print(f"  API:     http://localhost:8090/api/models")
    print(f"  Output:  {OUTPUT_DIR}\n")
    uvicorn.run(app, host="0.0.0.0", port=8090)
