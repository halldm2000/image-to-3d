#!/usr/bin/env python3
"""API server for browser-based 3D generation.

Provides REST endpoints so the viewer can trigger generation,
list output files, and stream progress.
"""

import asyncio
import json
import os
import subprocess
import sys
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


@app.get("/api/images")
def api_list_images():
    """List image files in the input directory."""
    INPUT_DIR.mkdir(exist_ok=True)
    results = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for img in INPUT_DIR.glob(ext):
            results.append({
                "filename": img.name,
                "modified": datetime.fromtimestamp(img.stat().st_mtime).isoformat(),
                "size_bytes": img.stat().st_size,
            })
    results.sort(key=lambda x: x["modified"], reverse=True)
    return {"images": results}


@app.get("/api/image/{filename}")
def api_get_image(filename: str):
    """Serve an image from the input directory."""
    path = INPUT_DIR / filename
    if not path.exists() or not path.is_relative_to(INPUT_DIR):
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type=f"image/{path.suffix.lstrip('.')}")


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


@app.get("/api/imagine/models")
def api_imagine_models():
    """List available text-to-image models."""
    try:
        from text_to_image import available_models, is_available
        return {
            "available": is_available(),
            "models": available_models(),
        }
    except ImportError:
        return {"available": False, "models": []}


@app.post("/api/imagine")
async def api_imagine(
    prompt: str = Form(...),
    model: str = Form("playground"),
    steps: Optional[int] = Form(None),
    guidance: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    width: int = Form(1024),
    height: int = Form(1024),
    low_vram: bool = Form(False),
    generate_3d: bool = Form(False),
    model_3d: str = Form("hunyuan3d"),
):
    """Generate an image from text. Optionally chain into 3D generation."""
    job_id = str(uuid.uuid4())[:8]
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_image = INPUT_DIR / f"{job_id}_imagined.png"

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "type": "imagine",
        "model": model,
        "prompt": prompt,
        "output_image": str(output_image),
        "created": datetime.now().isoformat(),
        "generate_3d": generate_3d,
        "model_3d": model_3d,
        "error": None,
    }

    asyncio.get_event_loop().run_in_executor(
        None, _run_imagine, job_id, prompt, output_image,
        model, steps, guidance, seed, width, height, low_vram,
        generate_3d, model_3d,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/imagine/image/{job_id}")
def api_imagine_image(job_id: str):
    """Serve a generated image."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    image_path = Path(job.get("output_image", ""))
    if not image_path.exists():
        raise HTTPException(404, "Image not ready")
    return FileResponse(image_path, media_type="image/png")


def _job_log(job_id, msg):
    """Append a timestamped log entry to a job.

    If msg starts with \\r, it replaces the last progress entry instead
    of appending (like a terminal carriage return).
    """
    if "logs" not in jobs[job_id]:
        jobs[job_id]["logs"] = []
    replace = msg.startswith("\r")
    if replace:
        msg = msg[1:]
    entry = {"t": datetime.now().isoformat(), "msg": msg}
    if replace:
        entry["r"] = True
        for i in range(len(jobs[job_id]["logs"]) - 1, -1, -1):
            if jobs[job_id]["logs"][i].get("r"):
                jobs[job_id]["logs"][i] = entry
                return
    jobs[job_id]["logs"].append(entry)


def _run_imagine(job_id, prompt, output_image, model_name,
                 steps, guidance, seed, width, height, low_vram,
                 generate_3d, model_3d):
    """Run text-to-image generation in a background thread."""
    jobs[job_id]["status"] = "generating_image"
    jobs[job_id]["started"] = datetime.now().isoformat()
    log = lambda msg: _job_log(job_id, msg)

    try:
        from text_to_image import generate as imagine

        log(f"Starting image generation with {model_name}")

        result = imagine(
            prompt=prompt,
            output_path=output_image,
            model_name=model_name,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            width=width,
            height=height,
            low_vram=low_vram,
            log=log,
        )

        jobs[job_id]["image_result"] = result
        jobs[job_id]["image_ready"] = True

        if not generate_3d:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["completed"] = datetime.now().isoformat()
            return

        log("Freeing image model, loading 3D model...")
        jobs[job_id]["status"] = "generating_3d"
        output_glb = OUTPUT_DIR / f"{job_id}_imagined.glb"
        jobs[job_id]["output"] = str(output_glb)

        from text_to_image import unload as unload_t2i
        unload_t2i()

        _run_generation(
            job_id, output_image, output_glb,
            model_3d, False, 30, 7.5, 256, None, low_vram, True,
        )

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


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
    log = lambda msg: _job_log(job_id, msg)

    if jobs[job_id]["status"] not in ("generating_3d",):
        jobs[job_id]["status"] = "running"
    if "started" not in jobs[job_id]:
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
            log("Preprocessing image (background removal)...")
            try:
                from preprocess import preprocess as pp
                input_path = pp(input_path)
                log("Preprocessing complete")
            except ImportError:
                log("Skipping preprocessing (rembg not installed)")

        log(f"Loading 3D model: {model_name}...")
        m = get_model(model_name)
        from download_progress import track_downloads
        with track_downloads(log):
            m.load(low_vram=low_vram)
        log("Generating 3D mesh...")
        result = m.generate(input_path, output_path, config)
        log(f"3D generation complete ({result.elapsed_seconds:.1f}s)")
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


DOWNLOAD_STATUS_DIR = PROJECT_DIR / ".download_status"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


@app.post("/api/download")
def api_start_download(repo_id: str = Form(...)):
    """Start downloading a model in a detached background process."""
    DOWNLOAD_STATUS_DIR.mkdir(exist_ok=True)
    status_file = DOWNLOAD_STATUS_DIR / f"{repo_id.replace('/', '--')}.json"

    existing = _read_download_status(status_file)
    if existing and existing.get("status") == "downloading":
        pid = existing.get("pid")
        if pid and _pid_alive(pid):
            return {"status": "already_downloading", "repo_id": repo_id}

    status_file.write_text(json.dumps({
        "repo_id": repo_id, "status": "spawning", "started": datetime.now().isoformat(),
    }))

    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "download_model.py"),
         repo_id, "--status-file", str(status_file)],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    status_file.write_text(json.dumps({
        "repo_id": repo_id, "status": "downloading", "pid": proc.pid,
        "started": datetime.now().isoformat(),
    }))

    return {"status": "started", "repo_id": repo_id, "pid": proc.pid}


@app.get("/api/download/status")
def api_download_status(repo_id: str = None):
    """Check download status for a model, including cache progress."""
    if repo_id:
        return _get_download_info(repo_id)

    DOWNLOAD_STATUS_DIR.mkdir(exist_ok=True)
    results = {}
    for f in DOWNLOAD_STATUS_DIR.glob("*.json"):
        info = _read_download_status(f)
        if info:
            results[info.get("repo_id", f.stem)] = info
    return {"downloads": results}


def _get_download_info(repo_id: str) -> dict:
    """Get download progress for a specific repo from cache files."""
    DOWNLOAD_STATUS_DIR.mkdir(exist_ok=True)
    status_file = DOWNLOAD_STATUS_DIR / f"{repo_id.replace('/', '--')}.json"
    proc_status = _read_download_status(status_file) or {}

    repo_dir = HF_CACHE_DIR / f"models--{repo_id.replace('/', '--')}" / "blobs"
    if not repo_dir.exists():
        return {"repo_id": repo_id, "cached": False, **proc_status}

    complete_bytes = 0
    incomplete_bytes = 0
    incomplete_files = []
    for f in repo_dir.iterdir():
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if f.name.endswith(".incomplete"):
            incomplete_bytes += size
            incomplete_files.append({"hash": f.stem[:12], "size": size})
        elif not f.name.startswith("."):
            complete_bytes += size

    return {
        "repo_id": repo_id,
        "cached": len(incomplete_files) == 0 and complete_bytes > 0,
        "complete_bytes": complete_bytes,
        "incomplete_bytes": incomplete_bytes,
        "incomplete_files": len(incomplete_files),
        **proc_status,
    }


def _read_download_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


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
