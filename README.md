<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.4-green?style=flat-square" alt="CUDA">
  <img src="https://img.shields.io/badge/license-MIT-purple?style=flat-square" alt="License">
</p>

<h1 align="center">image-to-3d</h1>

<p align="center">
  <strong>Text &rarr; Image &rarr; 3D Model</strong><br>
  <em>Describe an object. Get a textured 3D mesh. View it in your browser.</em>
</p>

<p align="center">
  <code>python generate.py --imagine "a weathered pirate treasure chest"</code>
</p>

---

## Overview

A complete pipeline for generating 3D assets from imagination to `.glb` file:

1. **Imagine** &mdash; Describe any object in natural language
2. **Generate** &mdash; AI creates a detailed image of that object
3. **Reconstruct** &mdash; A second AI converts the image into a textured 3D mesh
4. **Inspect** &mdash; View, rotate, and compare results in a browser-based viewer

Supports 5 text-to-image models and 5 image-to-3D models. Mix and match for different quality/speed tradeoffs. Runs on NVIDIA GPUs (designed for DGX Spark), viewable from any laptop.

---

## Text-to-Image Models

| Model | Steps | VRAM | Notes |
|-------|:-----:|:----:|-------|
| **Playground v2.5** | 30 | 8 GB | Aesthetic-tuned, great for isolated objects |
| **FLUX.1 Schnell** | 4 | 12 GB | Extremely fast, good quality |
| **Z-Image Turbo** | 4 | 16 GB | Alibaba 6B DiT, high fidelity |
| **SDXL** | 30 | 8 GB | Industry standard, very reliable |
| **FLUX.1 Dev** | 28 | 24 GB | Best quality, slower |

## Image-to-3D Models

| Model | Speed | VRAM | Texture Output | Quality |
|-------|:-----:|:----:|----------------|:-------:|
| **Hunyuan3D-2.1** | ~2 min | 10-29 GB | PBR (albedo/metallic/roughness) | Highest |
| **TRELLIS** | 20-60s | 16-24 GB | UV-textured | Top tier |
| **TripoSG** | 10-30s | 6-8 GB | Optional texture bake | Very good |
| **SPAR3D** | 0.7s | 7-10 GB | UV-mapped | Good |
| **TripoSR** | <1s | 4-6 GB | Vertex colors | Preview |

---

## Quick Start

```bash
# 1. Run the interactive setup wizard (on DGX or GPU machine)
python setup.py

# 2. Activate the environment
conda activate image-to-3d

# 3. Start the web server
python server.py
# Open http://localhost:8090
```

Or from your laptop, connect to a remote DGX in one command:

```bash
./connect.sh spark-1     # opens the viewer in your browser automatically
```

---

## Usage

### Text to 3D (end-to-end)

```bash
# Describe an object → get a 3D model
python generate.py --imagine "a red ceramic coffee mug with hand-painted flowers"

# Just generate the image (no 3D)
python generate.py --imagine-only "a steampunk pocket watch"

# Choose your image model
python generate.py --imagine "a crystal dragon" --imagine-model flux-dev
```

### Image to 3D

```bash
python generate.py photo.png                         # default (hunyuan3d)
python generate.py photo.png -m triposr              # fast preview
python generate.py photo.png -m trellis              # top quality
python generate.py photo.png --shape-only --low-vram # minimal VRAM
```

### Compare & Batch

```bash
python generate.py photo.png --compare           # run all models on one image
python generate.py input/ --batch                # process a whole folder
python generate.py input/ --batch -m triposr     # batch with specific model
```

### List Available Models

```bash
python generate.py --list-models
```

---

## Web UI

The browser interface (`python server.py` or `./connect.sh`) provides:

**Generate Panel**
- Text prompt with model/steps controls
- Image upload with drag-and-drop
- "Also generate 3D" toggle for text-to-image
- Live status and image preview during generation

**3D Viewer**
- Drag-and-drop `.glb` loading
- View modes: default, wireframe, normals, matcap
- HDR environment lighting (Poly Haven)
- 4 lighting presets: studio, outdoor, neutral, dramatic
- Background: dark, gradient, light, transparent, HDR
- Turntable auto-rotation
- Screenshot export (PNG)
- Bounding box overlay with dimensions
- Mesh stats: vertices, triangles, materials, file size

**Gallery**
- Auto-populated output gallery
- Click to load any previous generation

---

## Remote Workflow (DGX + Laptop)

Generate on GPU hardware, view locally in your browser:

```bash
# From your laptop:
./connect.sh spark-1          # SSH host or user@ip

# What happens:
#   1. Starts server.py on the remote if not running
#   2. Forwards port 8090 to localhost
#   3. Opens http://localhost:8090 in your browser
#   Ctrl-C to disconnect

# Stop the remote server:
./disconnect.sh spark-1
```

The connect script is self-healing &mdash; it will offer to install Miniconda, clone the repo, and run setup if anything is missing on the remote.

Environment variable overrides:
```bash
REMOTE_HOST=dgx-2 REMOTE_DIR=~/code/image-to-3d LOCAL_PORT=9090 ./connect.sh
```

---

## API Reference

All endpoints are served at `http://localhost:8090`.

### Text-to-Image

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/imagine/models` | List text-to-image models |
| POST | `/api/imagine` | Generate image from prompt (async) |
| GET | `/api/imagine/image/{job_id}` | Serve generated image |

### Image-to-3D

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List 3D model backends |
| POST | `/api/generate` | Upload image, start 3D generation (async) |
| GET | `/api/outputs` | List all generated .glb files |
| GET | `/api/output/{filename}` | Download a .glb file |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/jobs` | List all jobs |
| GET | `/api/jobs/{job_id}` | Poll job status |

**Job status flow:**
```
queued → generating_image → generating_3d → complete
                                          → failed
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | hunyuan3d | Image-to-3D model backend |
| `-o, --output` | output/{name}.glb | Output path |
| `--imagine PROMPT` | &mdash; | Generate image from text, then 3D |
| `--imagine-only PROMPT` | &mdash; | Generate image only |
| `--imagine-model` | playground | Text-to-image model |
| `--shape-only` | off | Skip texture generation |
| `--low-vram` | off | CPU offloading |
| `--steps` | 30 | Diffusion steps |
| `--guidance` | 7.5 | Guidance scale |
| `--octree-res` | 256 | Mesh resolution |
| `--seed` | random | Random seed |
| `--tex-res` | 512 | Texture resolution |
| `--no-preprocess` | off | Skip background removal |
| `--compare` | off | Run all models |
| `--batch` | off | Process directory |

---

## Project Structure

```
image-to-3d/
├── generate.py              # CLI entry point
├── text_to_image.py         # Text-to-image backends (FLUX, SDXL, Z-Image, Playground)
├── server.py                # FastAPI server (viewer + API)
├── setup.py                 # Interactive setup wizard
├── connect.sh               # One-command remote connection (laptop → DGX)
├── disconnect.sh            # Stop remote server
├── preprocess.py            # Background removal + centering
├── models/
│   ├── __init__.py          # Model registry with @register decorator
│   ├── base.py              # Abstract base class + dataclasses
│   ├── hunyuan3d.py         # Hunyuan3D-2.1 (shape + PBR texture)
│   ├── triposr.py           # TripoSR (fast, vertex colors)
│   ├── triposg.py           # TripoSG (good geo + optional texture)
│   ├── spar3d.py            # SPAR3D (fast, UV-mapped)
│   └── trellis.py           # TRELLIS (top tier, UV-textured)
├── viewer/
│   └── index.html           # Three.js viewer + generate UI
├── input/                   # Source images (uploaded or generated)
└── output/                  # Generated .glb meshes + .json metadata
```

---

## Requirements

- **GPU:** NVIDIA with CUDA 12.4 (DGX Spark recommended)
- **Python:** 3.10 via conda
- **HuggingFace:** Account + token (for model weight downloads)

The `setup.py` wizard handles everything: conda, dependencies, model repos, C++ extensions, and weight downloads.

---

## Future

- Integrate generated GLBs into [Worldscope](https://github.com/halldm2000/worldscope) via `Cesium.Model.fromGltfAsync` with lat/lon placement
