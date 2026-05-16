# image-to-3d

Generate 3D models (.glb) from 2D images using multiple AI backends, then inspect and compare results in a browser-based viewer.

## Supported models

| Model | VRAM | Speed | Texture | Quality | License |
|-------|------|-------|---------|---------|---------|
| **Hunyuan3D-2.1** | 10-29 GB | ~2 min | PBR (albedo/metallic/roughness) | Highest | Tencent |
| **TripoSR** | 4-6 GB | <1s | Vertex colors | Preview | MIT |
| **TripoSG** | 6-8 GB | 10-30s | Optional bake | Very good | MIT |
| **SPAR3D** | 7-10 GB | 0.7s | UV-mapped | Good | Stability AI |
| **TRELLIS** | 16-24 GB | 20-60s | UV-textured | Top tier | MIT |

## Quick start

```bash
# 1. Run the interactive setup wizard (on the DGX)
python setup.py

# 2. Activate the environment
conda activate image-to-3d

# 3. Generate a 3D model
python generate.py input/photo.png

# 4. View the result
open viewer/index.html   # drag-drop the .glb
```

## Usage

### Generate a model

```bash
python generate.py photo.png                            # default (hunyuan3d)
python generate.py photo.png -m triposr                 # fast preview
python generate.py photo.png -m triposg                 # good quality, moderate VRAM
python generate.py photo.png -m spar3d                  # fast with textures
python generate.py photo.png --shape-only --low-vram    # minimal VRAM
```

### Compare models

```bash
python generate.py photo.png --compare     # runs all installed models
```

Outputs each model's result to `output/compare/` with timing and mesh stats.

### Batch process

```bash
python generate.py input/ --batch                    # all images, default model
python generate.py input/ --batch -m triposr         # all images, specific model
```

### List models

```bash
python generate.py --list-models
```

### Web server (viewer + generation API)

```bash
python server.py
# Open http://localhost:8090
```

The server provides:
- `/` — 3D model viewer with gallery
- `/api/models` — list available backends
- `/api/generate` — upload image + generate (async)
- `/api/outputs` — list generated models
- `/api/jobs/{id}` — check generation status

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | hunyuan3d | Model backend |
| `-o, --output` | output/{name}.glb | Output path |
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

## Viewer features

- Drag-and-drop .glb loading
- View modes: default, wireframe, normals, matcap
- HDR environment lighting (Poly Haven)
- Lighting presets: studio, outdoor, neutral, dramatic
- Background: dark, light, gradient, transparent, HDR
- Turntable auto-rotation
- Screenshot export (PNG)
- Bounding box overlay with dimensions
- Mesh stats (vertices, triangles, materials, file size)
- Output gallery (when running with server.py)
- Settings: exposure, tone mapping, grid

## Project structure

```
image-to-3d/
├── generate.py              # CLI: image -> .glb (multi-model)
├── server.py                # FastAPI server (viewer + generation API)
├── setup.py                 # Interactive setup wizard
├── preprocess.py            # Background removal, centering, padding
├── models/
│   ├── __init__.py          # Model registry
│   ├── base.py              # Abstract base + GenerationResult
│   ├── hunyuan3d.py         # Hunyuan3D-2.1 backend
│   ├── triposr.py           # TripoSR backend
│   ├── triposg.py           # TripoSG backend
│   ├── spar3d.py            # SPAR3D backend
│   └── trellis.py           # TRELLIS backend
├── viewer/
│   └── index.html           # Three.js 3D viewer
├── scripts/
│   └── setup.sh             # Non-interactive setup (legacy)
├── input/                   # Source images
└── output/                  # Generated .glb + .json metadata
```

## Requirements

- NVIDIA GPU with CUDA 12.4
- Python 3.10, conda
- HuggingFace account (for model weight downloads)

## Future

- Integrate generated GLBs into Worldscope via `Cesium.Model.fromGltfAsync` with lat/lon placement
