#!/usr/bin/env python3
"""Generate 3D models (.glb) from 2D images using multiple AI backends."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from models import get_model, list_models, available_models
from models.base import GenerationConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Image → 3D model generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s photo.png                           # default model (hunyuan3d)
  %(prog)s photo.png -m triposr                # fast, low-VRAM option
  %(prog)s photo.png --compare                 # run all available models
  %(prog)s input/ -o output/ --batch           # process a folder of images
  %(prog)s photo.png --shape-only --low-vram   # minimal VRAM usage
  %(prog)s --list-models                       # show available models
        """,
    )
    p.add_argument("image", type=Path, nargs="?",
                   help="Input image or directory (with --batch)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output .glb path or directory (with --batch)")
    p.add_argument("-m", "--model", type=str, default="hunyuan3d",
                   help="Model backend (default: hunyuan3d)")

    modes = p.add_argument_group("modes")
    modes.add_argument("--batch", action="store_true",
                       help="Process all images in a directory")
    modes.add_argument("--compare", action="store_true",
                       help="Run all available models on the same image")
    modes.add_argument("--list-models", action="store_true",
                       help="List available model backends and exit")

    quality = p.add_argument_group("quality")
    quality.add_argument("--shape-only", action="store_true",
                         help="Skip texture generation")
    quality.add_argument("--steps", type=int, default=30,
                         help="Diffusion steps (default: 30)")
    quality.add_argument("--guidance", type=float, default=7.5,
                         help="Guidance scale (default: 7.5)")
    quality.add_argument("--octree-res", type=int, default=256,
                         help="Mesh resolution (default: 256)")
    quality.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")
    quality.add_argument("--tex-res", type=int, default=512,
                         help="Texture resolution (default: 512)")
    quality.add_argument("--views", type=int, default=6,
                         help="Texture viewpoints (default: 6)")

    perf = p.add_argument_group("performance")
    perf.add_argument("--low-vram", action="store_true",
                      help="CPU offloading for lower VRAM usage")
    perf.add_argument("--no-preprocess", action="store_true",
                      help="Skip background removal / centering")

    return p.parse_args()


def print_models():
    models = list_models()
    print(f"\n{'Name':<14} {'VRAM':>6} {'Texture':>9} {'PBR':>5}  {'Status':>10}  Description")
    print("─" * 90)
    for m in models:
        vram = f"{m['min_vram_gb']:.0f} GB"
        tex = "yes" if m["supports_texture"] else "no"
        pbr = "yes" if m["supports_pbr"] else "no"
        avail = "\033[32m ready\033[0m" if m["available"] else "\033[33m not installed\033[0m"
        print(f"{m['name']:<14} {vram:>6} {tex:>9} {pbr:>5}  {avail:>20}  {m['description']}")
    print()


def make_config(args) -> GenerationConfig:
    return GenerationConfig(
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        octree_resolution=args.octree_res,
        texture=not args.shape_only,
        texture_resolution=args.tex_res,
        texture_views=args.views,
        low_vram=args.low_vram,
        remove_background=not args.no_preprocess,
    )


def preprocess_image(image_path: Path, config: GenerationConfig) -> Path:
    if not config.remove_background:
        return image_path
    try:
        from preprocess import preprocess
        print(f"  Preprocessing {image_path.name}...")
        out = preprocess(image_path)
        print(f"  Saved preprocessed: {out}")
        return out
    except ImportError:
        print("  Warning: rembg not installed, skipping preprocessing")
        return image_path


def generate_single(image_path: Path, output_path: Path,
                    model_name: str, config: GenerationConfig):
    """Generate a single 3D model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model:  {model_name}")
    print(f"  Input:  {image_path}")
    print(f"  Output: {output_path}")
    mode = "shape-only" if not config.texture else "shape + texture"
    print(f"  Mode:   {mode}")
    print()

    processed = preprocess_image(image_path, config)

    model = get_model(model_name)
    print(f"  Loading {model_name}...")
    model.load(low_vram=config.low_vram)

    print(f"  Generating...")
    result = model.generate(processed, output_path, config)
    model.unload()

    print(f"\n  {result.summary()}")

    meta_path = output_path.with_suffix(".json")
    meta = {
        "source_image": str(image_path),
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": result.elapsed_seconds,
        "config": {
            "steps": config.steps,
            "guidance_scale": config.guidance_scale,
            "seed": config.seed,
            "octree_resolution": config.octree_resolution,
            "texture": config.texture,
            "texture_resolution": config.texture_resolution,
        },
        "mesh": {
            "vertices": result.vertex_count,
            "faces": result.face_count,
            "has_texture": result.has_texture,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata: {meta_path}")

    return result


def generate_compare(image_path: Path, output_dir: Path, config: GenerationConfig):
    """Run all available models on the same image for comparison."""
    models = available_models()
    if not models:
        print("Error: no models are available. Run setup.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Comparing {len(models)} models: {', '.join(models)}")
    print(f"  Input: {image_path}")
    print(f"  Output dir: {output_dir}")

    results = []
    for name in models:
        out = output_dir / f"{image_path.stem}_{name}.glb"
        try:
            r = generate_single(image_path, out, name, config)
            results.append(r)
        except Exception as e:
            print(f"\n  Error with {name}: {e}")
            results.append(None)

    print("\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)
    for r in results:
        if r:
            print(f"  {r.summary()}")
        else:
            print(f"  (failed)")
    print()


def generate_batch(input_dir: Path, output_dir: Path,
                   model_name: str, config: GenerationConfig):
    """Process all images in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in extensions and not p.stem.endswith("_preprocessed"))

    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Batch: {len(images)} images from {input_dir}")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")

    results = []
    for i, img in enumerate(images, 1):
        print(f"\n{'─' * 40}")
        print(f"  [{i}/{len(images)}] {img.name}")
        out = output_dir / f"{img.stem}.glb"
        try:
            r = generate_single(img, out, model_name, config)
            results.append((img.name, r))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((img.name, None))

    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE: {sum(1 for _, r in results if r)}/{len(results)} succeeded")
    for name, r in results:
        if r:
            print(f"  {name}: {r.elapsed_seconds:.1f}s -> {r.output_path.name}")
        else:
            print(f"  {name}: FAILED")
    print()


def main():
    args = parse_args()

    if args.list_models:
        print_models()
        return

    if not args.image:
        print("Error: image path required (or --list-models)", file=sys.stderr)
        sys.exit(1)

    if not args.image.exists():
        print(f"Error: not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    config = make_config(args)

    if args.compare:
        out_dir = args.output or Path("output/compare")
        generate_compare(args.image, out_dir, config)

    elif args.batch:
        if not args.image.is_dir():
            print("Error: --batch requires a directory path", file=sys.stderr)
            sys.exit(1)
        out_dir = args.output or Path("output/batch")
        generate_batch(args.image, out_dir, args.model, config)

    else:
        out = args.output or Path(f"output/{args.image.stem}.glb")
        generate_single(args.image, out, args.model, config)


if __name__ == "__main__":
    main()
