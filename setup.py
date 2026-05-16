#!/usr/bin/env python3
"""Interactive setup wizard for image-to-3d (multi-model)."""

import os
import subprocess
import shutil
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(PROJECT_DIR, "Hunyuan3D-2.1")
CONDA_ENV = "image-to-3d"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

MODELS = {
    "hunyuan3d": {
        "name": "Hunyuan3D-2.1",
        "vram": "10-29 GB",
        "desc": "Highest quality, PBR textures, shape+texture pipeline",
        "repo": "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git",
        "hf": "tencent/Hunyuan3D-2.1",
        "license_url": "https://huggingface.co/tencent/Hunyuan3D-2.1",
    },
    "triposr": {
        "name": "TripoSR",
        "vram": "4-6 GB",
        "desc": "Fast (<1s), vertex colors, great for quick previews",
        "repo": "https://github.com/VAST-AI-Research/TripoSR.git",
        "hf": "stabilityai/TripoSR",
    },
    "triposg": {
        "name": "TripoSG",
        "vram": "6-8 GB",
        "desc": "Excellent geometry, optional texture bake, 1.5B params",
        "repo": "https://github.com/VAST-AI-Research/TripoSG.git",
        "hf": "VAST-AI/TripoSG",
    },
    "spar3d": {
        "name": "SPAR3D",
        "vram": "7-10 GB",
        "desc": "Fast (0.7s), UV-textured, point-aware (successor to SF3D)",
        "repo": "https://github.com/Stability-AI/stable-point-aware-3d.git",
        "hf": "stabilityai/stable-point-aware-3d",
    },
    "trellis": {
        "name": "TRELLIS",
        "vram": "16-24 GB",
        "desc": "Top quality, textured meshes, Linux only",
        "repo": "https://github.com/microsoft/TRELLIS.git",
        "hf": "JeffreyXiang/TRELLIS-image-large",
    },
}

# ── Colors ────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"


def banner():
    print(f"""
{CYAN}{BOLD}  ╔══════════════════════════════════════════╗
  ║         image-to-3d  setup wizard         ║
  ╚══════════════════════════════════════════╝{RESET}
{DIM}  Multi-model 3D generation from 2D images
  Hunyuan3D | TripoSR | TripoSG | SPAR3D | TRELLIS
  Text-to-image: FLUX | SDXL{RESET}
""")


def step_header(num, total, title):
    bar = f"{BLUE}{'━' * 44}{RESET}"
    print(f"\n{bar}")
    print(f"  {BOLD}Step {num}/{total}{RESET}  {title}")
    print(bar)


def status(icon, msg):
    icons = {
        "ok": f"{GREEN}✓{RESET}",
        "skip": f"{YELLOW}→{RESET}",
        "fail": f"{RED}✗{RESET}",
        "info": f"{CYAN}ℹ{RESET}",
        "wait": f"{MAGENTA}⧖{RESET}",
        "warn": f"{YELLOW}!{RESET}",
    }
    print(f"  {icons.get(icon, '·')} {msg}")


def ask(prompt, default="y"):
    hint = "Y/n" if default == "y" else "y/N"
    try:
        resp = input(f"  {BOLD}?{RESET} {prompt} [{hint}] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not resp:
        return default == "y"
    return resp in ("y", "yes")


def ask_choice(prompt, options):
    """Ask user to pick from a numbered list. Returns list of selected keys."""
    print(f"\n  {BOLD}?{RESET} {prompt}")
    for i, (key, info) in enumerate(options.items(), 1):
        print(f"    {BOLD}{i}{RESET}) {info['name']:<14} {DIM}{info['vram']:>10}{RESET}  {info['desc']}")
    print(f"    {BOLD}a{RESET}) All of the above")
    print()
    try:
        resp = input(f"  Enter choices (e.g. 1,3 or a): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if resp == "a":
        return list(options.keys())
    keys = list(options.keys())
    selected = []
    for part in resp.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(keys):
                selected.append(keys[idx])
    return selected or [keys[0]]


def run(cmd, cwd=None, env=None, capture=False, check=True):
    merged_env = {**os.environ, **(env or {})}
    try:
        if capture:
            r = subprocess.run(cmd, shell=True, cwd=cwd, env=merged_env,
                               capture_output=True, text=True, timeout=600)
            if check and r.returncode != 0:
                return None
            return r.stdout.strip()
        else:
            print(f"  {DIM}$ {cmd}{RESET}")
            r = subprocess.run(cmd, shell=True, cwd=cwd, env=merged_env, timeout=1800)
            return r.returncode == 0
    except subprocess.TimeoutExpired:
        status("fail", "Command timed out")
        return None if capture else False
    except Exception as e:
        status("fail", f"Error: {e}")
        return None if capture else False


def conda_base():
    for path in [
        os.path.expanduser("~/opt/anaconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniconda3"),
        "/opt/conda",
    ]:
        if os.path.isdir(path):
            return path
    out = run("conda info --base 2>/dev/null", capture=True, check=False)
    return out if out and os.path.isdir(out) else None


def conda_run(cmd, cwd=None):
    base = conda_base()
    if not base:
        status("fail", "Cannot find conda installation")
        return False
    activate = f"source {base}/etc/profile.d/conda.sh && conda activate {CONDA_ENV}"
    return run(f"bash -c '{activate} && {cmd}'", cwd=cwd)


def env_exists():
    out = run("conda env list 2>/dev/null", capture=True, check=False)
    return out and CONDA_ENV in out


def check_gpu():
    out = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null",
              capture=True, check=False)
    return out


def install_miniconda():
    """Download and install Miniconda automatically."""
    import platform
    machine = platform.machine()
    system = platform.system()

    if system == "Linux":
        arch = "x86_64" if machine == "x86_64" else "aarch64"
        url = f"https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-{arch}.sh"
    elif system == "Darwin":
        arch = "arm64" if machine == "arm64" else "x86_64"
        url = f"https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-{arch}.sh"
    else:
        status("fail", f"Unsupported OS: {system}")
        return False

    install_path = os.path.expanduser("~/miniconda3")
    installer = "/tmp/miniconda_installer.sh"

    status("wait", f"Downloading Miniconda for {system} {arch}...")
    if shutil.which("wget"):
        dl_ok = run(f"wget -q --show-progress -O {installer} {url}")
    elif shutil.which("curl"):
        dl_ok = run(f"curl -fsSL -o {installer} {url}")
    else:
        status("fail", "Neither wget nor curl found — cannot download")
        return False

    if not dl_ok:
        status("fail", "Download failed")
        return False

    status("wait", f"Installing Miniconda to {install_path}...")
    if not run(f"bash {installer} -b -p {install_path}"):
        status("fail", "Miniconda installation failed")
        return False

    os.remove(installer)

    # Initialize conda for the current shell
    run(f"{install_path}/bin/conda init bash 2>/dev/null", capture=True, check=False)
    run(f"{install_path}/bin/conda init zsh 2>/dev/null", capture=True, check=False)

    status("ok", f"Miniconda installed at {install_path}")
    status("info", "Conda will be available in new shell sessions automatically")
    return True


def install_git():
    """Try to install git via system package manager."""
    import platform
    system = platform.system()

    if system == "Linux":
        if shutil.which("apt-get"):
            status("wait", "Installing git via apt...")
            return run("sudo apt-get update -qq && sudo apt-get install -y -qq git")
        elif shutil.which("dnf"):
            status("wait", "Installing git via dnf...")
            return run("sudo dnf install -y git")
        elif shutil.which("yum"):
            status("wait", "Installing git via yum...")
            return run("sudo yum install -y git")
    elif system == "Darwin":
        status("info", "On macOS, install Xcode command line tools:")
        status("info", "  xcode-select --install")
        return False

    status("fail", "Could not detect package manager")
    return False


# ── Steps ─────────────────────────────────────────────────────────────────

def step_check_system(num, total):
    step_header(num, total, "Check system requirements")

    py_ver = sys.version.split()[0]
    status("ok", f"Python {py_ver}")

    # Git
    if shutil.which("git"):
        status("ok", "git found")
    else:
        status("warn", "git not found")
        if ask("Install git now?"):
            if install_git():
                status("ok", "git installed")
            else:
                status("fail", "Could not install git — install it manually and re-run")
                sys.exit(1)
        else:
            status("fail", "git is required")
            sys.exit(1)

    # Conda
    base = conda_base()
    if base:
        status("ok", f"conda found at {base}")
    else:
        status("warn", "conda not found")
        if ask("Install Miniconda now? (~80 MB download)"):
            if install_miniconda():
                base = conda_base()
                if not base:
                    status("fail", "Miniconda installed but conda not found on PATH")
                    status("info", "Open a new terminal and re-run: python setup.py")
                    sys.exit(1)
            else:
                status("fail", "Miniconda installation failed")
                sys.exit(1)
        else:
            status("fail", "conda is required — install miniconda or anaconda")
            status("info", "https://docs.conda.io/en/latest/miniconda.html")
            sys.exit(1)

    # GPU
    gpu_info = check_gpu()
    if gpu_info:
        for line in gpu_info.strip().split("\n"):
            name, mem = [x.strip() for x in line.split(",")]
            status("ok", f"GPU: {name} ({mem})")
    else:
        status("warn", "No NVIDIA GPU detected (nvidia-smi failed)")
        status("info", "You'll need a GPU to run inference — setup can still proceed")
        if not ask("Continue without GPU?"):
            sys.exit(0)

    status("ok", "System checks passed")


def step_choose_models(num, total):
    step_header(num, total, "Choose model backends")
    print(f"  {DIM}Each model has different quality/speed/VRAM tradeoffs.{RESET}")
    print(f"  {DIM}Hunyuan3D is recommended. You can install more later.{RESET}")

    selected = ask_choice("Which models do you want to install?", MODELS)

    print()
    status("info", f"Selected: {', '.join(MODELS[k]['name'] for k in selected)}")
    return selected


def step_huggingface(num, total, selected_models):
    step_header(num, total, "HuggingFace account & model licenses")

    logged_in = run(
        "python3 -c \"from huggingface_hub import HfApi; print(HfApi().whoami()['name'])\" 2>/dev/null",
        capture=True, check=False
    )

    if logged_in:
        status("ok", f"Logged in as: {logged_in}")
    else:
        status("info", "You need a HuggingFace account to download model weights")
        print()

        license_models = [k for k in selected_models if "license_url" in MODELS[k]]
        if license_models:
            status("info", "Accept model licenses before first run:")
            for k in license_models:
                print(f"     {CYAN}{MODELS[k]['license_url']}{RESET}")
            print()

        status("info", "Create an access token at:")
        print(f"     {CYAN}https://huggingface.co/settings/tokens{RESET}")
        print()

        if shutil.which("huggingface-cli"):
            if ask("Run huggingface-cli login now?"):
                run("huggingface-cli login")
                logged_in = run(
                    "python3 -c \"from huggingface_hub import HfApi; print(HfApi().whoami()['name'])\" 2>/dev/null",
                    capture=True, check=False
                )
                if logged_in:
                    status("ok", f"Logged in as: {logged_in}")
                else:
                    status("warn", "Login may have failed — you can retry later")
            else:
                status("skip", "Skipped — remember to log in before your first generation")
        else:
            status("skip", "huggingface-cli not yet installed (will be available after Step 6)")
            status("info", "You'll be prompted to log in after dependencies are installed")


def step_clone_repos(num, total, selected_models):
    step_header(num, total, "Clone model repositories")

    for key in selected_models:
        model = MODELS[key]
        if key == "hunyuan3d":
            repo_dir = REPO_DIR
        else:
            repo_name = model["repo"].split("/")[-1].replace(".git", "")
            repo_dir = os.path.join(PROJECT_DIR, repo_name)

        if os.path.isdir(os.path.join(repo_dir, ".git")):
            status("ok", f"{model['name']} already cloned")
            continue

        status("wait", f"Cloning {model['name']}...")
        if run(f"git clone {model['repo']} {repo_dir}"):
            status("ok", f"{model['name']} cloned")
        else:
            status("fail", f"Failed to clone {model['name']}")
            if not ask("Continue with other models?"):
                sys.exit(1)


def step_conda_env(num, total):
    step_header(num, total, "Create conda environment")

    if env_exists():
        status("ok", f"Conda env '{CONDA_ENV}' already exists")
        if ask("Recreate from scratch?", default="n"):
            run(f"conda env remove -y -n {CONDA_ENV}")
        else:
            status("skip", "Using existing environment")
            return

    # Accept conda TOS if required (newer conda versions)
    base = conda_base()
    tos_check = run(
        f"bash -c 'source {base}/etc/profile.d/conda.sh && "
        f"conda create --dry-run -n _tos_test python=3.10' 2>&1",
        capture=True, check=False
    )
    if tos_check and "CondaToS" in str(tos_check):
        status("info", "Accepting conda Terms of Service...")
        run(f"bash -c 'source {base}/etc/profile.d/conda.sh && "
            f"conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main' 2>/dev/null",
            capture=True, check=False)
        run(f"bash -c 'source {base}/etc/profile.d/conda.sh && "
            f"conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r' 2>/dev/null",
            capture=True, check=False)
        status("ok", "TOS accepted")

    status("wait", "Creating conda env with Python 3.10...")
    if run(f"bash -c 'source {base}/etc/profile.d/conda.sh && "
           f"conda create -y -n {CONDA_ENV} python=3.10'"):
        status("ok", "Environment created")
    else:
        status("fail", "Failed to create conda env")
        sys.exit(1)


def step_install_deps(num, total, selected_models):
    step_header(num, total, "Install dependencies")

    torch_check = run(
        f"bash -c 'source {conda_base()}/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"python -c \"import torch; print(torch.__version__)\"' 2>/dev/null",
        capture=True, check=False
    )

    if torch_check and "2.5" in torch_check:
        status("ok", f"PyTorch {torch_check} already installed")
        if not ask("Reinstall dependencies?", default="n"):
            status("skip", "Keeping existing packages")
            return

    status("wait", "Installing PyTorch 2.5.1 + CUDA 12.4...")
    if not conda_run(
        "pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
        "--index-url https://download.pytorch.org/whl/cu124"
    ):
        status("fail", "PyTorch installation failed")
        sys.exit(1)
    status("ok", "PyTorch installed")

    # Core deps for the project itself
    status("wait", "Installing core dependencies (trimesh, Pillow, rembg, fastapi)...")
    conda_run("pip install trimesh pillow rembg fastapi uvicorn python-multipart")

    for key in selected_models:
        model = MODELS[key]
        if key == "hunyuan3d":
            req_file = os.path.join(REPO_DIR, "requirements.txt")
        else:
            repo_name = model["repo"].split("/")[-1].replace(".git", "")
            req_file = os.path.join(PROJECT_DIR, repo_name, "requirements.txt")

        if os.path.isfile(req_file):
            status("wait", f"Installing {model['name']} requirements...")
            if conda_run(f"pip install -r {req_file}", cwd=os.path.dirname(req_file)):
                status("ok", f"{model['name']} requirements installed")
            else:
                status("warn", f"Some {model['name']} requirements failed")
                if not ask("Continue?"):
                    sys.exit(1)
        else:
            status("warn", f"No requirements.txt found for {model['name']}")

    # Text-to-image support (FLUX / SDXL)
    print()
    if ask("Install text-to-image support (FLUX, SDXL)?"):
        status("wait", "Installing diffusers, transformers, accelerate...")
        if conda_run("pip install diffusers transformers accelerate sentencepiece"):
            status("ok", "Text-to-image packages installed")
        else:
            status("warn", "Some text-to-image packages failed — optional, can retry later")
    else:
        status("skip", "Skipped text-to-image (can install later: pip install diffusers transformers accelerate)")

    # Prompt for HuggingFace login now that huggingface-hub is installed
    logged_in = run(
        f"bash -c 'source {conda_base()}/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"python -c \"from huggingface_hub import HfApi; print(HfApi().whoami()[\\\"name\\\"])\"' 2>/dev/null",
        capture=True, check=False
    )
    if not logged_in:
        print()
        status("info", "HuggingFace login needed to download model weights")
        if ask("Run huggingface-cli login now?"):
            conda_run("huggingface-cli login")
        else:
            status("skip", "Remember to run: conda activate image-to-3d && huggingface-cli login")


def step_build_extensions(num, total, selected_models):
    if "hunyuan3d" not in selected_models:
        return

    step_header(num, total, "Build C++ extensions (Hunyuan3D)")
    print(f"  {DIM}Requires a C++ compiler and CUDA toolkit on your PATH.{RESET}")
    print()

    rasterizer_dir = os.path.join(REPO_DIR, "hy3dpaint", "custom_rasterizer")
    renderer_dir = os.path.join(REPO_DIR, "hy3dpaint", "DifferentiableRenderer")

    if os.path.isdir(rasterizer_dir):
        status("wait", "Building custom rasterizer...")
        if conda_run("pip install -e .", cwd=rasterizer_dir):
            status("ok", "Custom rasterizer built")
        else:
            status("fail", "Build failed — texture generation may not work")
            if not ask("Continue?"):
                sys.exit(1)

    compile_script = os.path.join(renderer_dir, "compile_mesh_painter.sh")
    if os.path.isfile(compile_script):
        status("wait", "Building differentiable renderer...")
        if conda_run("bash compile_mesh_painter.sh", cwd=renderer_dir):
            status("ok", "Differentiable renderer built")
        else:
            status("fail", "Build failed — shape-only mode will still work")
            if not ask("Continue?"):
                sys.exit(1)


def step_download_esrgan(num, total, selected_models):
    if "hunyuan3d" not in selected_models:
        return

    step_header(num, total, "Download Real-ESRGAN weights")

    esrgan_path = os.path.join(REPO_DIR, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
    if os.path.isfile(esrgan_path):
        size_mb = os.path.getsize(esrgan_path) / (1024 * 1024)
        status("ok", f"Already downloaded ({size_mb:.0f} MB)")
        return

    status("wait", "Downloading Real-ESRGAN weights (~64 MB)...")
    os.makedirs(os.path.dirname(esrgan_path), exist_ok=True)

    if shutil.which("wget"):
        dl_cmd = f"wget -q --show-progress -O {esrgan_path} {ESRGAN_URL}"
    elif shutil.which("curl"):
        dl_cmd = f"curl -L -o {esrgan_path} {ESRGAN_URL}"
    else:
        status("fail", "Neither wget nor curl found")
        return

    if run(dl_cmd):
        status("ok", "Downloaded")
    else:
        status("fail", "Download failed")


def step_verify(num, total, selected_models):
    step_header(num, total, "Verify installation")

    checks = [("Conda environment", env_exists())]

    torch_cuda = run(
        f"bash -c 'source {conda_base()}/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"python -c \"import torch; print(torch.cuda.is_available())\"' 2>/dev/null",
        capture=True, check=False
    )
    checks.append(("PyTorch CUDA", torch_cuda == "True"))

    if "hunyuan3d" in selected_models:
        checks.append(("Hunyuan3D-2.1 repo", os.path.isdir(os.path.join(REPO_DIR, "hy3dshape"))))
        esrgan_path = os.path.join(REPO_DIR, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
        checks.append(("Real-ESRGAN weights", os.path.isfile(esrgan_path)))

    for key in selected_models:
        if key == "hunyuan3d":
            continue
        model = MODELS[key]
        repo_name = model["repo"].split("/")[-1].replace(".git", "")
        repo_dir = os.path.join(PROJECT_DIR, repo_name)
        checks.append((f"{model['name']} repo", os.path.isdir(repo_dir)))

    all_ok = True
    for label, passed in checks:
        if passed:
            status("ok", label)
        else:
            status("fail", label)
            all_ok = False

    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}Setup complete!{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}Setup finished with warnings.{RESET}")

    model_list = ", ".join(MODELS[k]["name"] for k in selected_models)
    print(f"""
  {BOLD}Installed models:{RESET} {model_list}

  {BOLD}Quick start:{RESET}

    {CYAN}conda activate {CONDA_ENV}{RESET}

    {DIM}# Generate with default model (Hunyuan3D){RESET}
    {CYAN}python generate.py input/photo.png -o output/model.glb{RESET}

    {DIM}# Use a specific model{RESET}
    {CYAN}python generate.py input/photo.png -m triposr{RESET}

    {DIM}# Compare all installed models{RESET}
    {CYAN}python generate.py input/photo.png --compare{RESET}

    {DIM}# Generate from text (text → image → 3D){RESET}
    {CYAN}python generate.py --imagine "a wooden treasure chest"{RESET}

    {DIM}# Batch process a folder{RESET}
    {CYAN}python generate.py input/ --batch{RESET}

    {DIM}# Start the web server (viewer + API){RESET}
    {CYAN}python server.py{RESET}

    {DIM}# List available models{RESET}
    {CYAN}python generate.py --list-models{RESET}

    {DIM}# View locally (drag-drop .glb){RESET}
    {CYAN}open viewer/index.html{RESET}
""")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.chdir(PROJECT_DIR)
    banner()

    status("info", f"Project directory: {PROJECT_DIR}")
    print()

    if not ask("Ready to start setup?"):
        print(f"\n  {DIM}Run again with: python setup.py{RESET}\n")
        sys.exit(0)

    total = 9

    step_check_system(1, total)
    selected = step_choose_models(2, total)
    step_huggingface(3, total, selected)
    step_clone_repos(4, total, selected)
    step_conda_env(5, total)
    step_install_deps(6, total, selected)
    step_build_extensions(7, total, selected)
    step_download_esrgan(8, total, selected)
    step_verify(9, total, selected)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Setup interrupted.{RESET} Run {CYAN}python setup.py{RESET} to resume.\n")
        sys.exit(1)
