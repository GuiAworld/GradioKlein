#!/usr/bin/env python3
"""
Instalador do ambiente GradioKlein 9B (stable-diffusion.cpp pré-compilado).

- Modo `debug` (--mode debug / --debug):
  Instala apenas dependências Python e os 4 modelos YOLO de detecção.
  NÃO baixa nem instala stable-diffusion.cpp nem os modelos GGUF 9B.

- Modo `prod` (--mode prod, padrão):
  Instala dependências Python, os 4 modelos YOLO, os binários pré-compilados
  do stable-diffusion.cpp (sem compilar nada!) e os arquivos GGUF/VAE de menor
  uso de VRAM do modelo placetrtr/PornMaster-flux2-klein-9b.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HF_MODEL_REPO = "placetrtr/PornMaster-flux2-klein-9b"
HF_MODEL_FILES = {
    "unet": "unet/turbo/PornMaster_Flux2Klein_v4_Turbo_Q4_K.gguf",
    "llm": "text_encoders/Qwen3-VL-8B-Q4_K_S.gguf",
    "vae": "vae/full_encoder_small_decoder.safetensors",
}

YOLO_MODELS: List[Tuple[str, str, int]] = [
    (
        "yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        2_000_000,
    ),
    (
        "yolov8n-face-lindevs.pt",
        "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
        2_000_000,
    ),
    (
        "yoloboop.pt",
        "https://huggingface.co/Anzhc/Anzhcs_YOLOs/resolve/main/Anzhcs%20Breast%20size%20det%20cls%20v8%20640%20y11m.pt?download=true",
        5_000_000,
    ),
    (
        "yolov8m-pose.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt",
        10_000_000,
    ),
]

GHCR_REPO = "leejet/stable-diffusion.cpp"
GHCR_TAG = "master-cuda"
GITHUB_RELEASES_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases"
FALLBACK_RELEASE_TAG = "master-1104-5dcfa79"


def log(msg: str) -> None:
    print(msg, flush=True)


def download_file(url: str, dest: Path, headers: Optional[Dict[str, str]] = None, min_bytes: int = 1024) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log(f"  [OK] Já existe: {dest.name} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")
        return dest

    req_headers = {"User-Agent": "GradioKlein-Installer/1.0"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    log(f"  [BAIXANDO] {dest.name} <- {url}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        last_pct = -10
        with open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int((downloaded * 100) / total)
                    if pct >= last_pct + 20:
                        last_pct = pct
                        log(f"    ... {pct}% ({downloaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB)")

    if tmp_path.stat().st_size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download incompleto ou inválido para {dest.name}")

    shutil.move(str(tmp_path), str(dest))
    log(f"  [CONCLUÍDO] {dest.name} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")
    return dest


def install_pip_requirements(repo_dir: Path) -> None:
    req_file = repo_dir / "requirements.txt"
    if not req_file.exists():
        log("[AVISO] requirements.txt não encontrado; pulando pip install.")
        return

    log("[1/4] Instalando dependências Python (requirements.txt)...")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "-r", str(req_file)]
    subprocess.run(cmd, check=True)
    log("  [OK] Dependências Python instaladas.")


def install_yolo_models(yolo_dir: Path) -> None:
    log("[2/4] Verificando/baixando modelos YOLO de detecção...")
    yolo_dir.mkdir(parents=True, exist_ok=True)
    for filename, url, min_size in YOLO_MODELS:
        download_file(url, yolo_dir / filename, min_bytes=min_size)
    log("  [OK] Todos os 4 modelos YOLO estão prontos.")


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict:
    req_headers = {"User-Agent": "GradioKlein-Installer/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_ghcr_cuda_layer() -> Tuple[str, str, Dict[str, str]]:
    """
    Resolve a layer pré-compilada contendo /sd.cpp/bin da imagem oficial
    ghcr.io/leejet/stable-diffusion.cpp:master-cuda sem usar Docker nem compilar.
    """
    token_data = _http_get_json(f"https://ghcr.io/token?scope=repository:{GHCR_REPO}:pull")
    token = token_data["token"]
    accept_manifest = ", ".join(
        [
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        ]
    )
    auth_headers = {"Authorization": f"Bearer {token}", "Accept": accept_manifest}
    manifest = _http_get_json(f"https://ghcr.io/v2/{GHCR_REPO}/manifests/{GHCR_TAG}", headers=auth_headers)

    if "manifests" in manifest:
        amd64_digest = None
        for entry in manifest["manifests"]:
            plat = entry.get("platform", {})
            if plat.get("os") == "linux" and plat.get("architecture") == "amd64":
                amd64_digest = entry["digest"]
                break
        if not amd64_digest:
            raise RuntimeError("Manifesto linux/amd64 não encontrado na imagem GHCR.")
        manifest = _http_get_json(f"https://ghcr.io/v2/{GHCR_REPO}/manifests/{amd64_digest}", headers=auth_headers)

    layers = manifest.get("layers", [])
    if not layers:
        raise RuntimeError("Nenhuma layer encontrada no manifesto GHCR.")

    config_digest = manifest.get("config", {}).get("digest")
    selected_layer = None

    if config_digest:
        cfg = _http_get_json(
            f"https://ghcr.io/v2/{GHCR_REPO}/blobs/{config_digest}",
            headers={"Authorization": f"Bearer {token}"},
        )
        non_empty_history = [h for h in cfg.get("history", []) if not h.get("empty_layer", False)]
        if len(non_empty_history) == len(layers):
            for hist_item, layer_item in zip(non_empty_history, layers):
                created_by = hist_item.get("created_by", "")
                if "/sd.cpp/build/bin" in created_by or "/sd.cpp/bin" in created_by:
                    selected_layer = layer_item
                    break

    if selected_layer is None:
        # Fallback heurístico: a layer de binários compilados costuma ficar entre 50MB e 1.5GB
        # e não é a base CUDA gigante nem o pequeno script entrypoint do final.
        candidates = [l for l in layers if 30 * 1024 * 1024 <= int(l.get("size", 0)) <= 1500 * 1024 * 1024]
        selected_layer = candidates[-1] if candidates else layers[-1]

    blob_digest = selected_layer["digest"]
    blob_url = f"https://ghcr.io/v2/{GHCR_REPO}/blobs/{blob_digest}"
    return blob_url, selected_layer.get("mediaType", ""), {"Authorization": f"Bearer {token}"}


def _install_sdcpp_from_ghcr_cuda(sdcpp_dir: Path, dry_run: bool = False) -> bool:
    blob_url, media_type, headers = _resolve_ghcr_cuda_layer()
    log(f"  [GHCR] Layer CUDA resolvida: {blob_url} ({media_type})")
    if dry_run:
        return True

    sdcpp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        blob_path = Path(tmpdir) / "sdcpp_bin_layer.tar.gz"
        download_file(blob_url, blob_path, headers=headers, min_bytes=10 * 1024 * 1024)

        extracted_count = 0
        with tarfile.open(blob_path, "r:*") as tf:
            for member in tf.getmembers():
                clean_name = member.name.lstrip("./")
                if not clean_name.startswith("sd.cpp/bin/"):
                    continue
                rel_path = clean_name[len("sd.cpp/bin/") :]
                if not rel_path or member.isdir():
                    continue
                out_path = sdcpp_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if member.issym() or member.islnk():
                    target_name = os.path.basename(member.linkname)
                    if out_path.exists() or out_path.is_symlink():
                        out_path.unlink()
                    os.symlink(target_name, out_path)
                    extracted_count += 1
                    continue

                src_f = tf.extractfile(member)
                if src_f is None:
                    continue
                with src_f, open(out_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                if member.mode & 0o111 or out_path.name in ("sd-server", "sd-cli"):
                    out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                extracted_count += 1

    server_bin = sdcpp_dir / "sd-server"
    if extracted_count == 0 or not server_bin.exists():
        raise RuntimeError("Não foi possível extrair sd-server da layer GHCR.")

    # Validação de execução do binário pré-compilado (checa dependências dinâmicas no Linux)
    env = os.environ.copy()
    cuda_lib = "/usr/local/cuda/lib64"
    env["LD_LIBRARY_PATH"] = f"{sdcpp_dir}:{cuda_lib}:{env.get('LD_LIBRARY_PATH', '')}".strip(":")
    try:
        res = subprocess.run([str(server_bin), "--help"], env=env, capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            raise RuntimeError(f"sd-server --help retornou código {res.returncode}: {res.stderr}")
    except Exception as exc:
        raise RuntimeError(f"Falha ao validar binário CUDA extraído: {exc}") from exc

    log(f"  [OK] Binários CUDA pré-compilados instalados em {sdcpp_dir} ({extracted_count} arquivos).")
    return True


def _pick_release_assets_for_platform() -> List[Dict[str, str]]:
    release_urls = [
        f"{GITHUB_RELEASES_API}/latest",
        f"{GITHUB_RELEASES_API}/tags/{FALLBACK_RELEASE_TAG}",
    ]
    release_data = None
    for url in release_urls:
        try:
            release_data = _http_get_json(url)
            if release_data.get("assets"):
                break
        except Exception:
            continue

    if not release_data or not release_data.get("assets"):
        raise RuntimeError("Não foi possível consultar os releases pré-compilados de leejet/stable-diffusion.cpp.")

    assets = release_data["assets"]
    sys_name = platform.system().lower()
    chosen: List[Dict[str, str]] = []

    if sys_name == "linux":
        for pat in ("bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip", "bin-Linux-Ubuntu", "avx2.zip"):
            for asset in assets:
                name = asset.get("name", "")
                if pat in name and name.endswith(".zip"):
                    chosen.append(asset)
                    return chosen
    elif sys_name == "windows":
        has_nvidia = shutil.which("nvidia-smi") is not None
        target_pat = "bin-win-cuda12-x64.zip" if has_nvidia else "bin-win-vulkan-x64.zip"
        for asset in assets:
            if target_pat in asset.get("name", ""):
                chosen.append(asset)
        if has_nvidia:
            for asset in assets:
                if "cudart-sd-bin-win-cu12-x64.zip" in asset.get("name", ""):
                    chosen.append(asset)
        if chosen:
            return chosen

    raise RuntimeError(f"Nenhum pacote pré-compilado compatível encontrado para {sys_name}.")


def _install_sdcpp_from_github_release(sdcpp_dir: Path, dry_run: bool = False) -> None:
    assets = _pick_release_assets_for_platform()
    log("  [GitHub Release] Pacotes selecionados: " + ", ".join(a["name"] for a in assets))
    if dry_run:
        return

    sdcpp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for asset in assets:
            zip_path = Path(tmpdir) / asset["name"]
            download_file(asset["browser_download_url"], zip_path, min_bytes=100_000)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    base = os.path.basename(info.filename)
                    if not base:
                        continue
                    out_file = sdcpp_dir / base
                    with zf.open(info, "r") as src_f, open(out_file, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                    if platform.system().lower() == "linux" and base in ("sd-server", "sd-cli") or base.endswith(".so"):
                        out_file.chmod(out_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    log(f"  [OK] Binários pré-compilados extraídos em {sdcpp_dir}.")


def install_sdcpp_binaries(sdcpp_dir: Path, dry_run: bool = False) -> None:
    log("[3/4] Instalando binários pré-compilados do stable-diffusion.cpp (sem compilação)...")
    server_exe = sdcpp_dir / ("sd-server.exe" if platform.system().lower() == "windows" else "sd-server")
    if server_exe.exists() and not dry_run:
        log(f"  [OK] {server_exe.name} já presente em {sdcpp_dir}.")
        return

    sys_name = platform.system().lower()
    has_nvidia = shutil.which("nvidia-smi") is not None

    if sys_name == "linux" and has_nvidia:
        try:
            _install_sdcpp_from_ghcr_cuda(sdcpp_dir, dry_run=dry_run)
            return
        except Exception as exc:
            log(f"  [AVISO] Falha no binário GHCR CUDA ({exc}). Usando fallback GitHub Release...")

    _install_sdcpp_from_github_release(sdcpp_dir, dry_run=dry_run)


def install_flux2_klein_9b_models(flux_dir: Path) -> None:
    log(f"[4/4] Baixando modelos FLUX.2 Klein 9B (menor VRAM) de {HF_MODEL_REPO}...")
    from huggingface_hub import hf_hub_download

    flux_dir.mkdir(parents=True, exist_ok=True)
    for role, rel_path in HF_MODEL_FILES.items():
        local_target = flux_dir / rel_path
        if local_target.exists() and local_target.stat().st_size > 10_000_000:
            log(f"  [OK] {role.upper()}: {rel_path} já presente.")
            continue
        log(f"  [HF] Baixando {role.upper()}: {rel_path}...")
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=rel_path,
            local_dir=str(flux_dir),
        )
        log(f"  [OK] {role.upper()} concluído.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instalador do GradioKlein 9B (stable-diffusion.cpp)")
    parser.add_argument(
        "--mode",
        choices=["prod", "debug"],
        default="prod",
        help="prod instala tudo; debug instala apenas pip + YOLO (sem stable-diffusion.cpp e sem GGUFs).",
    )
    parser.add_argument("--debug", action="store_true", help="Atalho para --mode debug.")
    parser.add_argument("--models-dir", default="./models", help="Diretório base de modelos (padrão: ./models).")
    parser.add_argument("--sdcpp-dir", default="./sdcpp_bin", help="Diretório dos binários sd.cpp (padrão: ./sdcpp_bin).")
    parser.add_argument("--skip-pip", action="store_true", help="Pula pip install -r requirements.txt.")
    parser.add_argument(
        "--dry-run-sdcpp",
        action="store_true",
        help="Apenas resolve as URLs dos binários pré-compilados do stable-diffusion.cpp sem baixar.",
    )
    args = parser.parse_args()
    if args.debug:
        args.mode = "debug"
    return args


def main() -> None:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    models_dir = Path(args.models_dir).resolve()
    sdcpp_dir = Path(args.sdcpp_dir).resolve()

    log(f"=== GradioKlein 9B Installer | Modo: {args.mode.upper()} ===")

    if not args.skip_pip:
        install_pip_requirements(repo_dir)
    else:
        log("[1/4] Pulando pip install (--skip-pip).")

    install_yolo_models(models_dir / "yolo")

    if args.mode == "debug":
        log("[MODO DEBUG] Pulando instalação do stable-diffusion.cpp e dos modelos GGUF 9B.")
        log("✅ Instalação do Modo Debug concluída com sucesso!")
        return

    install_sdcpp_binaries(sdcpp_dir, dry_run=args.dry_run_sdcpp)
    if args.dry_run_sdcpp:
        log("[DRY-RUN] Pulando download dos modelos 9B (--dry-run-sdcpp).")
        return

    install_flux2_klein_9b_models(models_dir / "flux2_klein_9b")
    log("✅ Instalação completa concluída com sucesso!")


if __name__ == "__main__":
    main()
