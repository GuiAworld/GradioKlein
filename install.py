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


def _resolve_ghcr_cuda_info() -> Tuple[List[Dict], List[Dict], Dict[str, str]]:
    """
    Resolve o manifesto linux/amd64 e histórico de layers da imagem oficial
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

    non_empty_history: List[Dict] = []
    config_digest = manifest.get("config", {}).get("digest")
    if config_digest:
        cfg = _http_get_json(
            f"https://ghcr.io/v2/{GHCR_REPO}/blobs/{config_digest}",
            headers={"Authorization": f"Bearer {token}"},
        )
        non_empty_history = [h for h in cfg.get("history", []) if not h.get("empty_layer", False)]

    return layers, non_empty_history, {"Authorization": f"Bearer {token}"}


def _resolve_ghcr_cuda_layer() -> Tuple[str, str, Dict[str, str]]:
    layers, non_empty_history, blob_headers = _resolve_ghcr_cuda_info()
    selected_layer = None

    if len(non_empty_history) == len(layers):
        for hist_item, layer_item in zip(non_empty_history, layers):
            created_by = hist_item.get("created_by", "")
            if "/sd.cpp/build/bin" in created_by or "/sd.cpp/bin" in created_by:
                selected_layer = layer_item
                break

    if selected_layer is None:
        candidates = [l for l in layers if 30 * 1024 * 1024 <= int(l.get("size", 0)) <= 1500 * 1024 * 1024]
        selected_layer = candidates[-1] if candidates else layers[-1]

    blob_digest = selected_layer["digest"]
    blob_url = f"https://ghcr.io/v2/{GHCR_REPO}/blobs/{blob_digest}"
    return blob_url, selected_layer.get("mediaType", ""), blob_headers


def _build_linux_lib_paths(sdcpp_dir: Path) -> List[str]:
    runtime_dir = sdcpp_dir / "runtime_libs"
    lib_paths: List[str] = [str(sdcpp_dir)]
    if runtime_dir.is_dir():
        lib_paths.append(str(runtime_dir))
    lib_paths.extend(
        [
            "/usr/local/cuda/lib64",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
        ]
    )
    try:
        import site

        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            nv_root = Path(sp) / "nvidia"
            if nv_root.is_dir():
                for sub in nv_root.glob("*/lib"):
                    if sub.is_dir():
                        lib_paths.append(str(sub))
    except Exception:
        pass
    lib_paths.extend(["/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"])
    return list(dict.fromkeys(p for p in lib_paths if p))


def _build_linux_env(sdcpp_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    lib_paths = _build_linux_lib_paths(sdcpp_dir)
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        lib_paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in lib_paths if p))
    env["GGML_BACKEND_DIR"] = str(sdcpp_dir)
    return env


def _build_sd_server_cmd(sdcpp_dir: Path, extra_args: List[str]) -> Tuple[List[str], Dict[str, str]]:
    exe_name = "sd-server.exe" if platform.system().lower() == "windows" else "sd-server"
    server_bin = sdcpp_dir / exe_name
    if not server_bin.exists():
        raise RuntimeError(f"Binário {exe_name} não encontrado em {sdcpp_dir}.")

    if platform.system().lower() == "windows":
        return [str(server_bin), *extra_args], os.environ.copy()

    env = _build_linux_env(sdcpp_dir)
    ld_linux = sdcpp_dir / "runtime_libs" / "ld-linux-x86-64.so.2"
    if ld_linux.exists():
        lib_path_str = ":".join(_build_linux_lib_paths(sdcpp_dir))
        return [str(ld_linux), "--library-path", lib_path_str, str(server_bin), *extra_args], env
    return [str(server_bin), *extra_args], env


def _verify_sd_server_binary(sdcpp_dir: Path) -> None:
    cmd, env = _build_sd_server_cmd(sdcpp_dir, ["--help"])
    res = subprocess.run(cmd, env=env, cwd=str(sdcpp_dir), capture_output=True, text=True, timeout=15)
    if res.returncode != 0:
        err_msg = (res.stderr or res.stdout or "").strip()
        raise RuntimeError(f"sd-server --help retornou código {res.returncode}: {err_msg}")


def _extract_ubuntu24_runtime_libs(sdcpp_dir: Path, layers: List[Dict], non_empty_history: List[Dict], headers: Dict[str, str]) -> None:
    """
    Extrai ld-linux-x86-64.so.2, glibc (2.39), libstdc++ e libgomp da imagem Ubuntu 24.04
    para que o binário pré-compilado rode no Ubuntu 22.04 do Google Colab sem compilar nada.
    """
    runtime_dir = sdcpp_dir / "runtime_libs"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    target_layers: List[Dict] = [layers[0]]
    if len(non_empty_history) == len(layers):
        for hist_item, layer_item in zip(non_empty_history, layers):
            if "libgomp1" in hist_item.get("created_by", ""):
                target_layers.append(layer_item)
                break

    wanted_prefixes = (
        "ld-linux-x86-64.so",
        "libc.so",
        "libm.so",
        "libpthread.so",
        "libdl.so",
        "librt.so",
        "libresolv.so",
        "libgcc_s.so",
        "libstdc++.so",
        "libgomp.so",
    )

    pending_links: List[Tuple[Path, str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, layer in enumerate(target_layers):
            blob_url = f"https://ghcr.io/v2/{GHCR_REPO}/blobs/{layer['digest']}"
            blob_path = Path(tmpdir) / f"runtime_layer_{idx}.tar.gz"
            download_file(blob_url, blob_path, headers=headers, min_bytes=1_000_000)

            with tarfile.open(blob_path, "r:*") as tf:
                for member in tf.getmembers():
                    clean_name = member.name.lstrip("./")
                    if not (clean_name.startswith("usr/lib/x86_64-linux-gnu/") or clean_name.startswith("lib/x86_64-linux-gnu/")):
                        continue
                    base = os.path.basename(clean_name)
                    if not base or not base.startswith(wanted_prefixes) or member.isdir():
                        continue
                    out_path = runtime_dir / base
                    if member.issym() or member.islnk():
                        target_name = os.path.basename(member.linkname)
                        if target_name:
                            pending_links.append((out_path, target_name))
                        continue
                    if out_path.exists() or out_path.is_symlink():
                        out_path.unlink()
                    src_f = tf.extractfile(member)
                    if src_f is None:
                        continue
                    with src_f, open(out_path, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                    if "ld-linux" in base or ".so" in base:
                        out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    for out_path, target_name in pending_links:
        if out_path.exists() or out_path.is_symlink():
            out_path.unlink()
        os.symlink(target_name, out_path)


def _install_sdcpp_from_ghcr_cuda(sdcpp_dir: Path, dry_run: bool = False) -> bool:
    layers, non_empty_history, headers = _resolve_ghcr_cuda_info()
    blob_url, media_type, _ = _resolve_ghcr_cuda_layer()
    log(f"  [GHCR] Layer CUDA resolvida: {blob_url} ({media_type})")
    if dry_run:
        return True

    shutil.rmtree(sdcpp_dir, ignore_errors=True)
    sdcpp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        blob_path = Path(tmpdir) / "sdcpp_bin_layer.tar.gz"
        download_file(blob_url, blob_path, headers=headers, min_bytes=10 * 1024 * 1024)

        extracted_count = 0
        pending_links: List[Tuple[Path, str]] = []

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
                    if target_name:
                        pending_links.append((out_path, target_name))
                    continue

                if out_path.exists() or out_path.is_symlink():
                    out_path.unlink()

                src_f = tf.extractfile(member)
                if src_f is None:
                    continue
                with src_f, open(out_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                if member.mode & 0o111 or out_path.name in ("sd-server", "sd-cli") or ".so" in out_path.name:
                    out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                extracted_count += 1

        for out_path, target_name in pending_links:
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            os.symlink(target_name, out_path)
            extracted_count += 1

    if extracted_count == 0:
        raise RuntimeError("Não foi possível extrair arquivos da layer GHCR.")

    try:
        _verify_sd_server_binary(sdcpp_dir)
    except Exception as first_exc:
        err_str = str(first_exc)
        if any(tok in err_str for tok in ("GLIBC_", "GLIBCXX_", "CXXABI_", "libgomp")):
            log("  [GHCR] Host com glibc mais antiga (ex.: Colab Ubuntu 22.04); baixando runtime_libs Ubuntu 24.04...")
            try:
                _extract_ubuntu24_runtime_libs(sdcpp_dir, layers, non_empty_history, headers)
                _verify_sd_server_binary(sdcpp_dir)
            except Exception as second_exc:
                shutil.rmtree(sdcpp_dir, ignore_errors=True)
                raise RuntimeError(f"Falha ao validar binário CUDA com runtime_libs: {second_exc}") from second_exc
        else:
            shutil.rmtree(sdcpp_dir, ignore_errors=True)
            raise RuntimeError(f"Falha ao validar binário CUDA extraído: {first_exc}") from first_exc

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
        linux_assets = [
            a for a in assets
            if "linux" in a.get("name", "").lower()
            and a.get("name", "").endswith(".zip")
            and "rocm" not in a.get("name", "").lower()
        ]
        for pat in ("vulkan.zip", "ubuntu-24.04-x86_64.zip", "ubuntu", "avx2.zip"):
            for asset in linux_assets:
                if pat in asset.get("name", "").lower():
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

    shutil.rmtree(sdcpp_dir, ignore_errors=True)
    sdcpp_dir.mkdir(parents=True, exist_ok=True)
    pending_links: List[Tuple[Path, str]] = []

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
                    unix_mode = info.external_attr >> 16
                    if stat.S_ISLNK(unix_mode):
                        target_name = os.path.basename(zf.read(info).decode("utf-8", errors="ignore").strip())
                        if target_name:
                            pending_links.append((out_file, target_name))
                        continue

                    if out_file.exists() or out_file.is_symlink():
                        out_file.unlink()
                    with zf.open(info, "r") as src_f, open(out_file, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                    if platform.system().lower() == "linux" and (base in ("sd-server", "sd-cli") or ".so" in base):
                        out_file.chmod(out_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    for out_file, target_name in pending_links:
        if out_file.exists() or out_file.is_symlink():
            out_file.unlink()
        os.symlink(target_name, out_file)

    _verify_sd_server_binary(sdcpp_dir)
    log(f"  [OK] Binários pré-compilados extraídos e validados em {sdcpp_dir}.")


def install_sdcpp_binaries(sdcpp_dir: Path, dry_run: bool = False) -> None:
    log("[3/4] Instalando binários pré-compilados do stable-diffusion.cpp (sem compilação)...")
    server_exe = sdcpp_dir / ("sd-server.exe" if platform.system().lower() == "windows" else "sd-server")
    if server_exe.exists() and not dry_run:
        try:
            _verify_sd_server_binary(sdcpp_dir)
            log(f"  [OK] {server_exe.name} já presente e válido em {sdcpp_dir}.")
            return
        except Exception as exc:
            log(f"  [AVISO] Binário existente em {sdcpp_dir} inválido ({exc}); reinstalando...")
            shutil.rmtree(sdcpp_dir, ignore_errors=True)

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
