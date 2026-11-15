"""
Helper para baixar LoRAs do HuggingFace ou Civitai para o diretório LORA_DIR.
Mantém os links configurados pelo usuário no notebook original, agora opcionais.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional


def download_hf_loras(lora_dir: str, repo_id: str, filenames: Iterable[str]) -> List[str]:
    from huggingface_hub import hf_hub_download

    os.makedirs(lora_dir, exist_ok=True)
    downloaded: List[str] = []
    for fname in filenames:
        if not fname:
            continue
        path = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=lora_dir)
        downloaded.append(path)
    return downloaded


def download_url_loras(
    lora_dir: str,
    urls: Iterable[str],
    civitai_token: Optional[str] = None,
) -> List[str]:
    target_dir = Path(lora_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for idx, url in enumerate(urls):
        if not url:
            continue
        final_url = url.strip()
        if civitai_token and "civitai" in final_url.lower():
            sep = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{sep}token={civitai_token.strip()}"

        print(f"📥 Baixando LoRA: {url.split('?')[0]}", flush=True)
        req = urllib.request.Request(final_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            disp = resp.headers.get("Content-Disposition", "")
            filename = None
            if "filename=" in disp:
                part = disp.split("filename=", 1)[1].strip().strip('"').strip("'")
                if part:
                    filename = os.path.basename(part)
            if not filename:
                parsed = urllib.parse.urlparse(resp.geturl())
                filename = os.path.basename(parsed.path) or f"lora_{idx}.safetensors"
                if not filename.endswith(".safetensors"):
                    filename += ".safetensors"

            out_path = target_dir / filename
            tmp_path = target_dir / (filename + ".part")
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp_path.replace(out_path)
            print(f"  [OK] Salvo: {out_path}", flush=True)
            saved.append(str(out_path))

    return saved
