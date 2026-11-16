"""
Backend de inferência para o GradioKlein 9B:
- Modo Produção: gerencia um subprocesso persistente do `sd-server` (stable-diffusion.cpp)
  e envia requisições assíncronas para /sdcpp/v1/img_gen com streaming de progresso.
- Modo Debug: não exige nem inicia `sd-server` ou modelos GGUF 9B; simula passos de
  progresso e retorna uma imagem sólida preenchida com uma única cor.
"""

from __future__ import annotations

import atexit
import base64
import io
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests
from PIL import Image


SCHEDULER_MAP = {
    "Default": "flux2",
    "Karras": "karras",
    "Beta": "beta",
    "Exponential": "exponential",
}


def parse_hex_color(color_str: str) -> Tuple[int, int, int]:
    """Converte '#RRGGBB' ou 'rgb(...)' em tupla (R, G, B)."""
    if not color_str:
        return (255, 0, 255)
    raw = str(color_str).strip()
    if raw.startswith("#") and len(raw) == 7:
        try:
            return (int(raw[1:3], 16), int(raw[3:5], 16), int(raw[5:7], 16))
        except ValueError:
            pass
    if raw.startswith("#") and len(raw) == 4:
        try:
            return tuple(int(c * 2, 16) for c in raw[1:4])  # type: ignore[return-value]
        except ValueError:
            pass
    if raw.lower().startswith("rgb"):
        nums = [
            int(float(part.strip()))
            for part in raw[raw.find("(") + 1 : raw.rfind(")")].split(",")[:3]
            if part.strip()
        ]
        if len(nums) == 3:
            return (
                max(0, min(255, nums[0])),
                max(0, min(255, nums[1])),
                max(0, min(255, nums[2])),
            )
    return (255, 0, 255)


def pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def base64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str and b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class SDCppBackend:
    """Wrapper unificado entre `sd-server` (stable-diffusion.cpp) e Modo Debug."""

    def __init__(
        self,
        debug_mode: bool = False,
        sdcpp_dir: str = "./sdcpp_bin",
        models_dir: str = "./models",
        lora_dir: str = "./LoRAs",
        host: str = "127.0.0.1",
        port: int = 8188,
        debug_color: str = "#FF00FF",
        offload_to_cpu: bool = True,
        vae_tiling: bool = True,
        flash_attn: bool = True,
    ) -> None:
        self.debug_mode = bool(debug_mode)
        self.sdcpp_dir = Path(sdcpp_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.lora_dir = Path(lora_dir).resolve()
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}"
        self.debug_color = debug_color
        self.offload_to_cpu = offload_to_cpu
        self.vae_tiling = vae_tiling
        self.flash_attn = flash_attn

        self._proc: Optional[subprocess.Popen] = None
        self._log_lines: List[str] = []
        self._log_thread: Optional[threading.Thread] = None
        atexit.register(self.stop_server)

    def _find_model_paths(self) -> Dict[str, Path]:
        flux_dir = self.models_dir / "flux2_klein_9b"
        paths = {
            "unet": flux_dir / "unet/turbo/PornMaster_Flux2Klein_v4_Turbo_Q4_K.gguf",
            "llm": flux_dir / "text_encoders/Qwen3-VL-8B-Q4_K_S.gguf",
            "vae": flux_dir / "vae/full_encoder_small_decoder.safetensors",
        }
        missing = [f"{k}: {p}" for k, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Arquivos do modelo FLUX.2 Klein 9B ausentes. Execute `python install.py --mode prod` primeiro:\n"
                + "\n".join(missing)
            )
        return paths

    def _server_executable(self) -> Path:
        exe_name = "sd-server.exe" if platform.system().lower() == "windows" else "sd-server"
        exe_path = self.sdcpp_dir / exe_name
        if not exe_path.exists():
            raise FileNotFoundError(
                f"Binário `{exe_name}` não encontrado em `{self.sdcpp_dir}`. "
                "Execute `python install.py --mode prod` ou inicie com `--debug`."
            )
        return exe_path

    def _pump_logs(self, pipe) -> None:
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                text = line.rstrip()
                self._log_lines.append(text)
                if len(self._log_lines) > 200:
                    self._log_lines.pop(0)
                print(f"[sd-server] {text}", flush=True)
        except Exception:
            pass

    def start_server(self, timeout_s: float = 300.0) -> None:
        if self.debug_mode:
            print("[DEBUG MODE] Backend sólido ativo — sd-server NÃO será iniciado.", flush=True)
            return

        if self._proc is not None and self._proc.poll() is None:
            return

        # Reutiliza se já houver um servidor pronto nessa porta
        try:
            r = requests.get(f"{self.base_url}/sdcpp/v1/models/status", timeout=1.5)
            if r.status_code == 200 and r.json().get("state") == "ready":
                print(f"[sd-server] Já ativo em {self.base_url}.", flush=True)
                return
        except Exception:
            pass

        server_bin = self._server_executable()
        model_paths = self._find_model_paths()
        self.lora_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(server_bin),
            "--diffusion-model",
            str(model_paths["unet"]),
            "--llm",
            str(model_paths["llm"]),
            "--vae",
            str(model_paths["vae"]),
            "--lora-model-dir",
            str(self.lora_dir),
            "--listen-ip",
            self.host,
            "--listen-port",
            str(self.port),
            "-v",
        ]
        if self.offload_to_cpu:
            cmd.append("--offload-to-cpu")
        if self.vae_tiling:
            cmd.append("--vae-tiling")
        if self.flash_attn:
            cmd.extend(["--flash-attn", "on"])

        env = os.environ.copy()
        if platform.system().lower() != "windows":
            extra_libs = [str(self.sdcpp_dir)]
            runtime_dir = self.sdcpp_dir / "runtime_libs"
            if runtime_dir.is_dir():
                extra_libs.append(str(runtime_dir))
            extra_libs.extend(["/usr/local/cuda/lib64", "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64"])
            try:
                import site

                for sp in site.getsitepackages() + [site.getusersitepackages()]:
                    nv_root = Path(sp) / "nvidia"
                    if nv_root.is_dir():
                        for sub in nv_root.glob("*/lib"):
                            if sub.is_dir():
                                extra_libs.append(str(sub))
            except Exception:
                pass
            extra_libs.extend(["/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"])
            dedup_libs = list(dict.fromkeys(p for p in extra_libs if p))

            ld_linux = runtime_dir / "ld-linux-x86-64.so.2"
            if ld_linux.exists():
                cmd = [str(ld_linux), "--library-path", ":".join(dedup_libs), *cmd]

            existing_ld = env.get("LD_LIBRARY_PATH", "")
            if existing_ld:
                dedup_libs.append(existing_ld)
            env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(dedup_libs))
            env["GGML_BACKEND_DIR"] = str(self.sdcpp_dir)

        print(f"[sd-server] Iniciando em {self.base_url} ...", flush=True)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(self.sdcpp_dir),
        )
        self._log_thread = threading.Thread(target=self._pump_logs, args=(self._proc.stdout,), daemon=True)
        self._log_thread.start()

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                tail = "\n".join(self._log_lines[-20:])
                raise RuntimeError(f"sd-server encerrou inesperadamente (código {self._proc.returncode}):\n{tail}")
            try:
                resp = requests.get(f"{self.base_url}/sdcpp/v1/models/status", timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    state = data.get("state")
                    if state == "ready":
                        print("[sd-server] Modelo FLUX.2 Klein 9B carregado e pronto!", flush=True)
                        return
                    if state == "error":
                        raise RuntimeError(f"Erro ao carregar modelo no sd-server: {data.get('error')}")
            except requests.RequestException:
                pass
            time.sleep(1.0)

        raise TimeoutError(f"Tempo limite excedido ({timeout_s}s) aguardando sd-server ficar pronto.")

    def stop_server(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def list_server_loras(self) -> List[Dict[str, str]]:
        """Retorna os metadados de LoRA conhecidos pelo sd-server ({name, path})."""
        if self.debug_mode:
            return []
        try:
            resp = requests.get(f"{self.base_url}/sdcpp/v1/capabilities", timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("loras", []) or []
        except Exception:
            pass
        return []

    def _resolve_lora_entries(self, selected_loras: List[str], weights: List[float]) -> List[Dict]:
        clean_selected = [f for f in (selected_loras or []) if f]
        if not clean_selected:
            return []

        server_loras = self.list_server_loras()
        by_name = {item.get("name", ""): item.get("path", "") for item in server_loras}
        by_path = {item.get("path", ""): item.get("path", "") for item in server_loras}
        by_base = {os.path.basename(item.get("path", "")): item.get("path", "") for item in server_loras}

        entries: List[Dict] = []
        for idx, lora_file in enumerate(clean_selected):
            w = float(weights[idx]) if idx < len(weights) else 1.0
            stem = os.path.splitext(os.path.basename(lora_file))[0]
            resolved_path = (
                by_path.get(lora_file)
                or by_base.get(os.path.basename(lora_file))
                or by_name.get(stem)
                or lora_file
            )
            entries.append({"path": resolved_path, "multiplier": w, "is_high_noise": False})
        return entries

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        flow_shift: float,
        sigma_schedule: str = "Default",
        stochastic_sampling: bool = False,
        custom_sigmas: Optional[List[float]] = None,
        ref_image: Optional[Image.Image] = None,
        init_image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        inpaint_strength: float = 1.0,
        selected_loras: Optional[List[str]] = None,
        lora_weights: Optional[List[float]] = None,
        system_prompt: Optional[str] = None,
        debug_color: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Image.Image:
        """
        Gera ou edita uma imagem via `sd-server` ou retorna cor sólida em Modo Debug.
        """
        width = max(16, int(width))
        height = max(16, int(height))
        steps = max(1, int(steps))
        seed = int(seed)

        if self.debug_mode:
            rgb = parse_hex_color(debug_color or self.debug_color)
            for s in range(1, steps + 1):
                if progress_callback:
                    pct = (s / float(steps)) * 0.85 + 0.05
                    progress_callback(pct, f"[DEBUG] Simulando passo {s}/{steps}...")
                time.sleep(0.05)
            return Image.new("RGB", (width, height), rgb)

        self.start_server()

        full_prompt = prompt.strip()
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt.strip()}\n\n{full_prompt}" if full_prompt else system_prompt.strip()

        scheduler_name = SCHEDULER_MAP.get(sigma_schedule, "flux2")
        sample_method = "euler_a" if stochastic_sampling else "euler"

        sample_params: Dict = {
            "scheduler": scheduler_name,
            "sample_method": sample_method,
            "sample_steps": steps,
            "flow_shift": float(flow_shift),
            "guidance": {
                "txt_cfg": 1.0,
                "img_cfg": 1.0,
                "distilled_guidance": 1.0,
            },
        }

        if custom_sigmas and len(custom_sigmas) >= 1:
            sigmas_clean = [max(0.0, float(v)) for v in custom_sigmas[:steps]]
            if not sigmas_clean or sigmas_clean[-1] != 0.0:
                sigmas_clean.append(0.0)
            sample_params["custom_sigmas"] = sigmas_clean

        lora_entries = self._resolve_lora_entries(selected_loras or [], lora_weights or [])

        payload: Dict = {
            "prompt": full_prompt,
            "negative_prompt": "",
            "width": width,
            "height": height,
            "seed": seed,
            "batch_count": 1,
            "sample_params": sample_params,
            "lora": lora_entries,
            "vae_tiling_params": {
                "enabled": bool(self.vae_tiling),
                "tile_size_w": 512,
                "tile_size_h": 512,
                "target_overlap": 0.25,
            },
            "output_format": "png",
        }

        if ref_image is not None:
            payload["ref_images"] = [pil_to_base64_png(ref_image)]
            payload["ref_image_args"] = "resize_single_ref_to_target=true"

        if init_image is not None and mask_image is not None:
            payload["init_image"] = pil_to_base64_png(init_image)
            payload["mask_image"] = pil_to_base64_png(mask_image.convert("L"))
            payload["strength"] = max(0.05, min(1.0, float(inpaint_strength)))

        if progress_callback:
            progress_callback(0.08, f"Enviando job ao sd-server ({width}x{height})...")

        submit_resp = requests.post(f"{self.base_url}/sdcpp/v1/img_gen", json=payload, timeout=30.0)
        if submit_resp.status_code not in (200, 202):
            raise RuntimeError(f"sd-server rejeitou requisição ({submit_resp.status_code}): {submit_resp.text}")

        job_id = submit_resp.json()["id"]
        poll_url = f"{self.base_url}/sdcpp/v1/jobs/{job_id}"

        while True:
            job_resp = requests.get(poll_url, timeout=15.0)
            job_resp.raise_for_status()
            job = job_resp.json()
            status = job.get("status")

            prog = job.get("progress") or {}
            stage = prog.get("stage", "")
            comp = prog.get("completed")
            tot = prog.get("total")
            if progress_callback:
                if comp is not None and tot:
                    pct = (float(comp) / float(tot)) * 0.82 + 0.10
                    progress_callback(pct, f"sd-server [{stage}]: {comp}/{tot}")
                elif stage:
                    progress_callback(0.10, f"sd-server [{stage}]...")

            if status == "completed":
                images = (job.get("result") or {}).get("images") or []
                if not images or not images[0].get("b64_json"):
                    raise RuntimeError("sd-server concluiu sem retornar imagem.")
                return base64_to_pil(images[0]["b64_json"])

            if status in ("failed", "cancelled"):
                err = job.get("error") or {}
                raise RuntimeError(f"Falha na geração ({status}): {err.get('message', job)}")

            time.sleep(0.35)
