"""
GradioKlein 9B — Interface Web Gradio para FLUX.2 Klein 9B via stable-diffusion.cpp
Compatível com:
- Produção (Colab / GPU): `sd-server` pré-compilado + PornMaster-flux2-klein-9b (Q4_K GGUF + Qwen3-VL-8B GGUF + VAE).
- Modo Debug (`--debug`): Não instala/carrega stable-diffusion.cpp nem os modelos 9B;
  executa todas as detecções YOLO (pessoas, face, atributos, pose), máscara facial e blending,
  mas retorna uma imagem preenchida com uma única cor sólida ao gerar.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import logging
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from ultralytics import YOLO

from sdcpp_backend import SDCppBackend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MAX_SEED = 2**31 - 1
DEFAULT_LORA_NAME = ""
DEFAULT_SIGMAS_LIST = [0.95, 0.72, 0.52, 0.22]
MAX_PERSON_CARDS = 8
MAX_LORA_SLIDERS = 6
BASE_SEQ_LEN_05MP = 256
MAX_SEQ_LEN_20MP = 4096
DEFAULT_BASE_SHIFT = 0.5
DEFAULT_MAX_SHIFT = 1.15
PREVIEW_MAX_DIM = 512

# Estado global de runtime (inicializado por init_runtime ou sob demanda)
DEBUG_MODE: bool = os.environ.get("GRADIOKLEIN_DEBUG", "").strip().lower() in ("1", "true", "yes")
MODELS_DIR: Path = Path(os.environ.get("MODELS_DIR", "./models")).resolve()
LORA_DIR: str = os.environ.get("LORA_DIR", "/content/LoRAs" if os.path.exists("/content") else "./LoRAs")
SDCPP_DIR: Path = Path(os.environ.get("SDCPP_DIR", "./sdcpp_bin")).resolve()
SDCPP_PORT: int = int(os.environ.get("SDCPP_PORT", "8188"))

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_person_model: Optional[YOLO] = None
yolo_face_model: Optional[YOLO] = None
yolo_attr_model: Optional[YOLO] = None
yolo_pose_model: Optional[YOLO] = None
sd_backend: Optional[SDCppBackend] = None


def _image_widget_kwargs(buttons: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """Mantém compatibilidade entre Gradio 5.x e Gradio 6.x."""
    sig = inspect.signature(gr.Image.__init__)
    params = sig.parameters
    if buttons is not None:
        if "buttons" in params:
            kwargs["buttons"] = buttons
        else:
            if "show_download_button" in params:
                kwargs["show_download_button"] = ("download" in buttons)
            if "show_fullscreen_button" in params:
                kwargs["show_fullscreen_button"] = ("fullscreen" in buttons)
            if "show_share_button" in params:
                kwargs["show_share_button"] = False
    return kwargs


def _resolve_yolo_path(yolo_dir: Path, filename: str) -> str:
    candidate = yolo_dir / filename
    if candidate.exists():
        return str(candidate)
    local_root = Path(filename)
    if local_root.exists():
        return str(local_root)
    colab_path = Path("/content") / filename
    if colab_path.exists():
        return str(colab_path)
    return filename


def init_runtime(
    debug_mode: Optional[bool] = None,
    models_dir: Optional[str] = None,
    lora_dir: Optional[str] = None,
    sdcpp_dir: Optional[str] = None,
    sdcpp_port: Optional[int] = None,
    start_server: bool = True,
) -> None:
    """Carrega os 4 modelos YOLO e inicializa o backend (SDCpp ou Debug Sólido)."""
    global DEBUG_MODE, MODELS_DIR, LORA_DIR, SDCPP_DIR, SDCPP_PORT
    global yolo_person_model, yolo_face_model, yolo_attr_model, yolo_pose_model, sd_backend

    if debug_mode is not None:
        DEBUG_MODE = bool(debug_mode)
    if models_dir is not None:
        MODELS_DIR = Path(models_dir).resolve()
    if lora_dir is not None:
        LORA_DIR = lora_dir
    if sdcpp_dir is not None:
        SDCPP_DIR = Path(sdcpp_dir).resolve()
    if sdcpp_port is not None:
        SDCPP_PORT = int(sdcpp_port)

    os.makedirs(LORA_DIR, exist_ok=True)
    yolo_dir = MODELS_DIR / "yolo"

    if yolo_person_model is None:
        logger.info("Carregando YOLOv8n (pessoas)...")
        yolo_person_model = YOLO(_resolve_yolo_path(yolo_dir, "yolov8n.pt"))
        try:
            yolo_person_model.to(device)
        except Exception:
            pass

    if yolo_face_model is None:
        logger.info("Carregando YOLOv8n-Face...")
        yolo_face_model = YOLO(_resolve_yolo_path(yolo_dir, "yolov8n-face-lindevs.pt"))
        try:
            yolo_face_model.to(device)
        except Exception:
            pass

    if yolo_attr_model is None:
        logger.info("Carregando YOLO Atributos Corporais (yoloboop)...")
        yolo_attr_model = YOLO(_resolve_yolo_path(yolo_dir, "yoloboop.pt"))

    if yolo_pose_model is None:
        logger.info("Carregando YOLOv8m-Pose...")
        yolo_pose_model = YOLO(_resolve_yolo_path(yolo_dir, "yolov8m-pose.pt"))
        try:
            yolo_pose_model.to(device)
        except Exception:
            pass

    if sd_backend is None or sd_backend.debug_mode != DEBUG_MODE:
        sd_backend = SDCppBackend(
            debug_mode=DEBUG_MODE,
            sdcpp_dir=str(SDCPP_DIR),
            models_dir=str(MODELS_DIR),
            lora_dir=LORA_DIR,
            port=SDCPP_PORT,
        )

    if start_server and not DEBUG_MODE:
        sd_backend.start_server()


# ------------------------------------------------------------------------------
# 1. Pipeline de Compressão WebP em Memória (BytesIO)
# ------------------------------------------------------------------------------
def compress_webp_level1(img: Optional[Image.Image], max_dim: int = PREVIEW_MAX_DIM, quality: int = 65) -> Optional[Image.Image]:
    if img is None:
        return None
    target = img.copy()
    w, h = target.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        target = target.resize((max(16, int(round(w * scale))), max(16, int(round(h * scale)))), Image.Resampling.HAMMING)

    buf = io.BytesIO()
    target.convert("RGB").save(buf, format="WEBP", quality=quality, method=4)
    buf.seek(0)
    return Image.open(buf)


def compress_webp_level2(img: Optional[Image.Image], quality: int = 85) -> Optional[Image.Image]:
    if img is None:
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=quality, method=4)
    buf.seek(0)
    return Image.open(buf)


def compress_webp_level3(img: Optional[Image.Image], thumb_size: Tuple[int, int] = (110, 220), quality: int = 40) -> Optional[Image.Image]:
    if img is None:
        return None
    target = img.copy()
    target.thumbnail(thumb_size, Image.Resampling.HAMMING)

    buf = io.BytesIO()
    target.convert("RGB").save(buf, format="WEBP", quality=quality, method=2)
    buf.seek(0)
    return Image.open(buf)


def compute_image_signature(image: Optional[Image.Image]) -> Optional[str]:
    if image is None:
        return None
    try:
        raw_bytes = image.tobytes()
        sample = raw_bytes[:10000] + raw_bytes[-10000:] if len(raw_bytes) > 20000 else raw_bytes
        return hashlib.md5(f"{image.size}_{image.mode}_{len(raw_bytes)}".encode() + sample).hexdigest()
    except Exception:
        return f"{id(image)}"


# ------------------------------------------------------------------------------
# 2. Cálculo do Flow Shift / Dynamic Resolution (μ)
# ------------------------------------------------------------------------------
def calculate_mu_preview(
    shift_mode: str,
    shift_val: float,
    shift_offset: float,
    mp_val: float,
    gen_w: int = 1024,
    gen_h: int = 1024,
) -> Tuple[float, str]:
    image_seq_len = (gen_w // 16) * (gen_h // 16)
    if shift_mode == "Dynamic (Resolução)":
        base_s = round(max(0.01, DEFAULT_BASE_SHIFT + float(shift_offset)), 4)
        max_s = round(max(0.01, DEFAULT_MAX_SHIFT + float(shift_offset)), 4)
        slope = (max_s - base_s) / float(MAX_SEQ_LEN_20MP - BASE_SEQ_LEN_05MP)
        intercept = base_s - slope * BASE_SEQ_LEN_05MP
        mu = float(slope * image_seq_len + intercept)
        desc = f"μ = {mu:.3f} (D/Res: {base_s:.2f}→{max_s:.2f} | Seq: {image_seq_len} toks)"
    else:
        mu = float(shift_val)
        desc = f"μ = {mu:.3f} (Fixo)"
    return mu, desc


# ------------------------------------------------------------------------------
# 3. Detecções YOLO (Pessoas, Face, Atributo Corporal e Pose)
# ------------------------------------------------------------------------------
def detect_person_bbox(
    image: Image.Image,
    conf_threshold: float = 0.35,
) -> Tuple[List[Tuple[int, int, int, int, str]], str]:
    model = yolo_person_model
    if model is None:
        raise RuntimeError("YOLOv8 de pessoas não inicializado.")

    w, h = image.size
    results = model.predict(source=image, classes=[0], conf=float(conf_threshold), verbose=False)

    detected_list: List[Tuple[int, int, int, int, str]] = []
    if results and len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        raw_candidates = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            x1 = max(0, min(w, int(round(xyxy[0]))))
            y1 = max(0, min(h, int(round(xyxy[1]))))
            x2 = max(0, min(w, int(round(xyxy[2]))))
            y2 = max(0, min(h, int(round(xyxy[3]))))
            area = (x2 - x1) * (y2 - y1)
            raw_candidates.append((x1, y1, x2, y2, f"Pessoa ({conf:.2f})", area))

        raw_candidates.sort(key=lambda item: item[5], reverse=True)
        detected_list = [(x1, y1, x2, y2, label) for x1, y1, x2, y2, label, _ in raw_candidates]
        return detected_list, f"{len(detected_list)} pessoa(s) via YOLOv8"

    fallback_box = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
    detected_list.append((*fallback_box, "fallback (centro)"))
    return detected_list, "Fallback central aplicado"


def detect_face_yolo(crop_img: Image.Image) -> Optional[Dict[str, Any]]:
    face_model = yolo_face_model
    if face_model is None:
        return None

    w, h = crop_img.size
    results = face_model.predict(source=crop_img, conf=0.25, verbose=False)

    if not results or len(results) == 0 or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    best_box = None
    max_area = 0.0

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        bw = xyxy[2] - xyxy[0]
        bh = xyxy[3] - xyxy[1]
        area = bw * bh
        if area > max_area:
            max_area = area
            best_box = xyxy

    if best_box is None:
        return None

    bx1, by1, bx2, by2 = best_box
    norm_center = (((bx1 + bx2) / 2.0) / float(w), ((by1 + by2) / 2.0) / float(h))
    norm_size = (max(0.01, (bx2 - bx1) / float(w)), max(0.01, (by2 - by1) / float(h)))

    theta_deg = 0.0
    if hasattr(results[0], "keypoints") and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        kpts = results[0].keypoints.data[0]
        if len(kpts) >= 2:
            p1 = kpts[0]
            p2 = kpts[1]
            if float(p1[2]) > 0.3 and float(p2[2]) > 0.3:
                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                theta_deg = math.degrees(math.atan2(dy, dx))

    return {
        "norm_center": norm_center,
        "norm_size": norm_size,
        "theta": theta_deg,
    }


def detect_body_attribute(crop_img: Image.Image, conf_threshold: float = 0.25, imgsz: int = 640) -> str:
    attr_model = yolo_attr_model
    if attr_model is None:
        return ""

    try:
        results = attr_model.predict(source=crop_img, imgsz=int(imgsz), conf=float(conf_threshold), verbose=False)
    except Exception:
        results = attr_model.predict(source=crop_img, imgsz=int(imgsz), verbose=False)

    if not results or len(results) == 0:
        return ""

    res = results[0]
    if hasattr(res, "probs") and res.probs is not None:
        top1_idx = int(res.probs.top1)
        conf_val = float(res.probs.top1conf) if hasattr(res.probs, "top1conf") else 1.0
        if conf_val >= float(conf_threshold):
            return str(res.names[top1_idx]).strip()
        return ""

    if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
        top_box = sorted(res.boxes, key=lambda b: float(b.conf[0]), reverse=True)[0]
        if float(top_box.conf[0]) >= float(conf_threshold):
            cls_id = int(top_box.cls[0])
            return str(res.names[cls_id]).strip()

    return ""


def detect_pose_yolo(crop_img: Image.Image, conf: float = 0.25) -> Dict[str, str]:
    pose_model = yolo_pose_model
    if pose_model is None:
        return {"posture": "", "facing": "", "summary": ""}

    results = pose_model.predict(source=crop_img, conf=float(conf), verbose=False)
    if (
        not results
        or len(results) == 0
        or not hasattr(results[0], "keypoints")
        or results[0].keypoints is None
        or len(results[0].keypoints.data) == 0
    ):
        return {"posture": "", "facing": "", "summary": ""}

    try:
        kpts = results[0].keypoints.data[0].cpu().numpy()
        if len(kpts) < 17:
            return {"posture": "", "facing": "", "summary": ""}
    except Exception:
        return {"posture": "", "facing": "", "summary": ""}

    def get_pt(idx: int, th: float = 0.30):
        if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] >= th:
            return (kpts[idx][:2], float(kpts[idx][2]))
        return (None, 0.0)

    nose, c_nose = get_pt(0)
    _, c_leye = get_pt(1)
    _, c_reye = get_pt(2)
    _, c_lear = get_pt(3)
    _, c_rear = get_pt(4)
    l_sho, _ = get_pt(5)
    r_sho, _ = get_pt(6)
    l_hip, _ = get_pt(11)
    r_hip, _ = get_pt(12)
    l_knee, _ = get_pt(13)
    r_knee, _ = get_pt(14)
    l_ankle, _ = get_pt(15)
    r_ankle, _ = get_pt(16)

    w, h = crop_img.size
    posture = "em pé"

    sho_pts = [p for p in [l_sho, r_sho] if p is not None]
    hip_pts = [p for p in [l_hip, r_hip] if p is not None]
    knee_pts = [p for p in [l_knee, r_knee] if p is not None]
    ankle_pts = [p for p in [l_ankle, r_ankle] if p is not None]

    if sho_pts and hip_pts:
        sho_mid = np.mean(sho_pts, axis=0)
        hip_mid = np.mean(hip_pts, axis=0)
        dx_torso = abs(hip_mid[0] - sho_mid[0])
        dy_torso = abs(hip_mid[1] - sho_mid[1])

        angle_torso = math.degrees(math.atan2(dy_torso, max(1e-5, dx_torso)))

        if angle_torso < 42.0:
            posture = "deitado"
        elif knee_pts and angle_torso < 55.0:
            knee_mid = np.mean(knee_pts, axis=0)
            dx_leg = abs(knee_mid[0] - hip_mid[0])
            dy_leg = abs(knee_mid[1] - hip_mid[1])
            if dy_leg < 0.6 * dx_leg:
                posture = "deitado"
        else:
            if knee_pts:
                knee_mid = np.mean(knee_pts, axis=0)
                dy_thigh = abs(knee_mid[1] - hip_mid[1])
                dx_thigh = abs(knee_mid[0] - hip_mid[0])

                if ankle_pts:
                    ankle_mid = np.mean(ankle_pts, axis=0)
                    total_leg_drop = abs(ankle_mid[1] - hip_mid[1])

                    if dy_thigh < 0.52 * dy_torso or dy_thigh < 0.85 * dx_thigh:
                        posture = "sentado"
                    elif total_leg_drop < 1.15 * dy_torso:
                        posture = "sentado"
                    else:
                        posture = "em pé"
                else:
                    if dy_thigh < 0.55 * dy_torso or dy_thigh < 0.9 * dx_thigh:
                        posture = "sentado"
                    else:
                        posture = "em pé"
            else:
                posture = "sentado" if w > h * 1.15 else "em pé"

    elif knee_pts and hip_pts:
        hip_mid = np.mean(hip_pts, axis=0)
        knee_mid = np.mean(knee_pts, axis=0)
        dy_k = abs(knee_mid[1] - hip_mid[1])
        dx_k = abs(knee_mid[0] - hip_mid[0])
        if dy_k < 0.5 * dx_k:
            posture = "deitado"
        elif dy_k < dx_k:
            posture = "sentado"
        else:
            posture = "em pé"
    else:
        posture = "deitado" if w > h * 1.25 else "em pé"

    eyes_count = sum([c_leye >= 0.35, c_reye >= 0.35])
    ears_count = sum([c_lear >= 0.35, c_rear >= 0.35])
    nose_present = c_nose >= 0.35

    torso_len = None
    if sho_pts and hip_pts:
        sho_mid = np.mean(sho_pts, axis=0)
        hip_mid = np.mean(hip_pts, axis=0)
        torso_len = math.hypot(hip_mid[0] - sho_mid[0], hip_mid[1] - sho_mid[1])

    is_side_profile = False
    if l_sho is not None and r_sho is not None and torso_len and torso_len > 20:
        shoulder_span = abs(l_sho[0] - r_sho[0])
        if shoulder_span < 0.36 * torso_len:
            is_side_profile = True

    if not is_side_profile and (eyes_count == 1) and (ears_count == 1):
        is_side_profile = True

    if is_side_profile:
        facing = "de lado"
    elif not nose_present and eyes_count == 0:
        facing = "de costas"
    else:
        facing = "de frente"

    summary = f"{posture} {facing}"
    return {"posture": posture, "facing": facing, "summary": summary}


def format_pose_prompt_tag(
    pose_data: Dict[str, str],
    txt_standing: str,
    txt_sitting: str,
    txt_lying: str,
    txt_front: str,
    txt_back: str,
    txt_side: str,
) -> str:
    if not pose_data:
        return ""
    parts = []
    posture = pose_data.get("posture", "")
    facing = pose_data.get("facing", "")

    if posture == "em pé" and txt_standing and txt_standing.strip():
        parts.append(txt_standing.strip())
    elif posture == "sentado" and txt_sitting and txt_sitting.strip():
        parts.append(txt_sitting.strip())
    elif posture == "deitado" and txt_lying and txt_lying.strip():
        parts.append(txt_lying.strip())

    if facing == "de frente" and txt_front and txt_front.strip():
        parts.append(txt_front.strip())
    elif facing == "de costas" and txt_back and txt_back.strip():
        parts.append(txt_back.strip())
    elif facing == "de lado" and txt_side and txt_side.strip():
        parts.append(txt_side.strip())

    return ", ".join(parts)


def generate_face_preserve_mask(
    target_img: Image.Image,
    blur_fraction_pct: float = 20.0,
    expand_ratio: float = 1.35,
    cached_face_info: Optional[Dict[str, Any]] = None,
    enable_safety_ellipse: bool = True,
) -> Tuple[Image.Image, Image.Image, str, Optional[Dict[str, Any]]]:
    w, h = target_img.size
    face_info = cached_face_info if cached_face_info is not None else detect_face_yolo(target_img)

    has_mask = False
    theta = 0.0
    cx = cy = fw = fh = 0.0

    if face_info is not None:
        norm_cx, norm_cy = face_info["norm_center"]
        norm_fw, norm_fh = face_info["norm_size"]
        cx = norm_cx * w
        cy = norm_cy * h
        fw = norm_fw * w
        fh = norm_fh * h
        theta = face_info["theta"]
        status = f"Face [{int(cx)},{int(cy)},{int(fw)}x{int(fh)}] {theta:.1f}°"
        has_mask = True
    elif enable_safety_ellipse:
        cx, cy = w / 2.0, h * 0.22
        fw = min(w, h) * 0.35
        fh = fw * 1.25
        theta = 0.0
        status = "Elipse de segurança"
        has_mask = True
    else:
        status = "Sem máscara (face não detectada)"
        has_mask = False

    if has_mask:
        calculated_blur = max(3, int(min(fw, fh) * (float(blur_fraction_pct) / 100.0)))
        ew = fw * float(expand_ratio)
        eh = fh * float(expand_ratio)

        ellipse_layer = Image.new("L", (w, h), 255)
        draw_ellipse = ImageDraw.Draw(ellipse_layer)
        draw_ellipse.ellipse([cx - ew / 2.0, cy - eh / 2.0, cx + ew / 2.0, cy + eh / 2.0], fill=0)

        if abs(theta) > 0.5:
            ellipse_layer = ellipse_layer.rotate(theta, resample=Image.Resampling.BILINEAR, center=(cx, cy), fillcolor=255)

        canvas_mask = ellipse_layer
        if calculated_blur > 0:
            canvas_mask = canvas_mask.filter(ImageFilter.GaussianBlur(radius=calculated_blur))
    else:
        canvas_mask = Image.new("L", (w, h), 255)

    overlay = target_img.copy().convert("RGBA")
    mask_np = np.array(canvas_mask).astype(np.float32) / 255.0

    tint_layer = np.zeros((h, w, 4), dtype=np.uint8)
    tint_layer[..., 0] = 255
    tint_layer[..., 1] = 40
    tint_layer[..., 2] = 40
    tint_layer[..., 3] = (mask_np * 95).astype(np.uint8)

    tint_img = Image.fromarray(tint_layer, mode="RGBA")
    overlay = Image.alpha_composite(overlay, tint_img).convert("RGB")

    return canvas_mask, overlay, status, face_info


# ------------------------------------------------------------------------------
# 4. Geometria, Crop e Blend com Padding Horizontal & Inferior Customizáveis
# ------------------------------------------------------------------------------
def calculate_crop_and_resolution(
    img_w: int,
    img_h: int,
    bbox: Tuple[int, int, int, int],
    megapixels: float,
    padding_mult: float = 1.1,
    padding_h: float = 0.0,
    padding_bottom: float = 0.0,
):
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    bw = max(16, bx2 - bx1)
    bh = max(16, by2 - by1)
    cx = (bx1 + bx2) / 2.0

    pad_m = max(1.0, float(padding_mult))
    margin_top = bh * (pad_m - 1.0) / 2.0
    margin_bottom = bh * (pad_m - 1.0) / 2.0

    raw_w = max(16.0, bw * max(0.1, pad_m + float(padding_h)))
    x1_ideal = cx - raw_w / 2.0

    extra_bottom_px = bh * float(padding_bottom)
    y1_ideal = by1 - margin_top
    y2_ideal = by2 + margin_bottom + extra_bottom_px

    if y2_ideal <= y1_ideal + 16:
        y2_ideal = y1_ideal + 16.0
    raw_h = y2_ideal - y1_ideal

    if raw_w >= img_w:
        x1, x2 = 0, img_w
    else:
        x1 = int(round(x1_ideal))
        x2 = int(round(x1 + raw_w))
        if x1 < 0:
            x2 = min(img_w, x2 - x1)
            x1 = 0
        elif x2 > img_w:
            x1 = max(0, x1 - (x2 - img_w))
            x2 = img_w

    if raw_h >= img_h:
        y1, y2 = 0, img_h
    else:
        y1 = int(round(y1_ideal))
        y2 = int(round(y1 + raw_h))
        if y1 < 0:
            shift = -y1
            y1 = 0
            y2 = min(img_h, y2 + shift)
        elif y2 > img_h:
            y2 = img_h
            y1 = max(0, int(round(y2 - raw_h)))

    if x1 <= 2:
        x1 = 0
    if y1 <= 2:
        y1 = 0
    if x2 >= img_w - 2:
        x2 = img_w
    if y2 >= img_h - 2:
        y2 = img_h

    crop_w = max(16, x2 - x1)
    crop_h = max(16, y2 - y1)
    target_ratio_val = crop_w / float(crop_h)
    best_ratio_name = f"{target_ratio_val:.2f}"

    target_pixels = megapixels * 1_000_000
    gen_h = math.sqrt(target_pixels / target_ratio_val)
    gen_w = gen_h * target_ratio_val
    gen_w = max(16, int(round(gen_w / 16.0)) * 16)
    gen_h = max(16, int(round(gen_h / 16.0)) * 16)

    return (x1, y1, x2, y2), (gen_w, gen_h), best_ratio_name


def draw_debug_visualization(
    image: Image.Image,
    detected_boxes,
    selected_indices,
    crop_box,
    ratio_name,
    gen_size,
):
    w, h = image.size
    scale = PREVIEW_MAX_DIM / float(max(w, h)) if max(w, h) > PREVIEW_MAX_DIM else 1.0
    vis_img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.HAMMING).convert("RGB")
    draw = ImageDraw.Draw(vis_img)
    gen_w, gen_h = gen_size

    for idx, (bx1, by1, bx2, by2, label) in enumerate(detected_boxes):
        is_selected = idx in selected_indices
        box_color = "#00FF00" if is_selected else "#666666"
        line_w = 3 if is_selected else 1
        sbx1, sby1 = int(bx1 * scale), int(by1 * scale)
        sbx2, sby2 = int(bx2 * scale), int(by2 * scale)
        draw.rectangle([sbx1, sby1, sbx2, sby2], outline=box_color, width=line_w)
        txt = f"#{idx + 1}" + (" [FOCO]" if is_selected else "")
        draw.rectangle([sbx1, max(0, sby1 - 16), sbx1 + (len(txt) * 6) + 6, max(0, sby1)], fill=box_color)
        draw.text((sbx1 + 2, max(0, sby1 - 14)), txt, fill="#000000")

    cx1, cy1, cx2, cy2 = crop_box
    scx1, scy1 = int(cx1 * scale), int(cy1 * scale)
    scx2, scy2 = int(cx2 * scale), int(cy2 * scale)
    draw.rectangle([scx1, scy1, scx2, scy2], outline="#00FFFF", width=3)
    tag = f"[{gen_w}x{gen_h}]"
    draw.rectangle([scx1, scy1, scx1 + (len(tag) * 6) + 10, scy1 + 18], fill="#00FFFF")
    draw.text((scx1 + 4, scy1 + 2), tag, fill="#000000")
    return vis_img


def blend_crop_into_original(original_img, edited_crop, crop_box, blend_radius=24):
    x1, y1, x2, y2 = crop_box
    target_w, target_h = x2 - x1, y2 - y1
    resized_crop = edited_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    orig_w, orig_h = original_img.size

    touch_left = x1 <= 2
    touch_top = y1 <= 2
    touch_right = x2 >= orig_w - 2
    touch_bottom = y2 >= orig_h - 2

    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)

    r_left = 0 if touch_left else blend_radius
    r_top = 0 if touch_top else blend_radius
    r_right = 0 if touch_right else blend_radius
    r_bottom = 0 if touch_bottom else blend_radius

    draw.rectangle([r_left, r_top, target_w - r_right, target_h - r_bottom], fill=255)
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(blend_radius / 2))

    draw_fix = ImageDraw.Draw(mask_blurred)
    safety_margin = blend_radius * 2
    if touch_left:
        draw_fix.rectangle([0, 0, safety_margin, target_h], fill=255)
    if touch_top:
        draw_fix.rectangle([0, 0, target_w, safety_margin], fill=255)
    if touch_right:
        draw_fix.rectangle([max(0, target_w - safety_margin), 0, target_w, target_h], fill=255)
    if touch_bottom:
        draw_fix.rectangle([0, max(0, target_h - safety_margin), target_w, target_h], fill=255)

    reconstituted = original_img.copy().convert("RGB")
    reconstituted.paste(resized_crop, (x1, y1), mask=mask_blurred)
    return reconstituted


def list_loras_on_disk() -> List[str]:
    try:
        os.makedirs(LORA_DIR, exist_ok=True)
        files = sorted([f for f in os.listdir(LORA_DIR) if f.endswith(".safetensors")])
        return files
    except Exception:
        return []


# ------------------------------------------------------------------------------
# 5. Workflow de Geração (sd-server ou Modo Debug Sólido)
# ------------------------------------------------------------------------------
def generate_image_workflow(
    prompt: str,
    input_image: Optional[Image.Image],
    crop_data: Dict[str, Any],
    steps: int,
    seed: int,
    selected_loras: Optional[List[str]],
    lora_weights_list: List[float],
    shift_mode: str,
    shift_val: float,
    shift_offset: float,
    time_shift_type: str,
    stochastic_sampling: bool,
    sigma_schedule: str,
    use_custom_sigmas: bool,
    custom_sigmas: List[float],
    use_custom_system: bool,
    diffusion_system_prompt: str,
    use_inpaint_pipeline: bool,
    inpaint_strength: float,
    use_attr_in_prompt: bool,
    detected_attr_tag: str,
    use_pose_in_prompt: bool,
    detected_pose_tag: str,
    debug_color: str = "#FF00FF",
    progress=gr.Progress(track_tqdm=True),
):
    if sd_backend is None:
        init_runtime(start_server=not DEBUG_MODE)
    assert sd_backend is not None

    final_prompt = prompt
    tags_to_append = []
    if use_attr_in_prompt and detected_attr_tag and detected_attr_tag.strip():
        tags_to_append.append(detected_attr_tag.strip())
    if use_pose_in_prompt and detected_pose_tag and detected_pose_tag.strip():
        tags_to_append.append(detected_pose_tag.strip())

    if tags_to_append:
        tag_str = ", ".join(tags_to_append)
        final_prompt = f"{final_prompt}, {tag_str}" if final_prompt.strip() else tag_str

    seed = max(0, min(MAX_SEED, int(seed)))
    steps = int(steps)

    is_editing = input_image is not None and bool(crop_data) and "crop_box" in crop_data
    if is_editing:
        crop_box = crop_data["crop_box"]
        gen_w, gen_h = crop_data["gen_size"]
        cropped_input = input_image.crop(crop_box).resize((gen_w, gen_h), Image.Resampling.LANCZOS)
        reference_image = cropped_input
        width, height = gen_w, gen_h
    else:
        cropped_input = None
        reference_image = None
        width, height = 1024, 1024

    mu_val, _ = calculate_mu_preview(shift_mode, shift_val, shift_offset, 1.0, width, height)

    face_mask: Optional[Image.Image] = None
    face_mask_preview = None
    mask_overlay_preview = None
    init_img_for_backend = None
    mask_img_for_backend = None

    if is_editing and use_inpaint_pipeline and cropped_input is not None:
        progress(0.05, desc="Recuperando máscara facial em cache...")
        if "cached_mask" in crop_data and crop_data["cached_mask"] is not None:
            face_mask = crop_data["cached_mask"]
            if face_mask.size != (width, height):
                face_mask = face_mask.resize((width, height), Image.Resampling.BILINEAR)
        else:
            face_mask, _, _, _ = generate_face_preserve_mask(target_img=cropped_input)

        face_mask_preview = compress_webp_level1(face_mask)
        mask_overlay_preview = compress_webp_level1(crop_data.get("cached_overlay", None))

        if float(inpaint_strength) < 1.0:
            init_img_for_backend = cropped_input
            mask_img_for_backend = face_mask

    sigmas_to_pass: Optional[List[float]] = None
    if use_custom_sigmas and shift_mode == "Estático":
        if not custom_sigmas or len(custom_sigmas) != steps:
            timesteps = np.linspace(1, 1000, steps, dtype=np.float32)[::-1].copy()
            sigmas = timesteps / 1000.0
            sigmas_shifted = shift_val * sigmas / (1 + (shift_val - 1) * sigmas)
            custom_sigmas = [float(s) for s in sigmas_shifted]
        sigmas_to_pass = custom_sigmas[:steps]

    sys_prompt_to_pass = diffusion_system_prompt if use_custom_system else None
    clean_loras = [f for f in (selected_loras or []) if f]
    loras_tag = (
        "+".join(f"{os.path.splitext(f)[0][:10]}({w:.2f})" for f, w in zip(clean_loras, lora_weights_list))
        if clean_loras
        else "None"
    )

    progress(0.06, desc=f"Inferindo em {width}x{height} ({'DEBUG' if DEBUG_MODE else 'sd-server'})...")

    result_crop = sd_backend.generate(
        prompt=final_prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        flow_shift=mu_val,
        sigma_schedule=sigma_schedule,
        stochastic_sampling=stochastic_sampling,
        custom_sigmas=sigmas_to_pass,
        ref_image=reference_image,
        init_image=init_img_for_backend,
        mask_image=mask_img_for_backend,
        inpaint_strength=float(inpaint_strength),
        selected_loras=clean_loras,
        lora_weights=lora_weights_list,
        system_prompt=sys_prompt_to_pass,
        debug_color=debug_color,
        progress_callback=lambda pct, msg: progress(pct, desc=msg),
    )

    progress(0.92, desc="Aplicando máscara facial e reconstruindo saída...")

    if is_editing and cropped_input is not None:
        if use_inpaint_pipeline and face_mask is not None:
            # Preservação facial suave em espaço de pixel respeitando o feathering da elipse:
            # preto (0) na face_mask preserva cropped_input (face); branco (255) aplica result_crop.
            result_crop = Image.composite(
                result_crop.resize((width, height), Image.Resampling.LANCZOS),
                cropped_input.convert("RGB"),
                face_mask.convert("L"),
            )

        final_image_uncompressed = blend_crop_into_original(
            original_img=input_image,
            edited_crop=result_crop,
            crop_box=crop_data["crop_box"],
        )
    else:
        final_image_uncompressed = result_crop

    res_mp_val = round((width * height) / 1_000_000.0, 2)
    clean_prompt_abbrev = (final_prompt[:32] + "...") if len(final_prompt) > 32 else final_prompt
    mode_badge = f"[DEBUG COR {debug_color}] " if DEBUG_MODE else "[9B sd-server] "

    status_card = (
        f"{mode_badge}"
        f"Prompt: \"{clean_prompt_abbrev}\" | "
        f"Steps: {steps} | "
        f"Seed: {seed} | "
        f"Res: {width}x{height} ({res_mp_val}MP) | "
        f"Shift: {mu_val:.2f} | "
        f"Sched: {sigma_schedule} | "
        f"SDE: {'ON' if stochastic_sampling else 'OFF'} | "
        f"LoRAs: [{loras_tag}]"
    )

    progress(1.0, desc="Concluído!")
    yield final_image_uncompressed, mask_overlay_preview, face_mask_preview, status_card


# ------------------------------------------------------------------------------
# 6. Construção da UI Gradio
# ------------------------------------------------------------------------------
def build_gradio_ui():
    with gr.Blocks(title="Flux Klein 9B (stable-diffusion.cpp) - ver2.0", fill_width=True) as demo:
        if DEBUG_MODE:
            gr.Markdown(
                "### 🧪 Flux Klein 9B — **MODO DEBUG LOCAL** (`stable-diffusion.cpp` NÃO instalado; geração retorna imagem de cor sólida)\n"
                "Todas as detecções YOLO (pessoas, rosto, atributos e pose), máscaras e blend continuam funcionando normalmente."
            )
        else:
            gr.Markdown("### Flux Klein 9B (`stable-diffusion.cpp` GGUF) — ver2.0 (YOLOv8 + YOLOv8-Face + Atributos + Pose)")

        crop_data_state = gr.State(value={})
        det_cache = gr.State(value={"sig": None, "raw_boxes": [], "conf_t": 0.35})
        selected_ids_state = gr.State(value=[0])
        shift_mode_state = gr.State(value="Dynamic (Resolução)")
        attr_label_state = gr.State(value="")

        with gr.Row():
            with gr.Column(scale=2):
                prompt_in = gr.Textbox(
                    label="Prompt de Edição",
                    lines=3,
                    value="Remove clothes, dress, shirt and pants. naked. nipples. ",
                    placeholder="Descreva as alterações...",
                )
                with gr.Row():
                    pose_posture_radio = gr.Radio(
                        choices=["em pé", "sentado", "deitado", "N/A"],
                        value="N/A",
                        label="Change Posture",
                        interactive=True,
                    )
                    pose_facing_radio = gr.Radio(
                        choices=["de frente", "de costas", "de lado", "N/A"],
                        value="N/A",
                        label="Change Orientation",
                        interactive=True,
                    )

                with gr.Row():
                    steps_in = gr.Slider(minimum=1, maximum=50, step=1, value=4, label="Steps")
                    seed_in = gr.Number(value=42, label="Seed", precision=0)
                    mp_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.025, value=1.05, label="MP")

                debug_color_in = gr.ColorPicker(
                    value="#FF00FF",
                    label="🎨 Cor Sólida da Geração (Modo Debug)",
                    visible=DEBUG_MODE,
                )

                with gr.Row():
                    crop_status = gr.Textbox(label="Status do Recorte", value="Aguardando imagem...", interactive=False)

                image_in = gr.Image(label="Imagem de Entrada", type="pil", sources=["upload", "clipboard"])

                with gr.Accordion("Pessoas Detectadas", open=True):
                    card_columns, card_images, card_buttons = [], [], []
                    with gr.Row():
                        for i in range(MAX_PERSON_CARDS):
                            with gr.Column(scale=1, min_width=60, visible=False) as col:
                                img_comp = gr.Image(**_image_widget_kwargs(buttons=[], interactive=False, show_label=False))
                                btn_comp = gr.Button(
                                    value="✓ Ativo" if i == 0 else "+ Incluir",
                                    variant="primary" if i == 0 else "secondary",
                                    size="sm",
                                )
                                card_columns.append(col)
                                card_images.append(img_comp)
                                card_buttons.append(btn_comp)

                with gr.Tabs():
                    with gr.Tab("Detecções"):
                        bbox_visualizer = gr.Image(
                            **_image_widget_kwargs(buttons=[], label="Enquadramento Ciano", interactive=False, height=300)
                        )
                    with gr.Tab("Recorte"):
                        crop_patch_preview = gr.Image(
                            **_image_widget_kwargs(buttons=[], label="Recorte Isolado", interactive=False, height=300)
                        )
                    with gr.Tab("Máscara Facial"):
                        with gr.Row():
                            mask_overlay_preview = gr.Image(
                                **_image_widget_kwargs(buttons=[], label="Overlay Rotacionado", interactive=False, height=300)
                            )
                            mask_preview_tab = gr.Image(
                                **_image_widget_kwargs(buttons=[], label="Máscara Facial P&B", interactive=False, height=300)
                            )

                    with gr.Tab("Parâmetros de Detecção"):
                        with gr.Accordion("👤 1. Detecção de Pessoas (YOLOv8)", open=True):
                            with gr.Row():
                                yolo_conf_slider = gr.Slider(
                                    minimum=0.10, maximum=0.90, step=0.05, value=0.25, label="Confiança de Detecção (Conf)"
                                )
                                crop_padding_slider = gr.Slider(
                                    minimum=1.0, maximum=2.5, step=0.05, value=1.25, label="Padding Base (Bounding Box)"
                                )
                            with gr.Row():
                                crop_padding_h_slider = gr.Slider(
                                    minimum=-0.8,
                                    maximum=1.5,
                                    step=0.05,
                                    value=0.0,
                                    label="Padding Horizontal Extra (- reduz / + adiciona)",
                                )
                                crop_padding_bottom_slider = gr.Slider(
                                    minimum=-0.8,
                                    maximum=1.5,
                                    step=0.05,
                                    value=0.0,
                                    label="Padding Inferior Extra (- reduz / + adiciona)",
                                )

                        with gr.Accordion("🎭 2. Preservação Facial & Inpainting", open=True):
                            with gr.Row():
                                use_inpaint_toggle = gr.Checkbox(value=True, label="Ativar Preservação Facial Automática (Inpaint)")
                                enable_safety_ellipse_cb = gr.Checkbox(
                                    value=False, label="Ativar Elipse de Segurança (se rosto não detectado)"
                                )
                            with gr.Row():
                                inpaint_strength_slider = gr.Slider(
                                    minimum=0.1, maximum=1.0, step=0.05, value=1.0, label="Inpaint Strength"
                                )
                                face_blur_pct_slider = gr.Slider(
                                    minimum=2.0, maximum=50.0, step=1.0, value=20.0, label="Feathering da Borda (% da Face)"
                                )
                                face_expand_slider = gr.Slider(
                                    minimum=0.8, maximum=2.0, step=0.05, value=1.35, label="Escala da Máscara Facial"
                                )

                        with gr.Accordion("🏷️ 3. Detecção de Atributos Corporais (YOLO-attr)", open=True):
                            with gr.Row():
                                attr_include_cb = gr.Checkbox(label="Incluir Atributo Corporal no Prompt", value=True)
                                attr_conf_slider = gr.Slider(
                                    minimum=0.10, maximum=0.90, step=0.05, value=0.25, label="Confiança Mínima Atributo"
                                )
                                attr_imgsz_dropdown = gr.Dropdown(choices=[320, 480, 640, 800], value=640, label="Resolução imgsz")

                        with gr.Accordion("🏃 4. Postura e Orientação (YOLO-pose)", open=True):
                            with gr.Row():
                                pose_include_cb = gr.Checkbox(label="Incluir Pose no Prompt", value=True)
                                pose_conf_slider = gr.Slider(
                                    minimum=0.10, maximum=0.90, step=0.05, value=0.25, label="Confiança Mínima Pose"
                                )

                            gr.Markdown("##### Textos Customizáveis para Injeção no Prompt")
                            with gr.Group():
                                gr.Markdown("**Postura (Em pé / Sentado / Deitado)**")
                                with gr.Group():
                                    with gr.Row():
                                        pose_txt_standing = gr.Textbox(value="standing", label="Texto se 'Em pé'")
                                        pose_txt_sitting = gr.Textbox(value="sitting", label="Texto se 'Sentado'")
                                        pose_txt_lying = gr.Textbox(value="lying down", label="Texto se 'Deitado'")

                                gr.Markdown("**Orientação (Frente / Costas / Lado)**")
                                with gr.Group():
                                    with gr.Row():
                                        pose_txt_front = gr.Textbox(value="front view", label="Texto se 'De frente'")
                                        pose_txt_back = gr.Textbox(value="back view", label="Texto se 'De costas'")
                                        pose_txt_side = gr.Textbox(value="side view", label="Texto se 'De lado'")

                with gr.Accordion("Configurações do Scheduler & Live Monitor", open=True):
                    with gr.Tabs() as shift_tabs:
                        with gr.Tab("Estático") as tab_static:
                            sched_shift_val = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Shift Fixo Estático")
                            use_custom_sigmas_cb = gr.Checkbox(label="Usar Custom Sigmas (Sobrescreve Shift Estático)", value=True)
                            custom_sigmas_state = gr.State(value=DEFAULT_SIGMAS_LIST)

                            with gr.Column(visible=True) as custom_sigmas_group:
                                @gr.render(inputs=[steps_in])
                                def dynamic_sigmas_ui(steps):
                                    steps = int(steps)
                                    calc_sigmas_btn = gr.Button("Calcular Sigmas do Shift Selecionado")

                                    sliders = []
                                    for i in range(steps):
                                        init_val = DEFAULT_SIGMAS_LIST[i] if i < len(DEFAULT_SIGMAS_LIST) else 1.0
                                        sl = gr.Slider(
                                            minimum=0.0,
                                            maximum=1.0,
                                            step=0.0001,
                                            value=init_val,
                                            interactive=True,
                                            label=f"Sigma Step {i + 1}",
                                        )
                                        sliders.append(sl)

                                    def calc_sigmas(shift_val):
                                        timesteps = np.linspace(1, 1000, steps, dtype=np.float32)[::-1].copy()
                                        sigmas = timesteps / 1000.0
                                        sigmas_shifted = shift_val * sigmas / (1 + (shift_val - 1) * sigmas)
                                        return [float(s) for s in sigmas_shifted]

                                    def sync_state(*vals):
                                        return list(vals)

                                    calc_sigmas_btn.click(fn=calc_sigmas, inputs=[sched_shift_val], outputs=sliders)
                                    gr.Markdown("### Configurações de Sigmas")
                                    gr.Markdown("#### 94 // 85 // 60 // 22")

                                    for sl in sliders:
                                        sl.change(fn=sync_state, inputs=sliders, outputs=[custom_sigmas_state])

                        with gr.Tab("Dynamic (Resolução)") as tab_dynamic:
                            with gr.Group():
                                with gr.Row():
                                    sched_shift_offset = gr.Slider(
                                        minimum=-3, maximum=3, step=0.05, value=0, scale=2, label="Shift Offset"
                                    )
                                    sched_time_shift_type = gr.Dropdown(
                                        choices=["exponential", "linear"], value="exponential", scale=1, label="Função de Shift"
                                    )
                                with gr.Row():
                                    preview_base = gr.Textbox(
                                        label="Shift Piso 0.5MP", value=f"{DEFAULT_BASE_SHIFT:.2f}", interactive=False
                                    )
                                    preview_max = gr.Textbox(
                                        label="Shift Teto 2.0MP", value=f"{DEFAULT_MAX_SHIFT:.2f}", interactive=False
                                    )

                    mu_live_preview = gr.Textbox(label="📊 Shift Efetivo Resultante (μ)", value="μ = 0.825 (D/Res)", interactive=False)

                    sched_sigma_schedule = gr.Radio(
                        choices=["Default", "Karras", "Beta", "Exponential"], value="Default", label="Distribuição dos Sigmas"
                    )
                    sched_stochastic = gr.Checkbox(value=False, label="Stochastic Sampling (Euler A)")

                with gr.Accordion("LoRA Management (Multi-LoRA)", open=True):
                    disk_loras = list_loras_on_disk()
                    default_selected = [DEFAULT_LORA_NAME] if DEFAULT_LORA_NAME and DEFAULT_LORA_NAME in disk_loras else []
                    lora_dropdown = gr.Dropdown(
                        choices=disk_loras,
                        value=default_selected,
                        label="LoRAs Disponíveis (FLUX.2 Klein 9B)",
                        multiselect=True,
                    )
                    lora_slider_comps = []
                    for s_idx in range(MAX_LORA_SLIDERS):
                        is_visible = s_idx < len(default_selected)
                        sl = gr.Slider(
                            minimum=-1.0,
                            maximum=2.0,
                            step=0.05,
                            value=1.0,
                            label=f"Peso: {default_selected[s_idx]}" if is_visible else f"Peso LoRA #{s_idx + 1}",
                            visible=is_visible,
                        )
                        lora_slider_comps.append(sl)

                    refresh_loras_btn = gr.Button("Atualizar Lista de LoRAs")

                with gr.Accordion("Configurações Avançadas (Qwen3-VL-8B Prompting)", open=False):
                    use_system_prompt_cb = gr.Checkbox(value=False, label="Prefixar System Prompt na Difusão")
                    system_prompt_tb = gr.Textbox(
                        label="Instrução de Sistema (Qwen3)",
                        lines=2,
                        value="You are an expert image editor that remove clothes. Keep face and body and contour. Describe all the body parts visible in detail.",
                    )

            with gr.Column(scale=3):
                generate_btn = gr.Button("Gerar", variant="primary", size="lg")

                output_image_raw = gr.Image(
                    **_image_widget_kwargs(buttons=["fullscreen", "download"], label="Imagem Final Gerada")
                )

                status_text = gr.Label(value="Status: Pronto" + (" (Modo Debug)" if DEBUG_MODE else ""), label="Status da Geração")

        # ----------------------------------------------------------------------
        # Callbacks e Sincronização
        # ----------------------------------------------------------------------
        tab_dynamic.select(lambda: "Dynamic (Resolução)", outputs=[shift_mode_state])
        tab_static.select(lambda: "Estático", outputs=[shift_mode_state])

        use_custom_sigmas_cb.change(
            fn=lambda use: gr.update(visible=use),
            inputs=[use_custom_sigmas_cb],
            outputs=[custom_sigmas_group],
        )

        def on_shift_param_change(mode, offset, s_val, crop_data, mp_val):
            gen_w, gen_h = crop_data.get("gen_size", (1024, 1024)) if crop_data else (1024, 1024)
            b = round(max(0.01, DEFAULT_BASE_SHIFT + float(offset)), 3)
            m = round(max(0.01, DEFAULT_MAX_SHIFT + float(offset)), 3)
            _, desc = calculate_mu_preview(mode, s_val, offset, mp_val, gen_w, gen_h)
            return f"{b:.2f}", f"{m:.2f}", desc

        for comp in [sched_shift_offset, sched_shift_val, shift_mode_state]:
            comp.change(
                fn=on_shift_param_change,
                inputs=[shift_mode_state, sched_shift_offset, sched_shift_val, crop_data_state, mp_slider],
                outputs=[preview_base, preview_max, mu_live_preview],
                show_progress="hidden",
            )

        def update_lora_sliders(selected_list):
            selected = selected_list or []
            updates = []
            for i in range(MAX_LORA_SLIDERS):
                if i < len(selected):
                    updates.append(gr.update(visible=True, label=f"Peso: {selected[i]}", value=1.0))
                else:
                    updates.append(gr.update(visible=False, value=1.0))
            return updates

        lora_dropdown.change(fn=update_lora_sliders, inputs=[lora_dropdown], outputs=lora_slider_comps)

        def refresh_lora_dropdown(current_sel):
            available = list_loras_on_disk()
            for item in current_sel or []:
                if item not in available:
                    available.append(item)
            return gr.update(choices=available)

        refresh_loras_btn.click(fn=refresh_lora_dropdown, inputs=[lora_dropdown], outputs=[lora_dropdown])

        def update_crop_from_selection(
            image,
            selected_indices,
            raw_boxes,
            mp_val,
            pad_mult,
            pad_h,
            pad_bottom,
            blur_pct,
            exp_ratio,
            mode,
            offset,
            s_val,
            enable_safety_ellipse,
            attr_conf,
            attr_imgsz,
            pose_conf,
        ):
            if image is None or not raw_boxes:
                return (
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    "Sem dados",
                    {},
                    gr.update(),
                    "",
                    gr.update(label="Incluir Atributo", value=False),
                    gr.update(value="N/A"),
                    gr.update(value="N/A"),
                )

            valid_indices = [idx for idx in selected_indices if 0 <= idx < len(raw_boxes)]
            if not valid_indices:
                valid_indices = [0]

            chosen_boxes = [raw_boxes[i] for i in valid_indices]
            combined_bbox = (
                min(b[0] for b in chosen_boxes),
                min(b[1] for b in chosen_boxes),
                max(b[2] for b in chosen_boxes),
                max(b[3] for b in chosen_boxes),
            )

            crop_box, gen_size, ratio_name = calculate_crop_and_resolution(
                img_w=image.width,
                img_h=image.height,
                bbox=combined_bbox,
                megapixels=mp_val,
                padding_mult=pad_mult,
                padding_h=pad_h,
                padding_bottom=pad_bottom,
            )

            raw_viz = draw_debug_visualization(image, raw_boxes, valid_indices, crop_box, ratio_name, gen_size)
            gen_w, gen_h = gen_size

            cropped_patch = image.crop(crop_box).resize((gen_w, gen_h), Image.Resampling.LANCZOS)

            detected_attr = detect_body_attribute(cropped_patch, conf_threshold=float(attr_conf), imgsz=int(attr_imgsz))
            detected_pose = detect_pose_yolo(cropped_patch, conf=float(pose_conf))

            mask, overlay, mask_tag, face_info = generate_face_preserve_mask(
                target_img=cropped_patch,
                blur_fraction_pct=float(blur_pct),
                expand_ratio=float(exp_ratio),
                cached_face_info=None,
                enable_safety_ellipse=bool(enable_safety_ellipse),
            )

            debug_viz = compress_webp_level1(raw_viz)
            crop_patch_fast = compress_webp_level1(cropped_patch)
            overlay_fast = compress_webp_level1(overlay)
            mask_fast = compress_webp_level1(mask)

            new_crop_data = {
                "crop_box": crop_box,
                "gen_size": gen_size,
                "ratio_name": ratio_name,
                "face_info": face_info,
                "cached_mask": mask,
                "cached_overlay": overlay,
            }

            attr_disp = f" | Attr: {detected_attr}" if detected_attr else ""

            posture_val = detected_pose.get("posture", "") or "N/A"
            facing_val = detected_pose.get("facing", "") or "N/A"

            pose_summary = ""
            if posture_val != "N/A" or facing_val != "N/A":
                p = posture_val if posture_val != "N/A" else ""
                f = facing_val if facing_val != "N/A" else ""
                pose_summary = f"{p} {f}".strip()

            pose_disp = f" | Pose: {pose_summary}" if pose_summary else ""
            status_msg = f"{len(valid_indices)} pessoa(s) | {gen_w}x{gen_h} | {mask_tag}{attr_disp}{pose_disp}"
            _, mu_desc = calculate_mu_preview(mode, s_val, offset, mp_val, gen_w, gen_h)

            toggle_attr_lbl = f"Incluir: {detected_attr}" if detected_attr else "Atributo não detectado"

            return (
                debug_viz,
                crop_patch_fast,
                overlay_fast,
                mask_fast,
                status_msg,
                new_crop_data,
                mu_desc,
                detected_attr,
                gr.update(label=toggle_attr_lbl, value=bool(detected_attr)),
                gr.update(value=posture_val),
                gr.update(value=facing_val),
            )

        def on_detection_change(
            image,
            mp_val,
            conf_t,
            pad_mult,
            pad_h,
            pad_bottom,
            blur_pct,
            exp_ratio,
            cache,
            mode,
            offset,
            s_val,
            enable_safety_ellipse,
            attr_conf,
            attr_imgsz,
            pose_conf,
        ):
            card_updates = []
            if image is None:
                for _ in range(MAX_PERSON_CARDS):
                    card_updates.extend([gr.update(visible=False), gr.update(value=None), gr.update(value="", variant="secondary")])
                return (
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    "Aguardando imagem...",
                    {},
                    {"sig": None, "raw_boxes": [], "conf_t": conf_t},
                    [0],
                    gr.update(),
                    "",
                    gr.update(label="Incluir Atributo", value=False),
                    "N/A",
                    "N/A",
                    *card_updates,
                )

            img_signature = compute_image_signature(image)

            if cache.get("sig") != img_signature or cache.get("conf_t") != conf_t or not cache.get("raw_boxes"):
                detected_boxes, debug_msg = detect_person_bbox(image, conf_threshold=conf_t)
                cache = {"sig": img_signature, "raw_boxes": detected_boxes, "conf_t": conf_t}
            else:
                detected_boxes = cache["raw_boxes"]
                debug_msg = "Cache mantido"

            total_detected = len(detected_boxes)
            initial_selection = [0]

            for idx in range(MAX_PERSON_CARDS):
                if idx < total_detected:
                    bx1, by1, bx2, by2, _ = detected_boxes[idx]
                    raw_thumb = image.crop((bx1, by1, bx2, by2))
                    thumb_compressed = compress_webp_level3(raw_thumb, thumb_size=(110, 220))
                    is_active = idx == 0
                    card_updates.extend(
                        [
                            gr.update(visible=True),
                            gr.update(value=thumb_compressed),
                            gr.update(
                                value="✓ Ativo" if is_active else "+ Incluir",
                                variant="primary" if is_active else "secondary",
                            ),
                        ]
                    )
                else:
                    card_updates.extend([gr.update(visible=False), gr.update(value=None), gr.update(value="", variant="secondary")])

            (
                debug_viz,
                cropped_patch,
                overlay_fast,
                mask_fast,
                status_msg,
                new_crop_data,
                mu_desc,
                attr_lbl,
                toggle_attr_upd,
                posture_val,
                facing_val,
            ) = update_crop_from_selection(
                image,
                initial_selection,
                detected_boxes,
                mp_val,
                pad_mult,
                pad_h,
                pad_bottom,
                blur_pct,
                exp_ratio,
                mode,
                offset,
                s_val,
                enable_safety_ellipse,
                attr_conf,
                attr_imgsz,
                pose_conf,
            )

            return (
                debug_viz,
                cropped_patch,
                overlay_fast,
                mask_fast,
                f"{debug_msg} | {status_msg}",
                new_crop_data,
                cache,
                initial_selection,
                mu_desc,
                attr_lbl,
                toggle_attr_upd,
                posture_val,
                facing_val,
                *card_updates,
            )

        def on_mp_slider_change(
            selected_indices,
            image,
            mp_val,
            pad_mult,
            pad_h,
            pad_bottom,
            blur_pct,
            exp_ratio,
            cache,
            mode,
            offset,
            s_val,
            enable_safety_ellipse,
            attr_conf,
            attr_imgsz,
            pose_conf,
        ):
            raw_boxes = cache.get("raw_boxes", [])
            return update_crop_from_selection(
                image,
                selected_indices,
                raw_boxes,
                mp_val,
                pad_mult,
                pad_h,
                pad_bottom,
                blur_pct,
                exp_ratio,
                mode,
                offset,
                s_val,
                enable_safety_ellipse,
                attr_conf,
                attr_imgsz,
                pose_conf,
            )

        def make_toggle_handler(card_idx: int):
            def toggle(
                current_sel,
                image,
                mp_val,
                pad_mult,
                pad_h,
                pad_bottom,
                blur_pct,
                exp_ratio,
                cache,
                mode,
                offset,
                s_val,
                enable_safety_ellipse,
                attr_conf,
                attr_imgsz,
                pose_conf,
            ):
                raw = cache.get("raw_boxes", [])
                if not raw or card_idx >= len(raw):
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "Sem alteração",
                        {},
                        current_sel,
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        *[gr.update() for _ in range(MAX_PERSON_CARDS)],
                    )

                selected = list(current_sel)
                if card_idx in selected:
                    if len(selected) > 1:
                        selected.remove(card_idx)
                else:
                    selected.append(card_idx)

                (
                    debug_viz,
                    cropped_patch,
                    overlay_fast,
                    mask_fast,
                    status_msg,
                    new_crop_data,
                    mu_desc,
                    attr_lbl,
                    toggle_attr_upd,
                    posture_val,
                    facing_val,
                ) = update_crop_from_selection(
                    image,
                    selected,
                    raw,
                    mp_val,
                    pad_mult,
                    pad_h,
                    pad_bottom,
                    blur_pct,
                    exp_ratio,
                    mode,
                    offset,
                    s_val,
                    enable_safety_ellipse,
                    attr_conf,
                    attr_imgsz,
                    pose_conf,
                )

                btn_updates = []
                for i in range(MAX_PERSON_CARDS):
                    if i < len(raw):
                        is_sel = i in selected
                        btn_updates.append(
                            gr.update(
                                value=f"✓ #{i + 1} Ativo" if is_sel else f"+ #{i + 1} Incluir",
                                variant="primary" if is_sel else "secondary",
                            )
                        )
                    else:
                        btn_updates.append(gr.update())

                return (
                    debug_viz,
                    cropped_patch,
                    overlay_fast,
                    mask_fast,
                    status_msg,
                    new_crop_data,
                    selected,
                    mu_desc,
                    attr_lbl,
                    toggle_attr_upd,
                    posture_val,
                    facing_val,
                    *btn_updates,
                )

            return toggle

        setup_card_outputs = []
        for col, img, btn in zip(card_columns, card_images, card_buttons):
            setup_card_outputs.extend([col, img, btn])

        det_inputs = [
            image_in,
            mp_slider,
            yolo_conf_slider,
            crop_padding_slider,
            crop_padding_h_slider,
            crop_padding_bottom_slider,
            face_blur_pct_slider,
            face_expand_slider,
            det_cache,
            shift_mode_state,
            sched_shift_offset,
            sched_shift_val,
            enable_safety_ellipse_cb,
            attr_conf_slider,
            attr_imgsz_dropdown,
            pose_conf_slider,
        ]
        det_outputs = [
            bbox_visualizer,
            crop_patch_preview,
            mask_overlay_preview,
            mask_preview_tab,
            crop_status,
            crop_data_state,
            det_cache,
            selected_ids_state,
            mu_live_preview,
            attr_label_state,
            attr_include_cb,
            pose_posture_radio,
            pose_facing_radio,
            *setup_card_outputs,
        ]

        for trigger in [
            image_in.change,
            yolo_conf_slider.release,
            crop_padding_slider.release,
            crop_padding_h_slider.release,
            crop_padding_bottom_slider.release,
            face_blur_pct_slider.release,
            face_expand_slider.release,
            enable_safety_ellipse_cb.change,
            attr_conf_slider.release,
            attr_imgsz_dropdown.change,
            pose_conf_slider.release,
        ]:
            trigger(fn=on_detection_change, inputs=det_inputs, outputs=det_outputs)

        mp_slider.release(
            fn=on_mp_slider_change,
            inputs=[
                selected_ids_state,
                image_in,
                mp_slider,
                crop_padding_slider,
                crop_padding_h_slider,
                crop_padding_bottom_slider,
                face_blur_pct_slider,
                face_expand_slider,
                det_cache,
                shift_mode_state,
                sched_shift_offset,
                sched_shift_val,
                enable_safety_ellipse_cb,
                attr_conf_slider,
                attr_imgsz_dropdown,
                pose_conf_slider,
            ],
            outputs=[
                bbox_visualizer,
                crop_patch_preview,
                mask_overlay_preview,
                mask_preview_tab,
                crop_status,
                crop_data_state,
                mu_live_preview,
                attr_label_state,
                attr_include_cb,
                pose_posture_radio,
                pose_facing_radio,
            ],
            show_progress="hidden",
        )

        for idx, btn in enumerate(card_buttons):
            btn.click(
                fn=make_toggle_handler(idx),
                inputs=[
                    selected_ids_state,
                    image_in,
                    mp_slider,
                    crop_padding_slider,
                    crop_padding_h_slider,
                    crop_padding_bottom_slider,
                    face_blur_pct_slider,
                    face_expand_slider,
                    det_cache,
                    shift_mode_state,
                    sched_shift_offset,
                    sched_shift_val,
                    enable_safety_ellipse_cb,
                    attr_conf_slider,
                    attr_imgsz_dropdown,
                    pose_conf_slider,
                ],
                outputs=[
                    bbox_visualizer,
                    crop_patch_preview,
                    mask_overlay_preview,
                    mask_preview_tab,
                    crop_status,
                    crop_data_state,
                    selected_ids_state,
                    mu_live_preview,
                    attr_label_state,
                    attr_include_cb,
                    pose_posture_radio,
                    pose_facing_radio,
                    *card_buttons,
                ],
                show_progress="minimal",
                show_progress_on=[bbox_visualizer, crop_patch_preview],
            )

        def prepare_and_generate(
            prompt,
            attr_toggle,
            attr_tag,
            pose_toggle,
            pose_posture,
            pose_facing,
            txt_standing,
            txt_sitting,
            txt_lying,
            txt_front,
            txt_back,
            txt_side,
            img,
            crop_data,
            steps,
            seed,
            selected_loras,
            *extra_args,
        ):
            lora_weights = list(extra_args[:MAX_LORA_SLIDERS])
            rest = extra_args[MAX_LORA_SLIDERS:]

            shift_mode, shift_val, shift_offset, time_shift_type = rest[0:4]
            stochastic_sampling, sigma_schedule = rest[4:6]
            use_custom_sigmas = rest[6]
            custom_sigmas = rest[7]
            use_sys, sys_prompt = rest[8:10]
            use_inpaint, inpaint_str = rest[10:12]
            debug_color = rest[12]

            pose_data = {
                "posture": pose_posture if pose_posture != "N/A" else "",
                "facing": pose_facing if pose_facing != "N/A" else "",
            }

            resolved_pose_tag = format_pose_prompt_tag(
                pose_data=pose_data,
                txt_standing=txt_standing,
                txt_sitting=txt_sitting,
                txt_lying=txt_lying,
                txt_front=txt_front,
                txt_back=txt_back,
                txt_side=txt_side,
            )

            yield from generate_image_workflow(
                prompt=prompt,
                input_image=img,
                crop_data=crop_data,
                steps=steps,
                seed=seed,
                selected_loras=selected_loras,
                lora_weights_list=lora_weights,
                shift_mode=shift_mode,
                shift_val=shift_val,
                shift_offset=shift_offset,
                time_shift_type=time_shift_type,
                stochastic_sampling=stochastic_sampling,
                sigma_schedule=sigma_schedule,
                use_custom_sigmas=use_custom_sigmas,
                custom_sigmas=custom_sigmas,
                use_custom_system=use_sys,
                diffusion_system_prompt=sys_prompt,
                use_inpaint_pipeline=use_inpaint,
                inpaint_strength=inpaint_str,
                use_attr_in_prompt=attr_toggle,
                detected_attr_tag=attr_tag,
                use_pose_in_prompt=pose_toggle,
                detected_pose_tag=resolved_pose_tag,
                debug_color=debug_color,
            )

        generate_btn.click(
            fn=prepare_and_generate,
            inputs=[
                prompt_in,
                attr_include_cb,
                attr_label_state,
                pose_include_cb,
                pose_posture_radio,
                pose_facing_radio,
                pose_txt_standing,
                pose_txt_sitting,
                pose_txt_lying,
                pose_txt_front,
                pose_txt_back,
                pose_txt_side,
                image_in,
                crop_data_state,
                steps_in,
                seed_in,
                lora_dropdown,
                *lora_slider_comps,
                shift_mode_state,
                sched_shift_val,
                sched_shift_offset,
                sched_time_shift_type,
                sched_stochastic,
                sched_sigma_schedule,
                use_custom_sigmas_cb,
                custom_sigmas_state,
                use_system_prompt_cb,
                system_prompt_tb,
                use_inpaint_toggle,
                inpaint_strength_slider,
                debug_color_in,
            ],
            outputs=[output_image_raw, mask_overlay_preview, mask_preview_tab, status_text],
            show_progress="full",
            show_progress_on=[output_image_raw],
        )

    return demo


def _start_cloudflare_tunnel(port: int, token: Optional[str]) -> Optional[subprocess.Popen]:
    if token and token.strip():
        print("🚀 [1/2] Iniciando Túnel Cloudflare com token...", flush=True)
        cmd = [
            "pycloudflared",
            "tunnel",
            "--region",
            "us",
            "run",
            "--http2-origin",
            "--token",
            token.strip(),
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        from pycloudflared import try_cloudflare

        print("🚀 [1/2] Iniciando Túnel TryCloudflare público...", flush=True)
        tunnel_url = try_cloudflare(port=port)
        print(f"🌐 URL Pública Cloudflare: {tunnel_url}", flush=True)
    except Exception as exc:
        print(f"⚠️ Não foi possível iniciar pycloudflared ({exc}). Use --share ou túnel manual.", flush=True)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GradioKlein 9B UI")
    parser.add_argument("--debug", action="store_true", help="Ativa Modo Debug (cor sólida sem stable-diffusion.cpp)")
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    parser.add_argument("--lora-dir", default=os.environ.get("LORA_DIR", LORA_DIR))
    parser.add_argument("--sdcpp-dir", default=os.environ.get("SDCPP_DIR", "./sdcpp_bin"))
    parser.add_argument("--sdcpp-port", type=int, default=int(os.environ.get("SDCPP_PORT", "8188")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Ativa link Gradio share")
    parser.add_argument("--cloudflare", action="store_true", help="Inicia túnel Cloudflare via pycloudflared")
    parser.add_argument("--tunnel-token", default=os.environ.get("CF_TUNNEL_TOKEN", ""), help="Token opcional do Cloudflare Tunnel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_runtime(
        debug_mode=args.debug or DEBUG_MODE,
        models_dir=args.models_dir,
        lora_dir=args.lora_dir,
        sdcpp_dir=args.sdcpp_dir,
        sdcpp_port=args.sdcpp_port,
        start_server=not (args.debug or DEBUG_MODE),
    )

    if args.cloudflare:
        _start_cloudflare_tunnel(port=args.port, token=args.tunnel_token)

    print("🚀 [2/2] Iniciando APP Gradio...", flush=True)
    demo = build_gradio_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True,
    )


if __name__ == "__main__":
    main()
