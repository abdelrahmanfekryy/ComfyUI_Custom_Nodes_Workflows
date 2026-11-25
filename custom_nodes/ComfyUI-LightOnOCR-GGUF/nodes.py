import gc
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import folder_paths
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("LightOnOCR")

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "ocr_models.json"
LLM_DIR = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(LLM_DIR, exist_ok=True)
folder_paths.folder_names_and_paths["llm"] = ([LLM_DIR], {".gguf"})

_model_cache = {}


def _load_config():
    default = {"base_dir": "llm", "models": {}}
    if not CONFIG_PATH.exists():
        return default
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or default
    except Exception as e:
        logger.warning("Failed to load config: %s", e)
        return default


CONFIG = _load_config()


# --- Path helpers ---


def _get_llm_paths():
    """Return resolved LLM search paths with fallback logic."""
    try:
        paths = folder_paths.get_folder_paths("llm")
        if paths:
            return paths
    except Exception:
        pass

    base_path = folder_paths.models_dir
    return [os.path.join(base_path, "llm")]


def _resolve_file(filename: str) -> str | None:
    """Resolve a filename to its full path across all registered LLM dirs."""
    for path in _get_llm_paths():
        full = os.path.join(path, filename)
        if os.path.exists(full):
            return full
    return None


def _get_model_files():
    all_files = folder_paths.get_filename_list("llm")
    models = sorted(f for f in all_files if not f.startswith("mmproj-"))
    if not models:
        models = ["(download .gguf files to " + LLM_DIR + ")"]
    return models


def _get_mmproj_files():
    all_files = folder_paths.get_filename_list("llm")
    mmprojs = sorted(f for f in all_files if f.startswith("mmproj-"))
    if not mmprojs:
        mmprojs = ["(download mmproj-*.gguf to " + LLM_DIR + ")"]
    return mmprojs


# --- Image helpers ---


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    import base64
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_for_ocr(pil_img: Image.Image, max_longest: int = 1540) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) > max_longest:
        scale = max_longest / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img


def _batch_to_b64(image: torch.Tensor | None, max_side: int) -> list[str | None]:
    """One base64 PNG per image in the batch; [None] for text-only inference."""
    if image is None:
        return [None]
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return [_image_to_base64(_resize_for_ocr(_tensor_to_pil(img), max_longest=max_side)) for img in image]


def _join_pages(texts: list[str]) -> str:
    """Join per-page OCR text with a Markdown page break."""
    if len(texts) == 1:
        return texts[0]
    return "\n\n---\n\n".join(f"<!-- page {i} -->\n\n{t}" for i, t in enumerate(texts, 1))


# --- Public handle types ---


class OCRModel:
    """Handle for a loaded GGUF model path."""
    def __init__(self, model_path: str):
        self.model_path = model_path


class OCRProjector:
    """Handle for a loaded mmproj path."""
    def __init__(self, projector_path: str):
        self.projector_path = projector_path


# --- LLM loading & inference ---


def _build_chat_handler(mmproj_path: str, image_max_tokens: int):
    """Try chat handlers in order: Pixtral → Mistral3 → Qwen25VL."""
    sig = __import__("inspect").signature

    # 1. PixtralChatHandler
    try:
        from llama_cpp.llama_chat_format import PixtralChatHandler
        init_sig = sig(PixtralChatHandler.__init__)
        kwargs = {"clip_model_path": mmproj_path, "image_max_tokens": image_max_tokens}
        kwargs = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
        logger.info("Using PixtralChatHandler")
        return PixtralChatHandler(**kwargs)
    except ImportError:
        pass

    # 2. Mistral3ChatHandler
    try:
        from llama_cpp.llama_chat_format import Mistral3ChatHandler
        init_sig = sig(Mistral3ChatHandler.__init__)
        kwargs = {"clip_model_path": mmproj_path, "image_max_tokens": image_max_tokens}
        kwargs = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
        logger.info("Using Mistral3ChatHandler")
        return Mistral3ChatHandler(**kwargs)
    except ImportError:
        pass

    # 3. Qwen25VLChatHandler (fallback)
    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        init_sig = sig(Qwen25VLChatHandler.__init__)
        kwargs = {"clip_model_path": mmproj_path, "image_max_tokens": image_max_tokens}
        kwargs = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
        logger.info("Using Qwen25VLChatHandler (fallback)")
        return Qwen25VLChatHandler(**kwargs)
    except ImportError:
        pass

    raise RuntimeError(
        "No VL chat handler found in llama-cpp-python. "
        "Install a multimodal-capable build: pip install llama-cpp-python"
    )


def _load_llm(model_path: str, mmproj_path: str, n_ctx: int, n_gpu_layers: int, n_batch: int, image_max_tokens: int):
    """Load GGUF model with mmproj projector. Returns Llama instance."""
    from llama_cpp import Llama

    init_sig = __import__("inspect").signature(Llama.__init__)

    chat_handler = _build_chat_handler(mmproj_path, image_max_tokens)

    llm_kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": max(n_batch, 512),
        "chat_handler": chat_handler,
        "verbose": False,
    }
    llm_kwargs = {k: v for k, v in llm_kwargs.items() if k in init_sig.parameters}

    logger.info("Loading: %s | mmproj: %s", Path(model_path).name, Path(mmproj_path).name)
    logger.info("ctx=%d gpu_layers=%d batch=%d img_tokens=%d", n_ctx, n_gpu_layers, n_batch, image_max_tokens)

    return Llama(**llm_kwargs)


def _clear_cache(llm):
    """Release a cached Llama model and free VRAM."""
    close_fn = getattr(llm, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _get_or_load_model(model_path: str, mmproj_path: str, n_ctx: int, n_gpu_layers: int, n_batch: int, image_max_tokens: int, keep_loaded: bool):
    """Get a model from cache or load it. Returns (llm, was_cached)."""
    cache_key = f"{model_path}_{n_ctx}_{n_gpu_layers}_{n_batch}_{image_max_tokens}"

    if keep_loaded and cache_key in _model_cache:
        logger.info("Using cached model: %s", cache_key)
        return _model_cache[cache_key], True

    if not keep_loaded and cache_key in _model_cache:
        _clear_cache(_model_cache[cache_key])
        del _model_cache[cache_key]

    llm = _load_llm(model_path, mmproj_path, n_ctx, n_gpu_layers, n_batch, image_max_tokens)

    if keep_loaded:
        _model_cache[cache_key] = llm
        logger.info("Model cached: %s", cache_key)

    return llm, False


# llama.cpp's mtmd wraps LightOnOCR image embeddings in <|im_start|> ... <|im_end|>
# (tools/mtmd/mtmd.cpp), not the <|vision_pad|>/<|vision_end|> tokens the model was
# trained with, so the first generated token is often a stray ":" or "." before the text.
_LEADING_ARTIFACT_RE = re.compile(r"^\s*(?::|\.(?=\s))\s*")


def _strip_leading_artifact(text: str) -> str:
    """Remove a stray leading ':' (or '.' followed by whitespace) and surrounding whitespace."""
    return _LEADING_ARTIFACT_RE.sub("", text, count=1).strip()


def run_inference(
    llm,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    """Run a single multimodal inference pass."""
    if image_b64:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]

    start = time.perf_counter()
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
        seed=seed,
    )
    elapsed = time.perf_counter() - start

    content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = content[0].get("text", str(content))

    usage = result.get("usage", {})
    logger.info(
        "%d prompt / %d completion tokens, %.1fs",
        usage.get("prompt_tokens", "?"),
        usage.get("completion_tokens", "?"),
        elapsed,
    )

    return _strip_leading_artifact(content)


# --- Model Loader Node ---


class OCRGGUFModelLoader:
    """Loads a GGUF text model. Pass the output to OCR Inference nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _get_model_files()
        return {
            "required": {
                "model_name": (models, {
                    "default": models[0],
                    "tooltip": "Select the GGUF model file to use. Models are loaded from ComfyUI/models/LLM/ directory.",
                }),
            },
        }

    RETURN_TYPES = ("OCR_MODEL",)
    RETURN_NAMES = ("ocr_model",)
    FUNCTION = "execute"
    CATEGORY = "LightOnOCR/Loaders"

    def execute(self, model_name):
        model_path = _resolve_file(model_name)
        if model_path is None:
            raise FileNotFoundError(f"Model '{model_name}' not found in any LLM path.")
        return (OCRModel(model_path),)


# --- Projector Loader Node ---


class OCRProjectorLoader:
    """Loads a mmproj projector file. Pass the output to OCR Inference nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        mmprojs = _get_mmproj_files()
        return {
            "required": {
                "mmproj_name": (mmprojs, {
                    "default": mmprojs[0],
                    "tooltip": "Select the mmproj projector file. Must match the quantization of the GGUF model (e.g., mmproj-BF16.gguf for BF16 model).",
                }),
            },
        }

    RETURN_TYPES = ("OCR_PROJECTOR",)
    RETURN_NAMES = ("ocr_projector",)
    FUNCTION = "execute"
    CATEGORY = "LightOnOCR/Loaders"

    def execute(self, mmproj_name):
        projector_path = _resolve_file(mmproj_name)
        if projector_path is None:
            raise FileNotFoundError(f"mmproj '{mmproj_name}' not found in any LLM path.")
        return (OCRProjector(projector_path),)


# --- Inference Nodes ---


class OCRInference:
    """Run OCR inference with a loaded model + projector."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocr_model": ("OCR_MODEL", {"tooltip": "Loaded GGUF model from OCRGGUFModelLoader."}),
                "ocr_projector": ("OCR_PROJECTOR", {"tooltip": "Loaded mmproj projector from OCRProjectorLoader."}),
                "prompt": (
                    "STRING",
                    {
                        "default": "Extract all text from this document. Preserve layout and formatting.",
                        "multiline": True,
                        "tooltip": "The OCR prompt sent to the model.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 32768,
                        "step": 256,
                        "tooltip": "Maximum number of tokens to generate.",
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep the model cached in memory for faster subsequent runs.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 1,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Random seed for reproducible outputs.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to perform OCR on. If omitted, runs text-only inference."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "LightOnOCR"

    def execute(self, ocr_model, ocr_projector, prompt, max_tokens, keep_loaded, seed, image=None):
        b64_pages = _batch_to_b64(image, 1540)

        llm, _ = _get_or_load_model(
            model_path=ocr_model.model_path,
            mmproj_path=ocr_projector.projector_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_batch=512,
            image_max_tokens=4096,
            keep_loaded=keep_loaded,
        )

        texts = [
            run_inference(
                llm=llm,
                system_prompt="You are a specialized OCR assistant. Output only the extracted text from the image.",
                user_prompt=prompt,
                image_b64=b64,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                seed=seed,
            )
            for b64 in b64_pages
        ]

        return (_join_pages(texts),)


class OCRInferenceAdvanced(OCRInference):
    """Run OCR inference with all parameters exposed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocr_model": ("OCR_MODEL", {"tooltip": "Loaded GGUF model from OCRGGUFModelLoader."}),
                "ocr_projector": ("OCR_PROJECTOR", {"tooltip": "Loaded mmproj projector from OCRProjectorLoader."}),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a specialized OCR assistant. Output only the extracted text from the image.",
                        "multiline": True,
                        "tooltip": "System instruction that defines the model's behavior.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Extract all text from this document. Preserve layout and formatting.",
                        "multiline": True,
                        "tooltip": "The OCR prompt sent to the model.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 32768,
                        "step": 256,
                        "tooltip": "Maximum number of tokens to generate.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. Lower = more deterministic. Higher = more creative.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Nucleus sampling threshold. Lower values produce more focused output.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Penalty for repeating tokens. Values > 1.0 reduce repetition.",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 2048,
                        "max": 131072,
                        "step": 1024,
                        "tooltip": "Context window size. Larger values allow longer context but use more memory.",
                    },
                ),
                "n_batch": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 32768,
                        "step": 64,
                        "tooltip": "Batch size for prompt processing. Larger batches process faster but use more memory.",
                    },
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 200,
                        "tooltip": "Number of layers to offload to GPU. -1 = all layers, 0 = CPU only.",
                    },
                ),
                "image_max_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 1024000,
                        "step": 256,
                        "tooltip": "Maximum image tokens for vision encoding. Higher = more detail, more memory.",
                    },
                ),
                "image_max_side": (
                    "INT",
                    {
                        "default": 1540,
                        "min": 512,
                        "max": 4096,
                        "step": 10,
                        "tooltip": "Max longest side for image resizing before OCR.",
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep the model cached in memory for faster subsequent runs.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 1,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Random seed for reproducible outputs.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to perform OCR on. If omitted, runs text-only inference."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute_advanced"
    CATEGORY = "LightOnOCR"

    def execute_advanced(
        self,
        ocr_model,
        ocr_projector,
        system_prompt,
        prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        n_ctx,
        n_batch,
        n_gpu_layers,
        image_max_tokens,
        image_max_side,
        keep_loaded,
        seed,
        image=None,
    ):
        b64_pages = _batch_to_b64(image, image_max_side)

        llm, _ = _get_or_load_model(
            model_path=ocr_model.model_path,
            mmproj_path=ocr_projector.projector_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            image_max_tokens=image_max_tokens,
            keep_loaded=keep_loaded,
        )

        texts = [
            run_inference(
                llm=llm,
                system_prompt=system_prompt,
                user_prompt=prompt,
                image_b64=b64,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
            for b64 in b64_pages
        ]

        return (_join_pages(texts),)


# --- Image Loader Node ---


class OCRLoadImage:
    """Load an image from the input directory for OCR."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"))
        )
        if not files:
            files = ["(put images in ComfyUI/input/)"]

        return {
            "required": {
                "image": (
                    files,
                    {
                        "default": files[0],
                        "tooltip": "Image file to load from ComfyUI/input/ directory.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "LightOnOCR"

    def execute(self, image):
        input_dir = folder_paths.get_input_directory()
        filepath = os.path.join(input_dir, image)
        pil = Image.open(filepath).convert("RGB")
        tensor = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)
        return (tensor,)


# --- Document Loader Node (PDF + images, with upload) ---


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")
DOCUMENT_EXTS = (".pdf",) + IMAGE_EXTS


def _list_input_documents():
    """All PDFs/images under ComfyUI/input (subfolders included), as relative paths."""
    input_dir = folder_paths.get_input_directory()
    files = []
    for root, _dirs, names in os.walk(input_dir):
        for name in names:
            if name.lower().endswith(DOCUMENT_EXTS):
                rel = os.path.relpath(os.path.join(root, name), input_dir)
                files.append(rel.replace(os.sep, "/"))
    return sorted(files)


def _parse_pages(spec: str, page_count: int) -> list[int]:
    """Parse '1', '1-3,5', or 'all' / '' / '0' into 0-based page indices."""
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "all", "*"):
        return list(range(page_count))
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a.strip() else 1
            end = int(b) if b.strip() else page_count
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    out = []
    for p in pages:
        if not 1 <= p <= page_count:
            raise ValueError(f"Page {p} is out of range: the document has {page_count} page(s).")
        if p - 1 not in out:
            out.append(p - 1)
    return out


def _render_pdf(path: str, pages_spec: str, dpi: int) -> tuple[list[Image.Image], int]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        total = len(pdf)
        images = []
        for idx in _parse_pages(pages_spec, total):
            page = pdf[idx]
            try:
                images.append(page.render(scale=dpi / 72.0).to_pil().convert("RGB"))
            finally:
                page.close()
        return images, total
    finally:
        pdf.close()


def _load_raster(path: str, pages_spec: str) -> tuple[list[Image.Image], int]:
    """Load an image file; multi-frame TIFF/GIF frames count as pages."""
    from PIL import ImageOps, ImageSequence

    with Image.open(path) as img:
        frames = [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(img)]
    return [frames[i] for i in _parse_pages(pages_spec, len(frames))], len(frames)


def _pils_to_batch(pils: list[Image.Image]) -> torch.Tensor:
    """Stack pages into one [B, H, W, C] batch, padding smaller pages with white."""
    w = max(p.width for p in pils)
    h = max(p.height for p in pils)
    batch = []
    for p in pils:
        if p.size != (w, h):
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            canvas.paste(p, (0, 0))
            p = canvas
        batch.append(torch.from_numpy(np.array(p).astype(np.float32) / 255.0))
    return torch.stack(batch)


class OCRLoadDocument:
    """Upload/load a PDF or image for OCR. PDF pages are rendered to images."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_documents() or ["(upload a PDF or image)"]
        return {
            "required": {
                "document": (files, {
                    "tooltip": "PDF or image in ComfyUI/input/. Use the upload button or drop a file on the node.",
                }),
                "pages": ("STRING", {
                    "default": "all",
                    "tooltip": "Pages to load (1-based): 'all', '1', '2-4', '1,3,5-7'. Applies to PDFs and multi-frame TIFF/GIF.",
                }),
                "dpi": ("INT", {
                    "default": 200, "min": 72, "max": 600, "step": 1,
                    "tooltip": "Render resolution for PDF pages. 200 is a good default for OCR.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "page_count")
    OUTPUT_TOOLTIPS = (
        "Selected pages as an image batch (one image per page).",
        "Total number of pages in the document.",
    )
    FUNCTION = "execute"
    CATEGORY = "LightOnOCR"
    OUTPUT_NODE = True  # so the rendered page preview shows on the node

    @staticmethod
    def _path(document):
        return folder_paths.get_annotated_filepath(document)

    @classmethod
    def IS_CHANGED(cls, document, pages, dpi):
        import hashlib
        path = cls._path(document)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return f"{h.hexdigest()}|{pages}|{dpi}"

    @classmethod
    def VALIDATE_INPUTS(cls, document, pages):
        if not folder_paths.exists_annotated_filepath(document):
            return f"File not found: {document}"
        if not document.lower().endswith(DOCUMENT_EXTS):
            return f"Unsupported file type: {document}"
        if any(c not in "0123456789,- " for c in pages.strip().lower()) and pages.strip().lower() not in ("all", "*"):
            return f"Invalid pages value: {pages!r}. Use 'all', '1', '2-4' or '1,3,5-7'."
        return True

    def execute(self, document, pages, dpi):
        path = self._path(document)
        if path.lower().endswith(".pdf"):
            pils, total = _render_pdf(path, pages, dpi)
        else:
            pils, total = _load_raster(path, pages)
        if not pils:
            raise ValueError(f"No pages selected from {document} (pages={pages!r}).")

        # Page previews for the node UI (first 16 pages).
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        stem = "ocr_doc_" + "".join(c if c.isalnum() else "_" for c in Path(document).stem)[:40]
        previews = []
        for i, p in enumerate(pils[:16]):
            name = f"{stem}_{int(time.time() * 1000)}_{i}.png"
            thumb = p.copy()
            thumb.thumbnail((1024, 1024))
            thumb.save(os.path.join(temp_dir, name), compress_level=1)
            previews.append({"filename": name, "subfolder": "", "type": "temp"})

        logger.info("Loaded %s: %d of %d page(s)", document, len(pils), total)
        return {"ui": {"images": previews}, "result": (_pils_to_batch(pils), total)}


# --- Text Display Node ---

def get_dict_value(data: dict, dict_key: str, default=None):
  """Gets a deeply nested value given a dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  found = data[key] if key in data else None
  if found is not None and len(keys) > 0:
    return get_dict_value(found, '.'.join(keys), default)
  return found if found is not None else default


class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False
  
any_type = AnyType("*")

class OCRDisplayAny:
    """Display any data type in the UI."""

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
        return {
            "required": {
                "source": (any_type, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
            }

    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True

    def main(self, source=None, unique_id=None, extra_pnginfo=None):
        value = 'None'
        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = 'source exists, but could not be serialized.'

        # Save the output to the pnginfo so it's pre-filled when loading the data.
        if extra_pnginfo and unique_id:
            for node in get_dict_value(extra_pnginfo, 'workflow.nodes', []):
                if str(node['id']) == str(unique_id):
                    node['widgets_values'] = [value]
                    break

        return {"ui": {"text": (value,)}}