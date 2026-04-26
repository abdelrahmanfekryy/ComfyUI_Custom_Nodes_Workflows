import gc
import io
import json
import logging
import os
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

    return content.strip()


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
        if image is not None:
            pil = _tensor_to_pil(image)
            pil = _resize_for_ocr(pil)
            b64 = _image_to_base64(pil)
        else:
            b64 = None

        llm, _ = _get_or_load_model(
            model_path=ocr_model.model_path,
            mmproj_path=ocr_projector.projector_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_batch=512,
            image_max_tokens=4096,
            keep_loaded=keep_loaded,
        )

        text = run_inference(
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

        return (text,)


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
        if image is not None:
            pil = _tensor_to_pil(image)
            pil = _resize_for_ocr(pil, max_longest=image_max_side)
            b64 = _image_to_base64(pil)
        else:
            b64 = None

        llm, _ = _get_or_load_model(
            model_path=ocr_model.model_path,
            mmproj_path=ocr_projector.projector_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            image_max_tokens=image_max_tokens,
            keep_loaded=keep_loaded,
        )

        text = run_inference(
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

        return (text,)


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


# --- Text Display Node ---


class OCRShowText:
    """Display extracted text as a widget and save to output directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "LightOnOCR"
    OUTPUT_NODE = True

    def execute(self, text):
        output_dir = folder_paths.get_output_directory()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"ocr_result_{timestamp}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Text saved to: %s", filepath)
        return {"ui": {"text": [text]}, "result": (text,)}
