import hashlib
import math
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io

MAX_LIST_IMAGES = 16

WEB_DIRECTORY = "./web"

LLAMA_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the "
    "user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _flatten_images(images):
    """Turn a list of IMAGE tensors (each possibly a batch) into a list of single-image tensors [1, H, W, C]."""
    flat = []
    for item in images or []:
        if item is None:
            continue
        for i in range(item.shape[0]):
            flat.append(item[i:i + 1])
    return flat


def _scale_to_area(samples, total_pixels, multiple=1):
    scale_by = math.sqrt(total_pixels / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by / multiple) * multiple
    height = round(samples.shape[2] * scale_by / multiple) * multiple
    return comfy.utils.common_upscale(samples, width, height, "area", "disabled")


class TextEncodeQwenImageEditMulti(io.ComfyNode):
    """Like TextEncodeQwenImageEditPlus, but takes any number of reference images.

    `images` may be a batch tensor, an image list, or a list of batches; each image becomes
    "Picture N" in the prompt. Images in a list can have different sizes.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditMulti",
            display_name="Text Encode Qwen Image Edit (Multi-Image)",
            category="advanced/conditioning",
            description="Qwen-Image-Edit text encoder that accepts any number of reference images "
                        "(a batch or a list). Images are numbered Picture 1, 2, 3... in input order.",
            is_input_list=True,
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("images", tooltip="A batch or a list of images. Qwen-Image-Edit 2511 works best with 1-3."),
            ],
            outputs=[
                io.Conditioning.Output(),
                io.Image.Output("first_image", tooltip="Picture 1, unscaled. Use it to size the output latent."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, images, vae=None) -> io.NodeOutput:
        # is_input_list: every input arrives as a list.
        clip = clip[0]
        prompt = prompt[0]
        vae = vae[0] if vae else None
        image_list = _flatten_images(images)
        if not image_list:
            raise ValueError("TextEncodeQwenImageEditMulti: no images were given.")

        ref_latents = []
        images_vl = []
        image_prompt = ""
        for i, image in enumerate(image_list):
            samples = image.movedim(-1, 1)
            images_vl.append(_scale_to_area(samples, 384 * 384).movedim(1, -1))
            if vae is not None:
                s = _scale_to_area(samples, 1024 * 1024, multiple=8)
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=LLAMA_TEMPLATE)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning, image_list[0])


class MakeImageList(io.ComfyNode):
    """Collects any number of images (of any size) into an image list. Connecting a slot adds a new one."""

    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image", min=1, max=MAX_LIST_IMAGES)
        return io.Schema(
            node_id="MakeImageListAutogrow",
            display_name="Make Image List",
            category="image/batch",
            description="Collects images of any size into a list, in slot order. Batched inputs are split into single images.",
            inputs=[io.Autogrow.Input("images", template=template)],
            outputs=[io.Image.Output(is_output_list=True)],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type) -> io.NodeOutput:
        ordered = [images[k] for k in sorted(images, key=lambda k: int(re.search(r"(\d+)$", k).group(1)))]
        return io.NodeOutput(_flatten_images(ordered))


def _parse_image_names(text):
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _resolve_input_image(name):
    """Path of an uploaded image, refusing anything outside ComfyUI's input directory."""
    path = os.path.realpath(folder_paths.get_annotated_filepath(name))
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    if os.path.commonpath([path, input_dir]) != input_dir:
        raise ValueError(f"Image is outside the input folder: {name}")
    return path


def _load_image(path):
    img = node_helpers.pillow(Image.open, path)
    img = node_helpers.pillow(ImageOps.exif_transpose, img)  # first frame only
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))
    rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    image = torch.from_numpy(rgb)[None,]
    if "A" in img.getbands():
        mask = 1.0 - torch.from_numpy(np.array(img.getchannel("A")).astype(np.float32) / 255.0)
    elif img.mode == "P" and "transparency" in img.info:
        mask = 1.0 - torch.from_numpy(np.array(img.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0)
    else:
        mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32)
    dtype = comfy.model_management.intermediate_dtype()
    return image.to(dtype=dtype), mask.unsqueeze(0).to(dtype=dtype)


class LoadImagesMultiUpload(io.ComfyNode):
    """Upload several images at once; outputs them as an image list (sizes may differ)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImagesMultiUpload",
            display_name="Load Images (Multi Upload)",
            category="image",
            description="Upload or drop several images; they are output as a list, in order. "
                        "Images can have different sizes.",
            inputs=[
                io.String.Input("images", multiline=True, default="",
                                tooltip="One file per line, from ComfyUI/input. Filled by the upload button; "
                                        "edit the lines to reorder or remove images."),
            ],
            outputs=[
                io.Image.Output("IMAGE", is_output_list=True),
                io.Mask.Output("MASK", is_output_list=True),
            ],
        )

    @classmethod
    def validate_inputs(cls, images):
        names = _parse_image_names(images)
        if not names:
            return "Load Images (Multi Upload): no images. Upload at least one."
        for name in names:
            try:
                path = _resolve_input_image(name)
            except ValueError as e:
                return str(e)
            if not os.path.isfile(path):
                return f"Invalid image file: {name}"
        return True

    @classmethod
    def fingerprint_inputs(cls, images):
        m = hashlib.sha256()
        for name in _parse_image_names(images):
            m.update(name.encode())
            with open(_resolve_input_image(name), "rb") as f:
                m.update(f.read())
        return m.hexdigest()

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        loaded = [_load_image(_resolve_input_image(n)) for n in _parse_image_names(images)]
        return io.NodeOutput([i for i, _ in loaded], [m for _, m in loaded])


class QwenImageEditMultiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TextEncodeQwenImageEditMulti, MakeImageList, LoadImagesMultiUpload]


async def comfy_entrypoint() -> QwenImageEditMultiExtension:
    return QwenImageEditMultiExtension()
