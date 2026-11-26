"""LTX-2 IC-LoRA support (pose / depth / canny control) for ComfyUI builds that lack the official nodes.

Core ComfyUI already handles small-grid IC-LoRA guides (latent_downscale_factor in the keyframe RoPE
positions, negative noise_mask entries dropped through grid_mask). What older builds lack is the two
wrapper nodes: one that reads the LoRA's reference_downscale_factor, and one that encodes the control
video at that reduced size and appends it as a guide.
"""
import logging

import torch

import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
from comfy_api.latest import io
from comfy_extras import nodes_lt


class ICLoRALoaderLTX(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ICLoRALoaderLTX",
            display_name="IC-LoRA Loader (LTX)",
            category="image/sprites",
            description="Loads an LTX-2 IC-LoRA and reads its reference_downscale_factor from the file metadata "
                        "(a '-ref0.5' LoRA expects the control video at half the output size, factor 2).",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Float.Output(display_name="latent_downscale_factor"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        try:
            factor = float((metadata or {})["reference_downscale_factor"])
        except (KeyError, TypeError, ValueError):
            factor = 1.0
            logging.warning("ICLoRALoaderLTX: no reference_downscale_factor in %s, using 1", lora_name)
        if strength_model != 0:
            model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return io.NodeOutput(model, factor)


def _dilate(latent: torch.Tensor, factor: int):
    """Spread a small-grid latent over the full grid: real tokens every `factor` cells, the rest masked
    out with -1 (core drops tokens whose noise mask is negative)."""
    b, c, t, h, w = latent.shape
    full = latent.new_zeros((b, c, t, h * factor, w * factor))
    full[..., ::factor, ::factor] = latent
    mask = latent.new_full((b, 1, t, h * factor, w * factor), -1.0)
    mask[..., ::factor, ::factor] = 1.0
    return full, mask


class AddICLoRAGuideLTX(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AddICLoRAGuideLTX",
            display_name="Add IC-LoRA Guide (LTX)",
            category="image/sprites",
            description="Adds a control video (pose skeleton, depth or edges) as an IC-LoRA guide aligned with the "
                        "generated clip. Use the video latent only (before concatenating audio). Works together "
                        "with first/last-frame image guides.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent", tooltip="Video latent (not the combined audio+video latent)."),
                io.Image.Input("control", tooltip="Control frames, same count and aspect ratio as the output clip."),
                io.Float.Input("latent_downscale_factor", default=1.0, min=1.0, max=8.0, step=1.0,
                               tooltip="From IC-LoRA Loader (LTX). 2 for '-ref0.5' LoRAs."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("frame_idx", default=0, min=0, max=9999,
                             tooltip="Pixel frame where the control starts. 0 = aligned with the clip."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, control, latent_downscale_factor, strength,
                frame_idx) -> io.NodeOutput:
        time_scale, width_scale, height_scale = vae.downscale_index_formula
        samples = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)
        _, _, latent_length, latent_h, latent_w = samples.shape

        factor = int(round(latent_downscale_factor))
        if latent_h % factor or latent_w % factor:
            raise ValueError(f"latent size {latent_w}x{latent_h} is not divisible by the downscale factor {factor}; "
                             f"make the video width/height multiples of {32 * factor}")

        keep = (control.shape[0] - 1) // time_scale * time_scale + 1
        control = control[:keep]
        pixels = comfy.utils.common_upscale(control.movedim(-1, 1), latent_w * width_scale // factor,
                                            latent_h * height_scale // factor, "bilinear", "disabled").movedim(1, -1)
        guide = vae.encode(pixels[..., :3])

        guide_mask = None
        if factor > 1:
            guide, guide_mask = _dilate(guide, factor)

        frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(positive, latent_length, keep, frame_idx,
                                                                       (time_scale, width_scale, height_scale))
        if latent_idx + guide.shape[2] > latent_length:
            raise ValueError(f"control video ({keep} frames) is longer than the clip; trim it or lengthen the clip")

        # No guide_attention_entries: core then requires one for every guide, and image guides added by other
        # nodes (e.g. LTXSequencer) do not register any.
        positive, negative, samples, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive, negative, frame_idx, samples, noise_mask, guide, strength,
            (time_scale, width_scale, height_scale), guide_mask=guide_mask,
            latent_downscale_factor=factor, causal_fix=frame_idx == 0,
        )
        return io.NodeOutput(positive, negative, {"samples": samples, "noise_mask": noise_mask})


class CropGuidesLTX(io.ComfyNode):
    """LTXVCropGuides counts guides by their distinct start times, so a first-frame image guide and a control
    video that both start at frame 0 are counted once and one guide latent stays glued to the clip. This
    crops back to the known clip length instead."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CropGuidesLTX",
            display_name="Crop Guides To Length (LTX)",
            category="image/sprites",
            description="Removes every appended guide latent by cutting the video latent back to the clip length. "
                        "Use instead of LTXVCropGuides when several guides start at the same frame.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent", tooltip="Video latent after sampling (separated from audio)."),
                io.Int.Input("frames", default=49, min=1, max=10000, tooltip="Clip length in pixel frames."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, latent, frames) -> io.NodeOutput:
        keep = (frames - 1) // 8 + 1  # LTX VAE: 8 pixel frames per latent frame, plus the first
        samples = latent["samples"][:, :, :keep].clone()
        noise_mask = nodes_lt.get_noise_mask(latent)[:, :, :keep]
        clear = {"keyframe_idxs": None, "guide_attention_entries": None}
        positive = node_helpers.conditioning_set_values(positive, clear)
        negative = node_helpers.conditioning_set_values(negative, clear)
        return io.NodeOutput(positive, negative, {"samples": samples, "noise_mask": noise_mask})
