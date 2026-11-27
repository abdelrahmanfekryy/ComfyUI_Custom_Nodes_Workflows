"""Sprite tools: pose-controlled loop videos and evenly spaced, seamlessly looping sprite frames."""
import logging
import math

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .iclora import AddICLoRAGuideLTX, CropGuidesLTX, ICLoRALoaderLTX
from .motion_io import LoadSpriteMotion, SaveSpriteMotion, SpriteMotionFromVideo
from .pose_loop import SpritePoseLoop

THUMB = 64  # long side of the thumbnails used to compare poses

LOOP_AUTO = "auto (find loop point)"
LOOP_WHOLE = "whole clip"
LOOP_RANGE = "pose cycle (loop_start/loop_end)"


def _thumbs(images: torch.Tensor) -> torch.Tensor:
    """[B, H, W, C] -> small [B, 3, h, w] float tensors for cheap frame comparison."""
    x = images[..., :3].movedim(-1, 1).float()
    h, w = x.shape[2:]
    s = THUMB / max(h, w)
    return F.interpolate(x, size=(max(1, round(h * s)), max(1, round(w * s))), mode="area")


class SpriteLoopFrames(io.ComfyNode):
    """Pick N evenly spaced frames covering exactly one cycle of a looping action.

    Video models rarely land the last frame exactly on the first, so in auto mode the cycle ends at the
    frame that looks most like frame 0 (searched after `min_cycle` of the clip). In pose-cycle mode the range
    comes from Sprite Pose Loop, which knows where its complete cycle lies. Frames are then sampled evenly
    over [loop_start, loop_end), so the sprite wraps from the last kept frame back to the first without a jump.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SpriteLoopFrames",
            display_name="Sprite Loop Frames",
            category="image/sprites",
            description="Finds where a generated action loops back to its first pose and samples N evenly spaced "
                        "frames from that one cycle.",
            inputs=[
                io.Image.Input("images", tooltip="Video frames, starting at the reference pose."),
                io.Int.Input("sprite_frames", default=8, min=1, max=256, tooltip="How many frames to keep."),
                io.Combo.Input("loop_mode", options=[LOOP_AUTO, LOOP_WHOLE, LOOP_RANGE], default=LOOP_AUTO,
                               tooltip="auto: end the cycle at the frame most similar to frame 0. "
                                       "whole clip: use every frame but the last (which repeats the first). "
                                       "pose cycle: use the complete cycle given by loop_start/loop_end "
                                       "(connect them from Sprite Pose Loop)."),
                io.Float.Input("min_cycle", default=0.3, min=0.05, max=1.0, step=0.05,
                               tooltip="Earliest loop point, as a fraction of the clip. Stops it from matching "
                                       "the frames right after the start, which always look like frame 0."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=1.0,
                               tooltip="Frame rate of the source video (connect GetVideoComponents fps)."),
                io.Int.Input("loop_start", default=0, min=0, max=100000, optional=True,
                             tooltip="pose cycle mode: first frame of the cycle."),
                io.Int.Input("loop_end", default=0, min=0, max=100000, optional=True,
                             tooltip="pose cycle mode: frame where the cycle has come full circle (not kept). "
                                     "0 = last frame."),
            ],
            outputs=[
                io.Image.Output(display_name="frames"),
                io.Int.Output(display_name="loop_end", tooltip="First frame of the next cycle (not included)."),
                io.Float.Output(display_name="playback_fps", tooltip="Frame rate that plays the sprite at the "
                                                                      "speed of the source video."),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, images, sprite_frames, loop_mode, min_cycle, fps, loop_start=0, loop_end=0) -> io.NodeOutput:
        count = images.shape[0]
        if count < 3:
            return io.NodeOutput(images, count, float(fps), f"only {count} frame(s), kept all")

        start = 0
        if loop_mode == LOOP_RANGE:
            end = loop_end if 0 < loop_end < count else count - 1
            start = min(loop_start, end - 1)
            mode_info = f"pose cycle, frames {start}-{end - 1}"
            if end != loop_end:
                mode_info += f" (loop_end {loop_end} is outside the {count}-frame clip, used the last frame)"
            loop_end = end
        elif loop_mode == LOOP_AUTO:
            t = _thumbs(images)
            diff = (t - t[:1]).abs().mean(dim=(1, 2, 3))  # 0..1 per frame
            search = min(max(2, math.ceil(min_cycle * (count - 1))), count - 1)
            loop_end = search + int(torch.argmin(diff[search:]).item())
            score = float(diff[loop_end]) * 255
            mode_info = f"loop point at frame {loop_end} (difference to frame 0: {score:.1f}/255)"
        else:
            loop_end = count - 1
            mode_info = f"whole clip, frames 0-{loop_end - 1}"

        length = loop_end - start
        n = min(sprite_frames, length)
        idx = [start + math.floor(i * length / n) for i in range(n)]
        frames = images[idx]
        playback_fps = float(fps) * n / length
        info = f"{mode_info}; kept {n} frames {idx}; play at {playback_fps:.2f} fps for real-time speed"
        if n < sprite_frames:
            info += f" (asked for {sprite_frames}, cycle has only {length} frames)"
        logging.info("SpriteLoopFrames: %s", info)
        return io.NodeOutput(frames, loop_end, playback_fps, info)


class SpriteToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SpriteLoopFrames, SpritePoseLoop, ICLoRALoaderLTX, AddICLoRAGuideLTX, CropGuidesLTX,
                SpriteMotionFromVideo, SaveSpriteMotion, LoadSpriteMotion]


async def comfy_entrypoint() -> SpriteToolsExtension:
    return SpriteToolsExtension()
