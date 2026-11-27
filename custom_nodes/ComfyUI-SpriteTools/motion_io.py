"""Sprite motions: extract one clean motion cycle from a video's DWPose skeleton, save it, load it again.

A motion file holds the body keypoints (OpenPose 18-point, video pixel coords) of exactly one cycle, already
smoothed and closed into a loop: the frame after the last one would be the first one again. Sprite Pose Loop
retargets it onto any character, so a motion is extracted once and reused.
"""
import json
import logging
import os

import numpy as np
import torch

import folder_paths
from comfy_api.latest import io, ui

from .pose_loop import (CYCLE_DETECT, CYCLE_WHOLE, MIRROR_OFF, _fill_and_smooth, _Rig, _sample, _video_track,
                        _VideoMotion)

MOTION_DIR = "sprite_motions"  # under the output folder
NONE = "(none)"


def _motion_dir():
    return os.path.join(folder_paths.get_output_directory(), MOTION_DIR)


def _motion_files():
    root = _motion_dir()
    found = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".json"):
                found.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return sorted(found)


def _to_pose_keypoint(frames, w, h, meta):
    """(N, 18, 2) array with NaN for unknown joints -> DWPose-style POSE_KEYPOINT list, tagged as a saved cycle."""
    out = []
    for pts in frames:
        flat = []
        for x, y in pts:
            flat += [0.0, 0.0, 0.0] if not np.isfinite(x) else [float(x), float(y), 1.0]
        out.append({"people": [{"pose_keypoints_2d": flat}], "canvas_width": int(w), "canvas_height": int(h),
                    "sprite_motion": meta})
    return out


def _from_pose_keypoint(motion):
    """POSE_KEYPOINT list -> (N, 18, 2) array with NaN for unknown joints, and the canvas size."""
    track, _ = _video_track(motion)
    return track, int(motion[0]["canvas_width"]), int(motion[0]["canvas_height"])


class SpriteMotionFromVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SpriteMotionFromVideo",
            display_name="Sprite Motion from Video",
            category="image/sprites",
            description="Turns DWPose on a video's frames into one clean, looping motion cycle: follows the main "
                        "person, fills in missed joints, smooths jitter, finds a complete repeat of the motion and "
                        "closes it into a loop (a walk across the frame becomes a walk on the spot). Save it with "
                        "Save Sprite Motion, then load it into the pose sprite workflow.",
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("video_pose", tooltip="DWPose on the video's frames (load the "
                                                                       "video at 24 fps)."),
                io.Combo.Input("video_cycle", options=[CYCLE_DETECT, CYCLE_WHOLE], default=CYCLE_DETECT,
                               tooltip="Find one complete repeat of the motion (a clip with several works best), "
                                       "or take the whole clip as one cycle (already trimmed to a loop)."),
                io.Float.Input("smoothing", default=1.0, min=0.0, max=10.0, step=0.1,
                               tooltip="Removes DWPose jitter (Gaussian, in frames). 0 = off."),
            ],
            outputs=[
                io.Custom("POSE_KEYPOINT").Output(display_name="motion"),
                io.Image.Output(display_name="preview", tooltip="The cycle's skeleton; loop it to check the seam."),
                io.Int.Output(display_name="frames", tooltip="Frames in the cycle (at the video's frame rate)."),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, video_pose, video_cycle, smoothing) -> io.NodeOutput:
        frames = video_pose if isinstance(video_pose, list) else [video_pose]
        w, h = int(frames[0]["canvas_width"]), int(frames[0]["canvas_height"])

        # the video's own skeleton is the rig: the most complete frame, gaps filled from its neighbours
        track, _ = _video_track(frames)
        filled = _fill_and_smooth(track, 0)
        counts = np.isfinite(track[:, :, 0]).sum(axis=1)
        if counts.max() == 0:
            raise ValueError("no person found in the video's pose keypoints; check its DWPose result")
        k = int(np.argmax(counts))
        valid = np.isfinite(filled[k, :, 0])
        rig = _Rig(np.nan_to_num(filled[k]), valid, None, [None, None])

        vm = _VideoMotion(rig, frames, MIRROR_OFF, smoothing, 1.0, video_cycle, False)
        a, n, count = vm.start, vm.period, vm.samples
        pos = a + np.arange(count + 1) * n / count  # last one = the start of the next cycle
        c = _sample(vm.track, pos)
        cycle = c[:-1] - (c[-1] - c[0]) * (np.arange(count) / count)[:, None, None]  # close the loop

        meta = {"cycle": True, "source_frames": [int(a), round(float(a + n), 2)], "smoothing": float(smoothing)}
        motion = _to_pose_keypoint(cycle, w, h, meta)
        preview = np.stack([rig.render(np.nan_to_num(p), w, h) for p in cycle])

        info = (f"{vm.frames_in} video frames ({vm.detected} with a person); cycle = video frames {a}-{a + n:.1f}, "
                f"saved as {count} frames")
        if np.isfinite(vm.score):
            info += f"; match {vm.score:.2f} (good is < 0.3)"
        if vm.no_cycle is not None:
            info += (f"\nno repeating cycle found ({vm.no_cycle}), so the whole video is used as one cycle: trim it "
                     f"to one cycle, or use a clip with several")
        missing = [i for i in range(18) if not np.isfinite(cycle[:, i, 0]).any()]
        if missing:
            info += f"\njoints never detected (Sprite Pose Loop keeps the character's own): {missing}"
        logging.info("SpriteMotionFromVideo: %s", info)
        return io.NodeOutput(motion, torch.from_numpy(preview).float() / 255.0, count, info)


class SaveSpriteMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveSpriteMotion",
            display_name="Save Sprite Motion",
            category="image/sprites",
            description=f"Saves a motion cycle as JSON in output/{MOTION_DIR}/, where Load Sprite Motion finds it.",
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("motion"),
                io.String.Input("name", default="walk", tooltip="File name prefix; a counter is added. A '/' makes "
                                                                 "a subfolder."),
            ],
            outputs=[io.String.Output(display_name="file")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, motion, name) -> io.NodeOutput:
        track, w, h = _from_pose_keypoint(motion)
        meta = dict(motion[0].get("sprite_motion") or {})
        if not meta.get("cycle"):
            raise ValueError("this motion is not a single cycle: connect Sprite Motion from Video's `motion`")
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            f"{MOTION_DIR}/{name.strip() or 'motion'}", folder_paths.get_output_directory())
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{filename}_{counter:05}_.json")
        data = {
            "sprite_motion": 1,
            "canvas_width": w, "canvas_height": h,
            "frame_count": int(track.shape[0]),
            **{k: v for k, v in meta.items() if k != "cycle"},
            # one row per frame: x0, y0, x1, y1, ... for the 18 OpenPose body joints; null = not detected
            "frames": [[None if not np.isfinite(v) else round(float(v), 2) for v in pts.reshape(-1)] for pts in track],
        }
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        rel = os.path.relpath(path, _motion_dir()).replace(os.sep, "/")
        logging.info("SaveSpriteMotion: saved %s", path)
        return io.NodeOutput(rel, ui=ui.PreviewText(f"saved {MOTION_DIR}/{rel} ({track.shape[0]} frames)"))


class LoadSpriteMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSpriteMotion",
            display_name="Load Sprite Motion",
            category="image/sprites",
            description=f"Loads a motion cycle saved by Save Sprite Motion (output/{MOTION_DIR}/), for Sprite Pose "
                        "Loop's video_pose. Press R in the editor to refresh the list after saving a new one.",
            inputs=[
                io.Combo.Input("motion_file", options=[NONE] + _motion_files(), default=NONE,
                               tooltip=f"Files in output/{MOTION_DIR}/."),
            ],
            outputs=[
                io.Custom("POSE_KEYPOINT").Output(display_name="motion"),
                io.Int.Output(display_name="frames", tooltip="Frames in the cycle."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, motion_file):
        path = os.path.join(_motion_dir(), motion_file)
        return os.path.getmtime(path) if os.path.isfile(path) else motion_file

    @classmethod
    def validate_inputs(cls, motion_file):
        if motion_file != NONE and not os.path.isfile(os.path.join(_motion_dir(), motion_file)):
            return f"motion file not found: output/{MOTION_DIR}/{motion_file}"
        return True

    @classmethod
    def execute(cls, motion_file) -> io.NodeOutput:
        if motion_file == NONE:
            return io.NodeOutput(None, 0)
        with open(os.path.join(_motion_dir(), motion_file)) as f:
            data = json.load(f)
        if data.get("sprite_motion") != 1:
            raise ValueError(f"{motion_file} is not a sprite motion file")
        rows = np.array([[np.nan if v is None else v for v in row] for row in data["frames"]], dtype=np.float64)
        meta = {"cycle": True, "file": motion_file}
        motion = _to_pose_keypoint(rows.reshape(len(rows), 18, 2), data["canvas_width"], data["canvas_height"], meta)
        return io.NodeOutput(motion, len(motion))
