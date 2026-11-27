"""Sprite motions: extract one clean motion cycle from a video's DWPose skeleton, save it, load it again.

A motion file holds the body keypoints (OpenPose 18-point, video pixel coords) of exactly one cycle, already
smoothed and closed into a loop: the frame after the last one would be the first one again. Sprite Pose Loop
retargets it onto any character, so a motion is extracted once and reused.

Saving and opening go through the browser (web/sprite_motion.js): Save Sprite Motion hands the file to the page,
which asks where to save it; Load Sprite Motion's button uploads a chosen file into input/sprite_motions/.
"""
import json
import logging
import os

import numpy as np
import torch

import folder_paths
from comfy_api.latest import io

from .pose_loop import (CYCLE_DETECT, CYCLE_WHOLE, MIRROR_OFF, _fill_and_smooth, _Rig, _sample, _video_track,
                        _VideoMotion)

MOTION_DIR = "sprite_motions"  # Load Sprite Motion's folder under input/ (uploads); older saves under output/
OUTPUT_TAG = " [output]"
NONE = "(none)"
TARGET_FPS = 24  # the LTX sprite workflow's frame rate


def _json_files(root):
    found = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".json"):
                found.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return sorted(found)


def _motion_files():
    """Files opened into input/sprite_motions, then older saves in output/sprite_motions (tagged)."""
    files = _json_files(os.path.join(folder_paths.get_input_directory(), MOTION_DIR))
    files += [f + OUTPUT_TAG for f in _json_files(os.path.join(folder_paths.get_output_directory(), MOTION_DIR))]
    return files


def _motion_path(name):
    if name.endswith(OUTPUT_TAG):
        return os.path.join(folder_paths.get_output_directory(), MOTION_DIR, name[:-len(OUTPUT_TAG)])
    return os.path.join(folder_paths.get_input_directory(), MOTION_DIR, name)


def _motion_json(motion):
    """POSE_KEYPOINT cycle -> the motion file's JSON text."""
    track, w, h = _from_pose_keypoint(motion)
    meta = dict(motion[0].get("sprite_motion") or {})
    if not meta.get("cycle"):
        raise ValueError("this motion is not a single cycle: connect Sprite Motion from Video's `motion`")
    data = {
        "sprite_motion": 1,
        "canvas_width": w, "canvas_height": h,
        "frame_count": int(track.shape[0]),
        **{k: v for k, v in meta.items() if k not in ("cycle", "file")},
        # one row per frame: x0, y0, x1, y1, ... for the 18 OpenPose body joints; null = not detected
        "frames": [[None if not np.isfinite(v) else round(float(v), 2) for v in pts.reshape(-1)] for pts in track],
    }
    return json.dumps(data, separators=(",", ":")), int(track.shape[0])


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
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01,
                               tooltip="Frame rate the video was loaded at (connect Video Info's fps). The "
                                       "cycle is resampled to 24 fps, the sprite workflow's rate."),
            ],
            outputs=[
                io.Custom("POSE_KEYPOINT").Output(display_name="motion"),
                io.Image.Output(display_name="preview", tooltip="The cycle's skeleton; loop it to check the seam."),
                io.Int.Output(display_name="frames", tooltip="Frames in the cycle, at 24 fps."),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, video_pose, video_cycle, smoothing, fps=24.0) -> io.NodeOutput:
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
        a, n = vm.start, vm.period
        count = max(2, round(n * TARGET_FPS / fps))
        pos = a + np.arange(count + 1) * n / count  # last one = the start of the next cycle
        c = _sample(vm.cycle_source(vm.track), pos)
        cycle = c[:-1] - (c[-1] - c[0]) * (np.arange(count) / count)[:, None, None]  # close the loop

        meta = {"cycle": True, "fps": TARGET_FPS, "source_frames": [int(a), round(float(a + n), 2)],
                "source_fps": float(fps), "smoothing": float(smoothing)}
        motion = _to_pose_keypoint(cycle, w, h, meta)
        preview = np.stack([rig.render(np.nan_to_num(p), w, h) for p in cycle])

        info = (f"{vm.frames_in} video frames at {fps:g} fps ({vm.detected} with a person); cycle = video frames "
                f"{a}-{a + n:.1f}, saved as {count} frames at {TARGET_FPS} fps")
        if vm.auto_loop:
            info += "\nthe clip is one loop (its end leads back into its start), used whole"
        if vm.swaps or vm.dropped:
            info += (f"\nfixed DWPose errors: {vm.swaps} left/right limb swaps, {vm.dropped} misplaced joints "
                     f"re-interpolated")
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
            description="After a run, press 'Save motion as...' on this node and pick where to save the motion "
                        "file (.json). Nothing is written to the server's folders. Open it again with Load Sprite "
                        "Motion's 'Open motion file...'.",
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("motion"),
                io.String.Input("name", default="walk", tooltip="Suggested file name in the save window."),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, motion, name) -> io.NodeOutput:
        text, count = _motion_json(motion)
        filename = (name.strip() or "motion").replace("/", "_").replace("\\", "_")
        if not filename.lower().endswith(".json"):
            filename += ".json"
        # the browser extension (web/sprite_motion.js) keeps this and writes it where the user chooses
        return io.NodeOutput(ui={"sprite_motion": [text], "sprite_motion_name": [filename],
                                 "sprite_motion_frames": [count]})


class LoadSpriteMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSpriteMotion",
            display_name="Load Sprite Motion",
            category="image/sprites",
            description="Loads a motion cycle saved by Save Sprite Motion, for Sprite Pose Loop's video_pose. "
                        f"Press 'Open motion file...' to pick a .json anywhere; it is copied to input/{MOTION_DIR}/ "
                        "and listed here.",
            inputs=[
                io.Combo.Input("motion_file", options=[NONE] + _motion_files(), default=NONE,
                               tooltip=f"Motion files opened before (input/{MOTION_DIR}/; '[output]' = "
                                       f"output/{MOTION_DIR}/)."),
            ],
            outputs=[
                io.Custom("POSE_KEYPOINT").Output(display_name="motion"),
                io.Int.Output(display_name="frames", tooltip="Frames in the cycle."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, motion_file):
        path = _motion_path(motion_file)
        return os.path.getmtime(path) if os.path.isfile(path) else motion_file

    @classmethod
    def validate_inputs(cls, motion_file):
        if motion_file != NONE and not os.path.isfile(_motion_path(motion_file)):
            return f"motion file not found: {motion_file}"
        return True

    @classmethod
    def execute(cls, motion_file) -> io.NodeOutput:
        if motion_file == NONE:
            return io.NodeOutput(None, 0)
        with open(_motion_path(motion_file)) as f:
            data = json.load(f)
        if data.get("sprite_motion") != 1:
            raise ValueError(f"{motion_file} is not a sprite motion file")
        rows = np.array([[np.nan if v is None else v for v in row] for row in data["frames"]], dtype=np.float64)
        meta = {"cycle": True, "file": motion_file}
        motion = _to_pose_keypoint(rows.reshape(len(rows), 18, 2), data["canvas_width"], data["canvas_height"], meta)
        return io.NodeOutput(motion, len(motion))
