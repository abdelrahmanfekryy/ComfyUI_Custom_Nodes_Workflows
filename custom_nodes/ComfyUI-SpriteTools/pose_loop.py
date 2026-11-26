"""Animate a character's own DWPose skeleton into a looping pose video (OpenPose-style render).

Because the skeleton comes from the character's reference image, the pose video matches its proportions
and position exactly, and frame 0 is the reference pose. The cycle starts at the phase closest to the
reference pose; when that still differs (side walk/run), a short lead-in eases from the reference into the
motion. Every cycle after the lead-in is complete, and the last frame equals the start of the last cycle,
so the generated clip loops.
"""
import logging
import math

import cv2
import numpy as np
import torch

from comfy_api.latest import io

# OpenPose 18-point body: 0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist, 5 l_shoulder, 6 l_elbow,
# 7 l_wrist, 8 r_hip, 9 r_knee, 10 r_ankle, 11 l_hip, 12 l_knee, 13 l_ankle, 14 r_eye, 15 l_eye, 16 r_ear,
# 17 l_ear. Limbs and colors follow the standard OpenPose/DWPose renderer.
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

R_SHO, R_ELB, R_WRI, L_SHO, L_ELB, L_WRI = 2, 3, 4, 5, 6, 7
R_HIP, R_KNE, R_ANK, L_HIP, L_KNE, L_ANK = 8, 9, 10, 11, 12, 13
UPPER = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]

MATCH_TOLERANCE = 0.02  # max joint offset (fraction of body height) that counts as the reference pose

MOTIONS = ["march in place", "walk in place (side view)", "run in place (side view)", "idle (breathing)"]


def _points(flat, w, h):
    """Flat [x, y, c, ...] list -> (N, 2) pixel coords and a validity mask. Handles normalized coords."""
    if not flat:
        return None, None
    a = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
    valid = a[:, 2] > 0
    xy = a[:, :2].copy()
    if valid.any() and xy[valid].max() <= 1.0:
        xy *= (w, h)
    return xy, valid


def _length(p, a, b, fallback):
    return float(np.linalg.norm(p[a] - p[b])) or fallback


def _down(angle, length):
    """Vector of `length` pointing straight down, rotated by `angle` (radians, + = toward +x)."""
    return np.array([math.sin(angle), math.cos(angle)]) * length


class _Rig:
    def __init__(self, body, valid, face, hands):
        self.body0, self.valid = body, valid
        self.face0, self.hands0 = face, hands
        p = body
        size = max(p[valid][:, 1].max() - p[valid][:, 1].min(), 1.0)
        self.size = size
        self.thigh = _length(p, R_HIP, R_KNE, 0.25 * size), _length(p, L_HIP, L_KNE, 0.25 * size)
        self.shin = _length(p, R_KNE, R_ANK, 0.25 * size), _length(p, L_KNE, L_ANK, 0.25 * size)
        self.upper_arm = _length(p, R_SHO, R_ELB, 0.17 * size), _length(p, L_SHO, L_ELB, 0.17 * size)
        self.forearm = _length(p, R_ELB, R_WRI, 0.15 * size), _length(p, L_ELB, L_WRI, 0.15 * size)
        # facing: nose ahead of the neck (side views); +1 = right
        self.facing = 1.0 if p[0, 0] >= p[1, 0] else -1.0

    def pose(self, motion, t, amp):
        """Body keypoints at cycle phase t in [0, 1)."""
        p = self.body0.copy()
        s = math.sin(2 * math.pi * t)
        c = math.cos(2 * math.pi * t)
        legs = ((R_HIP, R_KNE, R_ANK, 0), (L_HIP, L_KNE, L_ANK, 1))
        arms = ((R_SHO, R_ELB, R_WRI, 0), (L_SHO, L_ELB, L_WRI, 1))

        if motion == "march in place":
            # front view: a lifted thigh points at the camera, so it shortens on screen
            for hip, knee, ankle, i in legs:
                lift = max(0.0, s if i == 0 else -s) * math.radians(65) * amp
                k = p[hip] + (p[knee] - p[hip]) * [1, math.cos(lift)]
                p[ankle] = k + (p[ankle] - p[knee])
                p[knee] = k
            for sho, elb, wri, i in arms:  # arm swings with the opposite leg
                swing = max(0.0, -s if i == 0 else s) * math.radians(50) * amp
                e = p[sho] + (p[elb] - p[sho]) * [1, math.cos(swing)]
                p[wri] = e + (p[wri] - p[elb]) * [1, math.cos(swing * 1.6)]
                p[elb] = e
            p[UPPER] += [0.0, -0.012 * self.size * abs(s) * amp]

        elif motion.startswith(("walk", "run")):
            run = motion.startswith("run")
            hip_amp = math.radians(38 if run else 26) * amp
            knee_amp = math.radians(105 if run else 60) * amp
            arm_amp = math.radians(45 if run else 28) * amp
            f = self.facing
            ground = max(p[R_ANK, 1], p[L_ANK, 1])
            for hip, knee, ankle, i in legs:
                ph = 0.0 if i == 0 else math.pi
                th = hip_amp * math.sin(2 * math.pi * t + ph)
                flex = knee_amp * max(0.0, math.cos(2 * math.pi * t + ph)) ** 1.5  # bends while swinging forward
                p[knee] = p[hip] + _down(f * th, self.thigh[i])
                p[ankle] = p[knee] + _down(f * (th - flex), self.shin[i])
            for sho, elb, wri, i in arms:
                ph = math.pi if i == 0 else 0.0  # opposite to the same-side leg
                th = arm_amp * math.sin(2 * math.pi * t + ph)
                bend = math.radians(70 if run else 20) + 0.5 * max(0.0, th)
                p[elb] = p[sho] + _down(f * th, self.upper_arm[i])
                p[wri] = p[elb] + _down(f * (th + bend), self.forearm[i])
            # keep the lower foot on the ground: the body bobs up as the legs pass each other
            drop = ground - max(p[R_ANK, 1], p[L_ANK, 1])
            p[:, 1] += drop
            if run:
                p[UPPER] += [f * 0.04 * self.size * amp, 0.0]  # lean into the run

        else:  # idle
            breath = 0.5 - 0.5 * c
            p[UPPER] += [0.0, 0.010 * self.size * breath * amp]
            p[[0, 14, 15, 16, 17]] += [0.0, 0.004 * self.size * breath * amp]
        return p

    def render(self, p, w, h):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        scale = max(min(w, h) / 512.0, 0.5)
        stick = max(1, round(4 * scale))
        valid = self.valid
        for (a, b), color in zip(LIMBS, COLORS):
            if not (valid[a] and valid[b]):
                continue
            (x1, y1), (x2, y2) = p[a], p[b]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            length = math.hypot(x1 - x2, y1 - y2)
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stick), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, [int(v * 0.6) for v in color])
        for i, color in enumerate(COLORS):
            if valid[i]:
                cv2.circle(canvas, (int(p[i, 0]), int(p[i, 1])), stick, color, thickness=-1)

        # face and hands follow the head / wrists rigidly
        if self.face0 is not None:
            d = p[0] - self.body0[0]
            for x, y in self.face0[0][self.face0[1]] + d:
                cv2.circle(canvas, (int(x), int(y)), max(1, stick // 2), (255, 255, 255), thickness=-1)
        for hand, wrist in zip(self.hands0, (L_WRI, R_WRI)):
            if hand is None:
                continue
            pts, ok = hand[0] + (p[wrist] - self.body0[wrist]), hand[1]
            for k, (a, b) in enumerate(HAND_EDGES):
                if ok[a] and ok[b]:
                    hsv = np.uint8([[[int(180 * k / len(HAND_EDGES)), 255, 255]]])
                    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
                    cv2.line(canvas, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), rgb,
                             max(1, stick // 2))
            for x, y in pts[ok]:
                cv2.circle(canvas, (int(x), int(y)), max(1, stick // 2), (0, 0, 255), thickness=-1)
        return canvas


class SpritePoseLoop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SpritePoseLoop",
            display_name="Sprite Pose Loop",
            category="image/sprites",
            description="Animates the skeleton detected on the character reference (DWPose) into a looping pose "
                        "video, drawn OpenPose-style, for pose-controlled video models. The skeleton keeps the "
                        "character's own proportions and position. Frame 0 is the reference pose; motions that "
                        "never pass through it (side walk/run) get a lead-in first. loop_start/loop_end mark the "
                        "last complete cycle, for Sprite Loop Frames.",
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("pose_keypoint", tooltip="DWPose output for the reference image."),
                io.Combo.Input("motion", options=MOTIONS, default=MOTIONS[0],
                               tooltip="Side-view motions need a side-view reference; march and idle suit any."),
                io.Int.Input("frames", default=49, min=9, max=1025, step=8,
                             tooltip="Clip length, lead-in included. Rounded down to 8n+1 for LTX."),
                io.Int.Input("cycles", default=1, min=1, max=16, tooltip="Motion cycles after the lead-in."),
                io.Float.Input("amplitude", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="Scales the swing of legs and arms."),
                io.Int.Input("lead_in", default=8, min=0, max=256, optional=True,
                             tooltip="Frames that ease from the reference pose into the motion, used only when the "
                                     "motion never passes through the reference pose (side walk/run). They come "
                                     "before the cycles and are left out of the sprite."),
            ],
            outputs=[
                io.Image.Output(display_name="pose_frames"),
                io.Int.Output(display_name="frames"),
                io.Int.Output(display_name="loop_start", tooltip="First frame of the last complete cycle."),
                io.Int.Output(display_name="loop_end", tooltip="Frame where that cycle has come full circle (its "
                                                               "pose equals loop_start's; not part of the sprite)."),
                io.Int.Output(display_name="image_guides",
                              tooltip="Reference image guides to use: 2 (first and last frame) when the clip ends "
                                      "on the reference pose, else 1 (first frame only)."),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, pose_keypoint, motion, frames, cycles, amplitude, lead_in=8) -> io.NodeOutput:
        frames = (frames - 1) // 8 * 8 + 1
        data = pose_keypoint[0] if isinstance(pose_keypoint, list) else pose_keypoint
        w, h = int(data["canvas_width"]), int(data["canvas_height"])
        people = [pp for pp in data.get("people", []) if pp.get("pose_keypoints_2d")]
        if not people:
            raise ValueError("no person found in the pose keypoints; check the DWPose result on the reference")
        person = max(people, key=lambda pp: sum(1 for v in pp["pose_keypoints_2d"][2::3] if v > 0))

        body, valid = _points(person["pose_keypoints_2d"], w, h)
        body = body[:18]
        valid = valid[:18]
        face = _points(person.get("face_keypoints_2d"), w, h)
        hands = [_points(person.get(k), w, h) for k in ("hand_left_keypoints_2d", "hand_right_keypoints_2d")]
        rig = _Rig(body, valid, face if face[0] is not None else None,
                   [hd if hd[0] is not None else None for hd in hands])

        # start the cycle at the phase that looks most like the reference
        def offset(t):
            return float(np.abs(rig.pose(motion, t, amplitude) - body)[valid].max()) / rig.size
        phase0 = min((k / 64 for k in range(64)), key=offset)
        gap = offset(phase0)
        lead = 0 if gap <= MATCH_TOLERANCE else min(lead_in, frames - 9)
        span = frames - 1 - lead  # frames covered by the cycles; frame `frames - 1` repeats the cycle start

        out = []
        for i in range(frames):
            t = (phase0 + (i - lead) * cycles / span) % 1.0
            p = rig.pose(motion, t, amplitude)
            if i < lead:  # ease out of the reference pose into the already-moving cycle
                x = i / lead
                k = x * x * (3 - 2 * x)
                p = body + (p - body) * k
            out.append(rig.render(p, w, h))

        loop_end = frames - 1
        loop_start = loop_end - round(span / cycles)
        guides = 2 if lead == 0 else 1
        info = (f"{motion}: {frames} frames, lead-in {lead}, {cycles} cycle(s) of {span / cycles:g} frames; "
                f"sprite cycle = frames {loop_start}-{loop_end - 1}; "
                f"start pose offset from reference {gap * 100:.1f}% of body height; image guides: {guides}")
        logging.info("SpritePoseLoop: %s", info)
        return io.NodeOutput(torch.from_numpy(np.stack(out)).float() / 255.0, frames, loop_start, loop_end, guides,
                             info)
