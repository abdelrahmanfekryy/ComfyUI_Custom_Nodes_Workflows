"""Animate a character's own DWPose skeleton into a looping pose video (OpenPose-style render).

The motion is either procedural (march, side walk/run, idle) or taken from a driving video: DWPose runs on
every video frame and the motion is retargeted onto the character's own skeleton. Limbs follow the video's
bone directions, but bone lengths, torso and head shape stay the character's. Either way, the pose video
matches the character's proportions and position, and frame 0 is the reference pose. The cycle starts at
the phase closest to the reference pose. When that still differs, a short lead-in eases from the reference
into the motion. Every cycle after the lead-in is complete, and the last frame equals the start of the last
cycle, so the generated clip loops.

`scale` / `offset_x` / `offset_y` move and resize the skeleton. By default the same transform is applied to the
reference image (output `reference`), so the first-frame image guide still matches the skeleton.
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

NOSE, NECK = 0, 1
R_SHO, R_ELB, R_WRI, L_SHO, L_ELB, L_WRI = 2, 3, 4, 5, 6, 7
R_HIP, R_KNE, R_ANK, L_HIP, L_KNE, L_ANK = 8, 9, 10, 11, 12, 13
UPPER = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]
HEAD = [0, 14, 15, 16, 17]
ANKLES = [R_ANK, L_ANK]
# limb bones whose direction comes from the video, parent first (parents are placed before children)
LIMB_BONES = [(R_SHO, R_ELB), (R_ELB, R_WRI), (L_SHO, L_ELB), (L_ELB, L_WRI),
              (R_HIP, R_KNE), (R_KNE, R_ANK), (L_HIP, L_KNE), (L_KNE, L_ANK)]
# left/right limb pairs that DWPose can mix up
LR_GROUPS = [((R_KNE, R_ANK), (L_KNE, L_ANK)), ((R_ELB, R_WRI), (L_ELB, L_WRI))]
# right <-> left, for mirroring a video
LR_SWAP = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]

MATCH_TOLERANCE = 0.02  # max joint offset (fraction of body height) that counts as the reference pose
MIN_CYCLE = 8  # shortest motion cycle searched for in a video, in frames
REPEAT_TOLERANCE = 0.35  # how far the clip may stray from repeating at the cycle length (same unit)
POOR_MATCH = 0.5  # cycle match score (seam difference / motion within the cycle) above which no cycle is found

VIDEO = "from video (video_pose)"
MOTIONS = ["march in place", "walk in place (side view)", "run in place (side view)", "idle (breathing)", VIDEO]
CYCLE_DETECT = "detect cycle"
CYCLE_WHOLE = "whole video is one cycle"
TIMING_NATURAL = "natural speed (sets frames)"
TIMING_FIT = "fit to frames"
MIRROR_AUTO, MIRROR_OFF, MIRROR_ON = "auto (match facing)", "off", "on"

MISSING = object()  # an unconnected optional input, as opposed to a connected one that is not evaluated yet


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


def _angle(v):
    return math.atan2(v[1], v[0])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rotate(v, a):
    """Rotate row vectors v (..., 2) by angle a (image coords: + = clockwise on screen)."""
    c, s = math.cos(a), math.sin(a)
    return v @ np.array([[c, s], [-s, c]])


def _turn(p, p0, a, b):
    """How far bone a->b has turned from the reference pose p0 to p."""
    v, v0 = p[b] - p[a], p0[b] - p0[a]
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(v0) < 1e-6:
        return 0.0
    return _wrap(_angle(v) - _angle(v0))


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
        hips = [j for j in (R_HIP, L_HIP) if valid[j]]
        self.root = p[hips].mean(axis=0) if hips else p[NECK] + [0.0, 0.3 * size]  # mid-hip
        ankles = [j for j in ANKLES if valid[j]]
        self.ground = p[ankles][:, 1].max() if ankles else p[valid][:, 1].max()
        self.spine = float(np.linalg.norm(p[NECK] - self.root)) or 0.3 * size
        self.leg = (sum(self.thigh) + sum(self.shin)) / 2

    def bone_length(self, a, b):
        return {(R_SHO, R_ELB): self.upper_arm[0], (R_ELB, R_WRI): self.forearm[0],
                (L_SHO, L_ELB): self.upper_arm[1], (L_ELB, L_WRI): self.forearm[1],
                (R_HIP, R_KNE): self.thigh[0], (R_KNE, R_ANK): self.shin[0],
                (L_HIP, L_KNE): self.thigh[1], (L_KNE, L_ANK): self.shin[1]}[(a, b)]

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

        # face and hands keep their reference shape: they follow the head / forearm, turned and scaled with it
        body0 = self.body0
        root = p[[R_HIP, L_HIP]].mean(axis=0) if valid[R_HIP] and valid[L_HIP] else None
        root0 = body0[[R_HIP, L_HIP]].mean(axis=0) if root is not None else None
        k = 1.0
        if root is not None and valid[NECK]:
            k = float(np.linalg.norm(p[NECK] - root)) / max(float(np.linalg.norm(body0[NECK] - root0)), 1e-6)
        if self.face0 is not None and valid[NOSE]:
            turn = _turn(p, body0, NECK, NOSE) if valid[NECK] else 0.0
            pts = p[NOSE] + _rotate(self.face0[0] - body0[NOSE], turn) * k
            for x, y in pts[self.face0[1]]:
                cv2.circle(canvas, (int(x), int(y)), max(1, stick // 2), (255, 255, 255), thickness=-1)
        for hand, wrist, elbow in zip(self.hands0, (L_WRI, R_WRI), (L_ELB, R_ELB)):
            if hand is None or not valid[wrist]:
                continue
            turn = _turn(p, body0, elbow, wrist) if valid[elbow] else 0.0
            pts, ok = p[wrist] + _rotate(hand[0] - body0[wrist], turn) * k, hand[1]
            for e, (a, b) in enumerate(HAND_EDGES):
                if ok[a] and ok[b]:
                    hsv = np.uint8([[[int(180 * e / len(HAND_EDGES)), 255, 255]]])
                    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
                    cv2.line(canvas, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), rgb,
                             max(1, stick // 2))
            for x, y in pts[ok]:
                cv2.circle(canvas, (int(x), int(y)), max(1, stick // 2), (0, 0, 255), thickness=-1)
        return canvas


def _largest_person(people, w, h):
    """The most complete, then tallest, body in one frame of DWPose output."""
    best, best_key = None, None
    for person in people:
        pts, ok = _points(person.get("pose_keypoints_2d"), w, h)
        if pts is None or not ok[:18].any():
            continue
        pts, ok = pts[:18], ok[:18]
        key = (int(ok.sum()), float(np.ptp(pts[ok][:, 1])))
        if best_key is None or key > best_key:
            best, best_key = (pts, ok), key
    return best


def _video_track(video_pose):
    """Per-frame body keypoints of the main person in a DWPose result -> (M, 18, 2) with NaN where missing.

    The person is the most complete body in the first frame that has one, then whoever is closest to the
    previous position in each later frame.
    """
    frames = video_pose if isinstance(video_pose, list) else [video_pose]
    out = np.full((len(frames), 18, 2), np.nan)
    prev = None
    for i, data in enumerate(frames):
        w, h = int(data["canvas_width"]), int(data["canvas_height"])
        people = [pp for pp in data.get("people", []) if pp.get("pose_keypoints_2d")]
        if prev is None:
            found = _largest_person(people, w, h)
        else:
            found, best = None, None
            for person in people:
                pts, ok = _points(person["pose_keypoints_2d"], w, h)
                pts, ok = pts[:18], ok[:18]
                if not ok.any():
                    continue
                d = float(np.linalg.norm(pts[ok].mean(axis=0) - prev))
                if best is None or d < best:
                    found, best = (pts, ok), d
        if found is None:
            continue
        pts, ok = found
        out[i][ok] = pts[ok]
        prev = pts[ok].mean(axis=0)
    width = int(frames[0]["canvas_width"])
    return out, width


def _is_saved_motion(video_pose):
    first = video_pose[0] if isinstance(video_pose, list) and video_pose else video_pose
    return isinstance(first, dict) and bool((first.get("sprite_motion") or {}).get("cycle"))


def _body_height(track):
    """Median height of the detected body over the clip, in pixels."""
    heights = [np.ptp(f[np.isfinite(f[:, 1]), 1]) for f in track if np.isfinite(f[:, 1]).sum() > 1]
    return max(float(np.median(heights)), 1.0) if heights else 1.0


def _fix_left_right(track, cyclic):
    """Undo DWPose's left/right mix-ups (common in side views, where the legs cross).

    For legs (knee + ankle) and arms (elbow + wrist) separately, picks per frame whether to swap the left and
    right labels so each limb keeps moving the way it was going: each frame is compared with the constant-
    velocity prediction from the two before it, which tells limbs that pass each other from limbs that bounce
    back. It's a shortest path over the clip (for a looping clip, wrapping from the last frame to the first).
    Of the two labelings that come out (flipping every frame costs the same), keeps the one closer to
    DWPose's own. Returns the track and the number of limb swaps made.
    """
    out = track.copy()
    m = len(out)
    swaps = 0
    if m < 4:
        return out, 0
    cap = 0.1 * _body_height(track)  # one bad joint can't outweigh the rest
    for right, left in LR_GROUPS:
        r, l = list(right), list(left)
        both = np.stack([np.concatenate([track[:, r], track[:, l]], axis=1),
                         np.concatenate([track[:, l], track[:, r]], axis=1)])  # [state, frame, joint, xy]

        def miss(pred, i, s):
            d = np.linalg.norm(both[s, i] - pred, axis=-1)
            d = np.minimum(d[np.isfinite(d)], cap)
            return float(d.sum()) if len(d) else 0.0

        def cost2(i0, i1, i2, a, b, c):  # frame i2 in state c, against the motion of i0 (a) -> i1 (b)
            return miss(2 * both[b, i1] - both[a, i0], i2, c)

        best_total, best_states = np.inf, None
        for s1 in ((0, 1) if cyclic else (None,)):
            # acc[a, b]: best cost with frame i-1 in state a and frame i in state b; frame 0 is state 0
            acc = np.full((2, 2), np.inf)
            for b in (0, 1):
                if s1 is None or b == s1:
                    acc[0, b] = miss(both[0, 0], 1, b)
            backs = []
            for i in range(2, m):
                new_acc = np.full((2, 2), np.inf)
                back = np.zeros((2, 2), dtype=int)
                for b in (0, 1):
                    for c in (0, 1):
                        options = [acc[a, b] + cost2(i - 2, i - 1, i, a, b, c) for a in (0, 1)]
                        back[b, c] = int(np.argmin(options))
                        new_acc[b, c] = min(options)
                acc = new_acc
                backs.append(back)
            if cyclic:  # the motion must also carry on into frames 0 and 1
                for a in (0, 1):
                    for b in (0, 1):
                        acc[a, b] += cost2(m - 2, m - 1, 0, a, b, 0) + cost2(m - 1, 0, 1, b, 0, s1)
            a, b = np.unravel_index(int(np.argmin(acc)), acc.shape)
            if acc[a, b] < best_total:
                states = [0] * m
                states[m - 2], states[m - 1] = int(a), int(b)
                for i in range(m - 1, 1, -1):
                    states[i - 2] = int(backs[i - 2][states[i - 1], states[i]])
                best_total, best_states = acc[a, b], np.array(states)
        states = best_states
        if states.sum() > m / 2:
            states = 1 - states
        flip = states == 1
        out[np.ix_(flip, r)], out[np.ix_(flip, l)] = track[np.ix_(flip, l)], track[np.ix_(flip, r)]
        swaps += int(flip.sum())
    return out, swaps


def _reject_outliers(track, cyclic):
    """Drop joints that jump away from where their neighbouring frames put them (misdetections such as both
    ankles on the same foot); they are interpolated again later. Returns the track and how many were dropped.
    """
    out = track.copy()
    m = len(out)
    if m < 4:
        return out, 0
    size = _body_height(track)
    dropped = 0

    def deviations(j):
        """Distance of each frame's joint j from the line between its nearest valid neighbours."""
        ok = np.flatnonzero(np.isfinite(out[:, j, 0]))
        dev = np.full(m, np.nan)
        if len(ok) < 3:
            return dev
        for k, i in enumerate(ok):
            if cyclic:
                p, n = ok[k - 1], ok[(k + 1) % len(ok)]
                tp, tn = (p if p < i else p - m), (n if n > i else n + m)
            elif 0 < k < len(ok) - 1:
                p, n = ok[k - 1], ok[k + 1]
                tp, tn = p, n
            else:
                continue
            pred = out[p, j] + (out[n, j] - out[p, j]) * (i - tp) / (tn - tp)
            dev[i] = np.linalg.norm(out[i, j] - pred)
        return dev

    for j in range(out.shape[1]):
        dev = deviations(j)
        if not np.isfinite(dev).any():
            continue
        limit = max(0.08 * size, 4.0 * float(np.nanmedian(dev)))
        for _ in range(m // 4):
            if not np.isfinite(dev).any():
                break
            i = int(np.nanargmax(dev))
            if dev[i] <= limit:
                break
            out[i, j] = np.nan
            dropped += 1
            dev = deviations(j)
    return out, dropped


def _fill_and_smooth(track, sigma, cyclic=False):
    """Interpolate missing joints over time, then Gaussian-smooth each joint (sigma in frames). A cyclic track
    (one loop: the frame after the last is the first) wraps around in both."""
    m = track.shape[0]
    t = np.arange(m)
    out = track.copy()
    for j in range(track.shape[1]):
        ok = np.isfinite(track[:, j, 0])
        if not ok.any():
            continue  # never seen: stays NaN, retargeting falls back to the reference pose
        for c in range(2):
            out[:, j, c] = np.interp(t, t[ok], track[ok, j, c], period=m if cyclic else None)
    if sigma > 0 and m > 1:
        r = max(1, int(math.ceil(3 * sigma)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        if cyclic:
            padded = out[np.arange(-r, m + r) % m]
        else:
            padded = np.pad(out, ((r, r), (0, 0), (0, 0)), mode="edge")
        sm = np.zeros_like(out)
        for i, kv in enumerate(k):
            sm += kv * padded[i:i + m]
        out = np.where(np.isfinite(out), sm, out)
    return out


def _upper_envelope_line(t, y):
    """Line through the lower points of the image (largest y): the ground under the feet."""
    sel = np.ones_like(y, dtype=bool)
    a, b = 0.0, float(np.median(y))
    for _ in range(4):
        if sel.sum() < 2:
            break
        a, b = np.polyfit(t[sel], y[sel], 1)
        sel = y >= a * t + b
    return a * t + b


def _sample(x, pos):
    """Linear interpolation of x (frames on axis 0) at fractional frame positions; extrapolates at the ends."""
    m = x.shape[0]
    i0 = np.clip(np.floor(pos).astype(int), 0, max(m - 2, 0))
    i1 = np.minimum(i0 + 1, m - 1)
    f = (pos - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - f) + x[i1] * f


class _VideoMotion:
    """A driving video's motion retargeted onto the character's rig, as one loopable cycle."""

    def __init__(self, rig, video_pose, mirror, smoothing, amplitude, cycle_mode, pin_feet):
        self.rig = rig
        self.pin_feet = pin_feet
        self.no_cycle = None
        # a saved motion (Sprite Motion from Video) is already smoothed and is exactly one cycle
        self.saved = _is_saved_motion(video_pose)
        if self.saved:
            smoothing, cycle_mode = 0.0, CYCLE_WHOLE
        track, width = _video_track(video_pose)
        found = np.isfinite(track[:, :, 0])
        if not found.any():
            raise ValueError("no person found in the driving video's pose keypoints; check its DWPose result")
        for joint, name in ((R_HIP, "hips"), (NECK, "neck")):
            if not found[:, joint].any() and not found[:, LR_SWAP[joint]].any():
                raise ValueError(f"the person in the driving video has no {name} in any frame")
        self.frames_in = track.shape[0]
        self.detected = int(found.any(axis=1).sum())

        # facing: nose ahead of the neck, as a fraction of the torso
        v = _fill_and_smooth(track, 0)
        vroot = np.nanmean(v[:, [R_HIP, L_HIP]], axis=1)
        vtorso = float(np.nanmedian(np.linalg.norm(v[:, NECK] - vroot, axis=1))) or 1.0
        vface = float(np.nanmean(v[:, NOSE, 0] - v[:, NECK, 0])) / vtorso if found[:, NOSE].any() else 0.0
        rface = float(rig.body0[NOSE, 0] - rig.body0[NECK, 0]) / rig.spine if rig.valid[NOSE] else 0.0
        if mirror == MIRROR_AUTO:
            self.mirrored = abs(vface) > 0.15 and abs(rface) > 0.15 and vface * rface < 0
        else:
            self.mirrored = mirror == MIRROR_ON
        if self.mirrored:
            track = track[:, LR_SWAP].copy()
            track[..., 0] = width - track[..., 0]

        # one cycle given as the whole clip: the frame after the last is the first again
        self.cyclic = self.saved or cycle_mode == CYCLE_WHOLE
        self.swaps = self.dropped = 0
        if not self.saved:
            track, self.swaps = _fix_left_right(track, self.cyclic)
            track, self.dropped = _reject_outliers(track, self.cyclic)
        v = _fill_and_smooth(track, smoothing, self.cyclic)
        self.track = v  # the video's own keypoints: labels fixed, gaps filled, smoothed, mirrored if asked
        m = v.shape[0]
        hips = v[:, [R_HIP, L_HIP]]
        vroot = np.nanmean(hips, axis=1)  # mid-hip, or the one hip that was seen

        # size ratio from the legs (they set how high the hips are), else from the torso
        legs = []
        for hip, knee, ankle in ((R_HIP, R_KNE, R_ANK), (L_HIP, L_KNE, L_ANK)):
            legs.append(np.linalg.norm(v[:, knee] - v[:, hip], axis=1) + np.linalg.norm(v[:, ankle] - v[:, knee], axis=1))
        vleg = float(np.nanmedian(np.concatenate(legs))) if np.isfinite(np.concatenate(legs)).any() else math.nan
        vtorso = float(np.nanmedian(np.linalg.norm(v[:, NECK] - vroot, axis=1))) or 1.0
        self.ratio = rig.leg / vleg if math.isfinite(vleg) and vleg > 0 else rig.spine / vtorso

        # root motion: horizontal position, and height of the hips above the ground line
        t = np.arange(m, dtype=np.float64)
        ankles_y = np.nanmax(v[:, ANKLES, 1], axis=1) if np.isfinite(v[:, ANKLES, 1]).any() else np.nanmax(v[..., 1], axis=1)
        ground = _upper_envelope_line(t, ankles_y)
        self.root_x = vroot[:, 0] * self.ratio
        self.height = (ground - vroot[:, 1]) * self.ratio

        # Bone angles over the clip, so `amplitude` can scale each bone's swing around its mean. A bone's length on
        # screen is kept too, relative to its length in the frames that look most like the reference pose: a knee
        # lifted toward the camera shortens the thigh, and without that the lift disappears.
        def angles(a, b):
            d = v[:, b] - v[:, a]
            return np.arctan2(d[:, 1], d[:, 0])

        seen = [(a, b) for a, b in LIMB_BONES if np.isfinite(v[:, [a, b], 0]).all()]
        unlike = np.zeros(m)
        for a, b in seen:
            unlike += np.abs(np.arctan2(np.sin(angles(a, b) - _angle(rig.body0[b] - rig.body0[a])),
                                        np.cos(angles(a, b) - _angle(rig.body0[b] - rig.body0[a]))))
        rest_frames = np.argsort(unlike)[:max(1, min(5, m // 8))]
        spine = np.arctan2(v[:, NECK, 1] - vroot[:, 1], v[:, NECK, 0] - vroot[:, 0])
        head = angles(NECK, NOSE) - spine if found[:, NOSE].any() else np.zeros(m)
        head = np.arctan2(np.sin(head), np.cos(head))
        head_mean = math.atan2(np.sin(head).mean(), np.cos(head).mean())
        def foreshortening(vec):
            length = np.linalg.norm(vec, axis=1)
            f = np.clip(length / max(float(np.median(length[rest_frames])), 1e-6), 0.0, 1.2)
            return np.clip(1.0 - amplitude * (1.0 - f), 0.0, 1.2)

        bone_angles, bone_scale = {}, {}
        for a, b in seen:
            ang = angles(a, b)
            mean = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
            bone_angles[(a, b)] = mean + amplitude * np.arctan2(np.sin(ang - mean), np.cos(ang - mean))
            bone_scale[(a, b)] = foreshortening(v[:, b] - v[:, a])
        spine_mean = math.atan2(np.sin(spine).mean(), np.cos(spine).mean())
        spine = spine_mean + amplitude * np.arctan2(np.sin(spine - spine_mean), np.cos(spine - spine_mean))
        spine_scale = foreshortening(v[:, NECK] - vroot)

        # retarget every frame, root at the origin
        b0, r0 = rig.body0, rig.root
        spine0 = _angle(b0[NECK] - r0)
        self.rel = np.zeros((m, 18, 2))
        for i in range(m):
            p = np.zeros((18, 2))
            turn = _wrap(spine[i] - spine0)
            p[NECK] = np.array([math.cos(spine[i]), math.sin(spine[i])]) * rig.spine * spine_scale[i]
            for j in (R_HIP, L_HIP):
                p[j] = _rotate(b0[j] - r0, turn)
            for j in (R_SHO, L_SHO):
                p[j] = p[NECK] + _rotate(b0[j] - b0[NECK], turn)
            head_turn = turn + amplitude * _wrap(head[i] - head_mean)
            for j in HEAD:
                p[j] = p[NECK] + _rotate(b0[j] - b0[NECK], head_turn)
            for a, b in LIMB_BONES:
                if (a, b) in bone_angles:
                    ang = bone_angles[(a, b)][i]
                    p[b] = p[a] + np.array([math.cos(ang), math.sin(ang)]) * rig.bone_length(a, b) * bone_scale[(a, b)][i]
                else:  # never seen in the video: keep the reference limb, turned with the torso
                    p[b] = p[a] + _rotate(b0[b] - b0[a], turn)
            self.rel[i] = p

        q = self.rel.copy()
        q[..., 1] -= self.height[:, None]  # include how high the body is
        self.desc = q[:, rig.valid] / rig.size  # pose descriptor for matching, in body heights
        self.loops_back = False
        self.start, self.period, self.score = self._find_cycle(cycle_mode)
        self.auto_loop = False
        if self.no_cycle is not None and not self.cyclic:
            # No repeat inside the clip. If, cleaned up as one loop (left/right fixes and gap filling wrapping
            # around), its end leads back into its start, then the clip is one loop: use that.
            loop = _VideoMotion(rig, video_pose, mirror, smoothing, amplitude, CYCLE_WHOLE, pin_feet)
            if loop.loops_back:
                self.__dict__.update(loop.__dict__)
                self.auto_loop = True

        # resample one cycle (its length can be fractional) and spread the remaining mismatch between its end
        # and its start over the cycle, so it loops without a jump. Root x is also centered on the reference.
        a, n = self.start, self.period
        self.samples = max(2, round(n))
        pos = a + np.arange(self.samples + 1) * n / self.samples  # last one = the start of the next cycle
        ramp = np.arange(self.samples) / self.samples

        def looped(x):
            if self.saved:  # frame m would be frame 0 again: nothing to resample or close
                return x.copy()
            c = _sample(self.cycle_source(x), pos)
            return c[:-1] - (c[-1] - c[0]) * ramp.reshape((-1,) + (1,) * (c.ndim - 1))
        self.cyc_rel = looped(self.rel)
        self.cyc_x = looped(self.root_x)
        self.cyc_x -= self.cyc_x.mean()
        self.cyc_h = looped(self.height)

    def cycle_source(self, x):
        """x, with frame 0 appended when the cycle wraps past the last frame of a looping clip."""
        if self.cyclic and self.start + self.period > len(x) - 1:
            return np.concatenate([x, x[:1]])
        return x

    def _whole(self, d):
        """The whole clip as one cycle. When its last frame repeats the first, that frame is the next cycle's."""
        m = d.shape[0]
        steps = np.diagonal(d, 1)
        self.loops_back = m > 2 and d[0, m - 1] <= 2.5 * np.median(steps)
        n = m - 1 if m > 2 and d[0, m - 1] <= np.median(steps) else m
        return 0, float(n), math.nan

    def _find_cycle(self, mode):
        """(start frame, cycle length in frames, match score) of the best-repeating stretch of the video.

        A cycle must start and end on the same pose and motion (the score: that difference relative to how much
        the pose moves within the cycle), and the clip must keep repeating at that period throughout. A clip
        with no such cycle is taken as one cycle.
        """
        q = self.desc
        m = q.shape[0]
        d = np.stack([np.linalg.norm(q - qi, axis=-1).mean(axis=-1) for qi in q])  # (m, m) pose distance
        if self.saved:
            return 0, float(m), math.nan
        if mode == CYCLE_WHOLE or m < 2 * MIN_CYCLE:
            return self._whole(d)
        cs = np.concatenate([np.zeros((m, 1)), np.cumsum(d, axis=1)], axis=1)
        per_period = {}
        for n in range(MIN_CYCLE, m):
            a = np.arange(0, m - n)
            # match over up to 3 frames, so the motion's direction matches too, not only the pose
            num, cnt = np.zeros(len(a)), np.zeros(len(a))
            for j in range(3):
                ok = a + n + j < m
                num[ok] += d[a[ok] + j, a[ok] + n + j]
                cnt[ok] += 1
            spread = (cs[a, a + n] - cs[a, a]) / n  # how much the pose moves during the cycle
            score = np.where(spread > 0.01, num / cnt / np.maximum(spread, 1e-6), np.inf)
            k = int(np.argmin(score))
            repeats = float(np.median(np.diagonal(d, n))) / max(float(spread[k]), 1e-6)
            per_period[n] = (float(score[k]), int(a[k]), float(spread[k]), repeats)
        found = {n: v for n, v in per_period.items() if v[3] <= REPEAT_TOLERANCE}
        if not found:
            self.no_cycle = "the clip does not repeat"
            return self._whole(d)
        n = min(found, key=lambda x: found[x][0])
        best, a, spread, _ = found[n]
        if not best <= POOR_MATCH:
            self.no_cycle = f"best match {best:.2f}, good is < 0.3"
            return self._whole(d)
        # the best match can span several cycles: take the shortest whole fraction of it that repeats all the way
        # through (a march's half cycle, from one leg to the other, matches at its ends but not throughout)
        for div in range(8, 1, -1):
            c = round(n / div)
            if c < MIN_CYCLE:
                continue
            repeat = float(np.mean([d[a + k, a + c + k] for k in range(n - c)])) / spread
            if repeat <= REPEAT_TOLERANCE:
                n, best = c, float(d[a, a + c]) / spread
                break
        # refine the length to a fraction of a frame
        best_len, best_err = float(n), math.inf
        for dn in np.linspace(-1.0, 1.0, 41):
            end = a + n + dn
            if end + 1 > m - 1:
                continue
            err = sum(float(np.linalg.norm(_sample(q, np.array([a + j]))[0] - _sample(q, np.array([end + j]))[0],
                                           axis=-1).mean()) for j in (0, 1))
            if err < best_err:
                best_len, best_err = n + dn, err
        return a, best_len, best

    def pose(self, t):
        """Body keypoints at cycle phase t in [0, 1)."""
        rig = self.rig
        n = self.samples
        s = (t % 1.0) * n
        i0 = int(math.floor(s)) % n
        i1 = (i0 + 1) % n
        f = s - math.floor(s)
        rel = self.cyc_rel[i0] * (1 - f) + self.cyc_rel[i1] * f
        x = self.cyc_x[i0] * (1 - f) + self.cyc_x[i1] * f
        hgt = self.cyc_h[i0] * (1 - f) + self.cyc_h[i1] * f
        p = rel + [rig.root[0] + x, rig.ground - hgt]
        if self.pin_feet:
            ok = [j for j in ANKLES if rig.valid[j]]
            if ok:
                p[:, 1] += rig.ground - p[ok, 1].max()
        return p


def _transform_image(image, s, tx, ty):
    """Scale an image by s and shift it by (tx, ty) pixels; uncovered areas get its border color."""
    out = []
    for frame in image.cpu().numpy():
        h, w = frame.shape[:2]
        border = np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]])
        fill = tuple(float(c) for c in np.median(border, axis=0))
        src, mx = frame, np.array([[s, 0.0, tx], [0.0, s, ty]])
        if s < 1:  # area-downscale first, it aliases less than warpAffine
            nw, nh = max(1, round(w * s)), max(1, round(h * s))
            src = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            mx = np.array([[s * w / nw, 0.0, tx], [0.0, s * h / nh, ty]])
        warped = cv2.warpAffine(src, mx, (w, h), flags=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        if warped.ndim == 2:
            warped = warped[..., None]
        out.append(np.clip(warped, 0.0, 1.0))
    return torch.from_numpy(np.stack(out)).to(image.dtype)


class SpritePoseLoop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SpritePoseLoop",
            display_name="Sprite Pose Loop",
            category="image/sprites",
            description="Animates the skeleton detected on the character reference (DWPose) into a looping pose "
                        "video, drawn OpenPose-style, for pose-controlled video models. The motion is procedural "
                        "or comes from a driving video (DWPose on every frame), retargeted onto the character's "
                        "own proportions. scale/offset move the skeleton, and the reference image with it. Frame 0 "
                        "is the reference pose; motions that never pass through it get a lead-in first. "
                        "loop_start/loop_end mark the last complete cycle, for Sprite Loop Frames.",
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("pose_keypoint", tooltip="DWPose output for the reference image."),
                io.Image.Input("reference", optional=True,
                               tooltip="The reference image DWPose ran on. Gets the same scale/offset as the "
                                       "skeleton (output `reference`, use it as the image guide)."),
                io.Custom("POSE_KEYPOINT").Input("video_pose", optional=True, lazy=True,
                                                 tooltip="The motion: a Load Sprite Motion file (made with the "
                                                         "Sprite Motion from Video workflow), or DWPose on a "
                                                         "video's frames (loaded at 24 fps). Only evaluated when "
                                                         "motion is 'from video'."),
                io.Combo.Input("motion", options=MOTIONS, default=MOTIONS[0],
                               tooltip="Side-view motions need a side-view reference; march and idle suit any. "
                                       "'from video' copies the driving video's motion; the video should show "
                                       "the same view as the reference (front or side)."),
                io.Int.Input("frames", default=49, min=9, max=1025, step=8,
                             tooltip="Clip length, lead-in included. Rounded down to 8n+1 for LTX. With a video "
                                     "and 'natural speed' timing, it's set from the video's cycle instead."),
                io.Int.Input("cycles", default=1, min=1, max=16, tooltip="Motion cycles after the lead-in."),
                io.Float.Input("amplitude", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="Scales the swing of legs and arms (for a video: around each bone's mean "
                                       "angle)."),
                io.Int.Input("lead_in", default=8, min=0, max=256, optional=True,
                             tooltip="Frames that ease from the reference pose into the motion, used only when the "
                                     "motion never passes through the reference pose (side walk/run, most "
                                     "videos, or a moved skeleton). They come before the cycles and are left "
                                     "out of the sprite."),
                io.Float.Input("scale", default=1.0, min=0.1, max=4.0, step=0.01, optional=True,
                               tooltip="Skeleton size, around the character's feet. Below 1 leaves room for "
                                       "wide arm swings or jumps."),
                io.Float.Input("offset_x", default=0.0, min=-1.0, max=1.0, step=0.01, optional=True,
                               tooltip="Moves the skeleton right, as a fraction of the frame width."),
                io.Float.Input("offset_y", default=0.0, min=-1.0, max=1.0, step=0.01, optional=True,
                               tooltip="Moves the skeleton down, as a fraction of the frame height."),
                io.Boolean.Input("transform_reference", default=True, optional=True,
                                 tooltip="Apply scale/offset to the reference image too, so frame 0 still "
                                         "matches. Off: the reference stays put and the lead-in moves the "
                                         "character to the new place."),
                io.Combo.Input("video_cycle", options=[CYCLE_DETECT, CYCLE_WHOLE], default=CYCLE_DETECT,
                               optional=True,
                               tooltip="Video: find one complete motion cycle in the clip, or use the whole clip "
                                       "as one cycle (already trimmed to a loop, or a one-off action)."),
                io.Combo.Input("video_timing", options=[TIMING_NATURAL, TIMING_FIT], default=TIMING_NATURAL,
                               optional=True,
                               tooltip="Video: keep its speed (frames = lead-in + cycles x cycle length, to the "
                                       "nearest 8n+1), or stretch the cycles to fill `frames`."),
                io.Combo.Input("mirror", options=[MIRROR_AUTO, MIRROR_OFF, MIRROR_ON], default=MIRROR_AUTO,
                               optional=True,
                               tooltip="Video: flip it left-right. auto flips when the video faces the other way "
                                       "from a side-view reference."),
                io.Float.Input("smoothing", default=1.0, min=0.0, max=10.0, step=0.1, optional=True,
                               tooltip="Video: removes DWPose jitter (Gaussian, in frames). 0 = off."),
                io.Boolean.Input("pin_feet", default=False, optional=True,
                                 tooltip="Video: keep the lower foot on the ground in every frame. Removes jumps "
                                         "and foot drift."),
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
                io.Image.Output(display_name="reference",
                                tooltip="The reference image with the skeleton's scale/offset applied."),
                io.Image.Output(display_name="preview", tooltip="The pose video drawn over the reference."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, motion, video_pose=MISSING, **kwargs):
        if motion == VIDEO and video_pose is None:
            return ["video_pose"]
        return []

    @classmethod
    def execute(cls, pose_keypoint, motion, frames, cycles, amplitude, lead_in=8, reference=None, video_pose=None,
                scale=1.0, offset_x=0.0, offset_y=0.0, transform_reference=True, video_cycle=CYCLE_DETECT,
                video_timing=TIMING_NATURAL, mirror=MIRROR_AUTO, smoothing=1.0, pin_feet=False) -> io.NodeOutput:
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

        notes = []
        if motion == VIDEO:
            if video_pose is None:
                raise ValueError("motion is 'from video' but there is no motion: pick a file in Load Sprite "
                                 "Motion (make one with the Sprite Motion from Video workflow), or connect "
                                 "DWPose on a video's frames to video_pose")
            vm = _VideoMotion(rig, video_pose, mirror, smoothing, amplitude, video_cycle, pin_feet)
            pose_at = vm.pose
            steps = vm.samples  # phases to try when looking for the reference pose
            if vm.saved:
                cycle = (f"saved motion: one {vm.frames_in}-frame cycle"
                         f"{', mirrored' if vm.mirrored else ''}")
            else:
                cycle = (f"video: {vm.frames_in} frames ({vm.detected} with a person), "
                         f"{'mirrored, ' if vm.mirrored else ''}cycle = video frames {vm.start}-"
                         f"{vm.start + vm.period:.1f} ({vm.period:.1f} frames")
                cycle += f", match {vm.score:.2f})" if math.isfinite(vm.score) else ")"
                if vm.auto_loop:
                    cycle += "; the clip is one loop (its end leads back into its start)"
                if vm.swaps or vm.dropped:
                    cycle += f"; fixed {vm.swaps} left/right limb swaps, {vm.dropped} misplaced joints"
            if vm.no_cycle is not None:
                cycle += (f"; no repeating cycle found ({vm.no_cycle}), so the whole video is used as one cycle: "
                          f"trim it to one cycle, or use a clip with several")
            notes.append(cycle)
        else:
            def pose_at(t):
                return rig.pose(motion, t, amplitude)
            steps = 64

        # skeleton transform: scale around the feet, then shift
        valid_pts = body[valid]
        anchor = np.array([(valid_pts[:, 0].min() + valid_pts[:, 0].max()) / 2, valid_pts[:, 1].max()])
        shift = anchor * (1 - scale) + [offset_x * w, offset_y * h]
        moved = scale != 1.0 or offset_x != 0.0 or offset_y != 0.0

        def place(p):
            return p * scale + shift
        start_ref = place(body) if transform_reference else body  # what frame 0 of the image guide shows

        # start the cycle at the phase that looks most like the reference
        def offset(t):
            return float(np.abs(place(pose_at(t)) - start_ref)[valid].max()) / (rig.size * scale)
        phase0 = min((k / steps for k in range(steps)), key=offset)
        gap = offset(phase0)
        lead = 0 if gap <= MATCH_TOLERANCE else lead_in
        if motion == VIDEO and video_timing == TIMING_NATURAL:
            frames = max(9, min(1025, round((lead + cycles * vm.samples) / 8) * 8 + 1))
        lead = min(lead, frames - 9)
        span = frames - 1 - lead  # frames covered by the cycles; frame `frames - 1` repeats the cycle start

        out = []
        for i in range(frames):
            t = (phase0 + (i - lead) * cycles / span) % 1.0
            p = place(pose_at(t))
            if i < lead:  # ease out of the reference pose into the already-moving cycle
                x = i / lead
                k = x * x * (3 - 2 * x)
                p = start_ref + (p - start_ref) * k
            out.append(rig.render(p, w, h))
        pose_frames = torch.from_numpy(np.stack(out)).float() / 255.0

        if reference is None:
            ref_out = torch.zeros((1, h, w, 3))
            preview = pose_frames
            if moved:
                notes.append("WARNING: scale/offset is set but `reference` is not connected, so the image guide "
                             "does not follow the skeleton")
        else:
            ref_out = reference
            if moved and transform_reference:
                rh, rw = reference.shape[1:3]
                ref_out = _transform_image(reference, scale, shift[0] * rw / w, shift[1] * rh / h)
            bg = ref_out[:1, ..., :3].float()
            if bg.shape[1:3] != (h, w):
                bg = torch.nn.functional.interpolate(bg.movedim(-1, 1), size=(h, w), mode="area").movedim(1, -1)
            drawn = (pose_frames.amax(dim=-1, keepdim=True) > 0).float()
            preview = pose_frames * drawn + bg * 0.55 * (1 - drawn)
        if moved:
            notes.append(f"skeleton scale {scale:g}, offset ({offset_x:+g}, {offset_y:+g}) of the frame"
                         + (", reference moved with it" if transform_reference and reference is not None else ""))

        loop_end = frames - 1
        loop_start = loop_end - round(span / cycles)
        guides = 2 if lead == 0 else 1
        info = (f"{motion}: {frames} frames, lead-in {lead}, {cycles} cycle(s) of {span / cycles:g} frames; "
                f"sprite cycle = frames {loop_start}-{loop_end - 1}; "
                f"start pose offset from reference {gap * 100:.1f}% of body height; image guides: {guides}")
        if notes:
            info += "\n" + "\n".join(notes)
        logging.info("SpritePoseLoop: %s", info)
        return io.NodeOutput(pose_frames, frames, loop_start, loop_end, guides, info, ref_out, preview)
