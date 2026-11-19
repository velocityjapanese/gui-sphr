"""Per-frame acoustic-guitar view: a real CC0 classical guitar, strings in motion.

The plate is a CC0 photograph of a 1912 classical guitar from The Metropolitan
Museum of Art (DP335036), background removed and warped so the measured nut and
saddle land on NUT_X/BRIDGE_X and the six strings fan exactly where the photo's
strings were. The photo supplies the frets, fretboard, headstock, tuning
machines and bridge; the strings, finger stops, plectrum and glows are drawn on
top in local space with real fret geometry (the scale halves every twelve
semitones). The whole layer is rotated so the headstock reads up-left and the
body lower right, with a small badge naming the sounding note and fret.
"""
import math
import queue
import random
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import FPS, H, W
from render.scene import load_font, make_spark_sprite, palette
from render.score_layer import PANEL_H, PANEL_W, PANEL_X, PANEL_Y

# High e down to low E: open pitches and the local y of each string at the nut
# and at the saddle (fanned, derived from the measured photo below).
OPEN_PITCHES = (64, 59, 55, 50, 45, 40)
STRING_NAMES = ("e", "B", "G", "D", "A", "E")

# Wound basses in bronze, nylon trebles in ivory.
STRING_COLORS = ((234, 238, 242), (224, 228, 233), (198, 190, 172),
                 (176, 152, 118), (164, 140, 108), (152, 126, 96))
STRING_THICK = (
    ((-1, 0.45), (0, 1.0), (1, 0.40)),
    ((-1, 0.50), (0, 1.0), (1, 0.45)),
    ((-2, 0.30), (-1, 0.60), (0, 1.0), (1, 0.55), (2, 0.28)),
    ((-2, 0.34), (-1, 0.64), (0, 1.0), (1, 0.58), (2, 0.30)),
    ((-2, 0.38), (-1, 0.68), (0, 1.0), (1, 0.62), (2, 0.34)),
    ((-3, 0.28), (-2, 0.48), (-1, 0.74), (0, 1.0), (1, 0.68), (2, 0.42), (3, 0.26)),
)
VIB_AMP = (1.1, 1.2, 1.5, 1.8, 2.1, 2.4)
VIB_WL = (52.0, 56.0, 62.0, 70.0, 78.0, 88.0)
VIB_OMEGA = (7.0, 6.7, 6.3, 5.8, 5.3, 4.8)

MID = 592.0
LOCAL_W, LOCAL_H = 3400, 1200
NUT_X, BRIDGE_X = 760, 2560
MAX_FRET = 19            # last stop the drawn board still shows
TARGET_FRET = 7          # stops favour the 5th-7th position, like a player
PLUCK_X_MIN, PLUCK_X_MAX = 1980.0, 2210.0

# The photo is upright: the neck runs bottom (headstock) to top (saddle).
BODY_PHOTO = Path(__file__).resolve().parents[1] / "assets" / "guitar_body.jpg"
PHOTO_NUT = (1422.5, 852.0)
PHOTO_SADDLE = (1349.6, 3055.0)
PHOTO_AXIS_U = (-0.03307, 0.99945)     # along the neck, nut -> saddle
PHOTO_AXIS_B = (0.99945, 0.03307)      # across it, toward the low E
PHOTO_NUT_XS = (1355.5, 1382.3, 1409.1, 1435.9, 1462.7, 1489.5)
PHOTO_SADDLE_XS = (1266.5, 1301.9, 1337.3, 1372.7, 1408.1, 1443.5)
PHOTO_STRING_W = (5, 6, 6, 7, 8, 9)

# Whole-layer placement: clockwise tilt, headstock up-left, body lower right.
TILT_DEG = 36.0
TILT_SCALE = 0.50
ANCHOR_LOCAL = (float(NUT_X), MID)
ANCHOR_SCREEN = (388.0, 298.0)

_NUT_V = np.array(PHOTO_NUT, dtype=np.float64)
_U_V = np.array(PHOTO_AXIS_U, dtype=np.float64)
_B_V = np.array(PHOTO_AXIS_B, dtype=np.float64)
_K = (BRIDGE_X - NUT_X) / float(
    np.linalg.norm(np.array(PHOTO_SADDLE, dtype=np.float64) - _NUT_V))


def _across(x_photo, y_photo):
    p = np.array((x_photo, y_photo), dtype=np.float64) - _NUT_V
    return float(MID + _K * float(_B_V @ p))


NUT_Y = tuple(_across(x, PHOTO_NUT[1]) for x in PHOTO_NUT_XS)
SADDLE_Y = tuple(_across(x, PHOTO_SADDLE[1]) for x in PHOTO_SADDLE_XS)


def _fret_ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _string_index(pitch):
    """String whose stop sits nearest the 5th-7th position."""
    best, best_cost = 0, 99
    for si in range(6):
        fret = pitch - OPEN_PITCHES[si]
        if 0 <= fret <= MAX_FRET:
            cost = abs(fret - TARGET_FRET)
            if cost < best_cost:
                best, best_cost = si, cost
    if best_cost < 99:
        return best
    return 0 if pitch > OPEN_PITCHES[0] else 5


def _stop_x(pitch, si):
    """Equal-tempered fret stop: one octave up is exactly half the scale."""
    semis = max(0, pitch - OPEN_PITCHES[si])
    return BRIDGE_X - (BRIDGE_X - NUT_X) * (2.0 ** (-semis / 12.0))


def _string_y(si, x):
    frac = (x - NUT_X) / float(BRIDGE_X - NUT_X)
    return NUT_Y[si] + (SADDLE_Y[si] - NUT_Y[si]) * frac


def _photo_affine():
    s = 2
    return s * np.array([
        [_K * _U_V[0], _K * _U_V[1], NUT_X - _K * float(_U_V @ _NUT_V)],
        [_K * _B_V[0], _K * _B_V[1], MID - _K * float(_B_V @ _NUT_V)],
    ])


def _photo_mask(img):
    """Silhouette of the instrument against the neutral studio backdrop."""
    h, w = img.shape[:2]
    mn = img.min(axis=2).astype(np.int16)
    mx = img.max(axis=2).astype(np.int16)
    backdrop = ((mx - mn) <= 22) & (mx >= 100)
    region = backdrop.astype(np.uint8)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    step = 6
    seeds = [(x, 0) for x in range(0, w, step)]
    seeds += [(x, h - 1) for x in range(0, w, step)]
    seeds += [(0, y) for y in range(0, h, step)]
    seeds += [(w - 1, y) for y in range(0, h, step)]
    for sx, sy in seeds:
        if region[sy, sx] == 1:
            cv2.floodFill(region, ffmask, (sx, sy), 2)
    fg = (region != 2).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.uint8)
    fg = _refine_mask(img, fg)
    # The photo's neck heel juts out above the upper bout; trim it so the
    # cut-out reads as one clean curve from the neck into the body.
    trim = np.array([[1486, 1855], [1528, 1855], [1580, 1948],
                     [1580, 1972], [1486, 1972]], np.int32)
    cv2.fillPoly(fg, [trim], 0)
    fg = (cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 3.0) > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.uint8)
    return fg


def _refine_mask(img, fg):
    """GrabCut strips the cast shadow the colour test leaves on the backdrop,
    then the contour is smoothed so the cut-out edge reads as a clean line."""
    h, w = fg.shape
    S = 0.5
    small = cv2.resize(img, (int(w * S), int(h * S)), interpolation=cv2.INTER_AREA)
    guess = cv2.resize(fg, (int(w * S), int(h * S)), interpolation=cv2.INTER_NEAREST)

    gc = np.where(guess > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    border = 12
    gc[:border, :] = gc[-border:, :] = cv2.GC_BGD
    gc[:, :border] = gc[:, -border:] = cv2.GC_BGD
    gc[cv2.erode(guess, np.ones((31, 31), np.uint8)) > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    fg_s = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    fg_s = cv2.morphologyEx(fg_s, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg_s = (cv2.GaussianBlur(fg_s.astype(np.float32), (0, 0), 2.5) > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg_s, 8)
    if n > 1:
        fg_s = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    fg_s = cv2.morphologyEx(fg_s, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.resize(fg_s, (w, h), interpolation=cv2.INTER_NEAREST)


_BODY_CACHE = {}


def _body_photo():
    """The CC0 photo cut out, strings inpainted, warped into local space."""
    cached = _BODY_CACHE.get("warp")
    if cached is not None:
        return cached
    img = cv2.imread(str(BODY_PHOTO), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"guitar body photo missing: {BODY_PHOTO}")
    img = img[:, :, ::-1]
    fg = _photo_mask(img)

    cut = np.zeros(img.shape[:2], np.uint8)
    for xn, xs, width in zip(PHOTO_NUT_XS, PHOTO_SADDLE_XS, PHOTO_STRING_W):
        cv2.line(cut, (int(round(xn - 2)), 800), (int(round(xs + 1)), 3070),
                 255, width)
    img = cv2.inpaint(img, cv2.dilate(cut, np.ones((3, 3), np.uint8)), 5,
                      cv2.INPAINT_TELEA)

    size = (LOCAL_W * 2, LOCAL_H * 2)
    fwd = _photo_affine()
    rgb = cv2.warpAffine(img, fwd, size, flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(fg * 255, fwd, size, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 2.2)

    rgb = cv2.resize(rgb.astype(np.float32) * 0.78, (LOCAL_W, LOCAL_H),
                     interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (LOCAL_W, LOCAL_H), interpolation=cv2.INTER_AREA)
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])
    _BODY_CACHE["warp"] = out
    return out


def make_guitar_base(palette_name):
    pal = palette(palette_name)
    accent = np.array(pal["accent"], dtype=np.float32)
    canvas = _body_photo().copy()
    canvas[:, :, :3] = np.clip(
        canvas[:, :, :3].astype(np.float32) + accent[None, None, :] * 0.022,
        0, 255).astype(np.uint8)

    img = Image.fromarray(canvas, "RGBA")
    blur = np.array(img.getchannel("A").filter(ImageFilter.GaussianBlur(24)),
                    dtype=np.float32)
    shifted = np.zeros_like(blur)
    dx, dy = 8, 16
    shifted[dy:, dx:] = blur[:LOCAL_H - dy, :LOCAL_W - dx]
    shadow = Image.new("RGBA", (LOCAL_W, LOCAL_H), (0, 0, 0, 0))
    shadow.putalpha(Image.fromarray(np.clip(shifted * 0.55, 0, 255).astype(np.uint8)))
    return np.array(Image.alpha_composite(shadow, img), dtype=np.uint8)


_PICK_CACHE = None


def _make_pick_sprites():
    """Plectrum pre-rendered at each stroke angle."""
    global _PICK_CACHE
    if _PICK_CACHE is not None:
        return _PICK_CACHE
    s = 2
    size = 104 * s
    h, w = 82 * s, 54 * s
    sprites = []
    for ang in range(-20, 21, 2):
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx = size / 2.0
        top = (size - h) / 2.0
        pts = [
            (cx, top + h),
            (cx - w * 0.50, top + h * 0.55),
            (cx - w * 0.47, top + h * 0.22),
            (cx - w * 0.24, top + h * 0.02),
            (cx + w * 0.24, top + h * 0.02),
            (cx + w * 0.47, top + h * 0.22),
            (cx + w * 0.50, top + h * 0.55),
        ]
        d.polygon(pts, fill=(214, 172, 106, 238))
        d.line(pts + [pts[0]], fill=(58, 40, 20, 235), width=3 * s, joint="curve")
        d.ellipse((cx - w * 0.30, top + h * 0.14, cx + w * 0.16, top + h * 0.46),
                  fill=(250, 224, 170, 80))
        img = img.rotate(ang, resample=Image.Resampling.BICUBIC,
                         center=(cx, size / 2.0))
        arr = np.asarray(img.resize((104, 104), Image.Resampling.LANCZOS),
                         dtype=np.float32)
        sprites.append((arr[:, :, :3], arr[:, :, 3] / 255.0))
    _PICK_CACHE = sprites
    return sprites


def _clip_add(roi, add):
    roi[:] = np.minimum(roi.astype(np.uint16) + add.astype(np.uint16), 255).astype(np.uint8)


def _glow_sprite(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    return np.clip(1.0 - r, 0, 1) ** 2.6


class GuitarStripGenerator:
    def __init__(self, section, notes, seed=0):
        pal = palette(section["palette"])
        self.base = make_guitar_base(section["palette"])
        self.key_color = np.array(pal["key_glow"], dtype=np.float32)
        self.accent = np.array(pal["accent"], dtype=np.float32)
        self.finger_glow = _glow_sprite(30)
        self.contact_glow = _glow_sprite(24)
        self.spark = make_spark_sprite(14)
        self.pick_sprites = _make_pick_sprites()
        glow_img = Image.fromarray((_glow_sprite(96) * 255).astype(np.uint8), "L")
        self.body_glow = (np.asarray(glow_img.resize((880, 320), Image.Resampling.BICUBIC),
                                     dtype=np.float32) / 255.0)
        self.M = cv2.getRotationMatrix2D(ANCHOR_LOCAL, -TILT_DEG, TILT_SCALE)
        self.M[0, 2] += ANCHOR_SCREEN[0] - ANCHOR_LOCAL[0]
        self.M[1, 2] += ANCHOR_SCREEN[1] - ANCHOR_LOCAL[1]

        self.badge_sprites = {}
        self.badge_pitch = None
        self.badge_si = 0
        self.badge_alpha = 0.0
        self.badge_since = -10.0
        self.badge_hold = 0.35

        self.section = section
        self.rng = random.Random(seed * 7919 + section["index"] * 104729)

        sec_start, sec_end = section["start"], section["end"]
        self.notes = [n for n in notes if n["start"] <= sec_end and n["end"] >= sec_start - 0.4]
        self.guitar_notes = [n for n in self.notes if n["hand"] == "R"]
        self.active = []
        self.next_idx = 0
        self.particles = []

        self.finger_x = [None] * 6
        self.finger_level = [0.0] * 6
        self.stop_anchor = [float(NUT_X)] * 6
        self.vib = [0.0] * 6

        self.pick_x = 2060.0
        self.pick_y = float(MID)
        self.pick_target_x = 2060.0
        self.pick_target_y = float(MID)
        self.pick_tilt = 0.0
        self.pick_target_tilt = 0.0
        self.pick_dir = 1.0
        self.pick_kick = 0.0
        self.pick_alpha = 0.0

        self.frame_events = {}
        for n in self.guitar_notes:
            f = int(round((n["start"] - sec_start) * FPS))
            if 0 <= f < section["frames"]:
                self.frame_events.setdefault(f, []).append(n)

    def frames(self):
        sec = self.section
        frames = sec["frames"]
        dt = 1.0 / FPS
        for i in range(frames):
            t = sec["start"] + i * dt
            layer = self.base.copy()
            guitar, piano = self._pressed(t)
            self._on_onsets(i, t)
            self._draw_strings(layer, t)
            self._draw_stops(layer, t, guitar)
            self._update_pick(guitar)
            self._draw_pick(layer)
            self._draw_body(layer, piano)
            self._update_particles(layer, dt)
            frame = cv2.warpAffine(layer, self.M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
            self._update_badge(t, guitar)
            self._draw_badge(frame)
            yield frame.tobytes()

    def _pressed(self, t):
        while self.next_idx < len(self.notes) and self.notes[self.next_idx]["start"] <= t:
            self.active.append(self.notes[self.next_idx])
            self.next_idx += 1
        self.active = [n for n in self.active if n["end"] + 0.25 >= t]
        guitar = {}
        piano = 0.0
        for n in self.active:
            if t < n["start"]:
                continue
            rel = t - n["start"]
            attack = min(1.0, rel / 0.012)
            sustain = 0.5 + 0.5 * math.exp(-rel / 0.55)
            value = attack * sustain * (n["vel"] / 127.0)
            if t > n["end"]:
                value *= math.exp(-(t - n["end"]) / 0.13)
            if n["hand"] == "R":
                prev = guitar.get(n["pitch"])
                if prev is None or value > prev[0]:
                    guitar[n["pitch"]] = (value, bool(n.get("arp", False)))
            else:
                piano = max(piano, value)
        return guitar, piano

    def _on_onsets(self, frame_index, t):
        for n in self.frame_events.get(frame_index, ()):
            si = _string_index(n["pitch"])
            self.vib[si] = min(1.35, self.vib[si] + 0.5 + n["vel"] / 240.0)
            self.pick_target_y = _string_y(si, self.pick_x)
            self.pick_kick = 5.0
            self.pick_dir = -self.pick_dir
            self.pick_target_tilt = self.pick_dir * (7.0 + 6.0 * n["vel"] / 127.0)
            target_x = 2010.0 + (n["pitch"] - 40) * 5.5
            self.pick_target_x = max(PLUCK_X_MIN, min(PLUCK_X_MAX, target_x))
            self._spawn(n, si)

    def _draw_strings(self, layer, t):
        xs = np.arange(NUT_X - 28, BRIDGE_X + 2)
        span = float(BRIDGE_X - NUT_X)
        frac = (xs - NUT_X) / span
        for si in range(5, -1, -1):
            y = NUT_Y[si] + (SADDLE_Y[si] - NUT_Y[si]) * frac
            vib = self.vib[si]
            if vib > 0.015:
                xa = max(float(NUT_X), self.stop_anchor[si])
                u = np.clip((xs - xa) / max(1.0, BRIDGE_X - xa), 0.0, 1.0)
                env = np.sin(np.pi * u)
                y = y + (VIB_AMP[si] * vib * env
                         * np.sin(2 * np.pi * (xs - xa) / VIB_WL[si]
                                  - t * VIB_OMEGA[si]))
            yi = np.clip(np.round(y).astype(np.int32), 6, LOCAL_H - 7)
            col = np.array(STRING_COLORS[si], dtype=np.float32) * (1.0 + 0.45 * min(1.0, vib))
            for dy, mul in STRING_THICK[si]:
                rows = np.clip(yi + dy, 0, LOCAL_H - 1)
                layer[rows, xs, :3] = np.clip(col * mul, 0, 255).astype(np.uint8)
                layer[rows, xs, 3] = 255

    def _draw_stops(self, layer, t, guitar):
        string_val = [0.0] * 6
        string_pitch = [None] * 6
        string_arp = [False] * 6
        for pitch, (val, arp) in guitar.items():
            si = _string_index(pitch)
            if val > string_val[si]:
                string_val[si] = val
                string_pitch[si] = pitch
                string_arp[si] = arp

        for si in range(6):
            self.finger_level[si] = max(string_val[si], self.finger_level[si] * 0.90)
            pitch = string_pitch[si]
            if pitch is not None:
                anchor = _stop_x(pitch, si)
                self.stop_anchor[si] = anchor
                if self.finger_x[si] is None:
                    self.finger_x[si] = anchor
                else:
                    self.finger_x[si] += (anchor - self.finger_x[si]) * 0.35

            fx = self.finger_x[si]
            lvl = self.finger_level[si]
            if fx is None or lvl <= 0.015:
                continue
            y = _string_y(si, fx)
            open_str = pitch is not None and pitch <= OPEN_PITCHES[si]
            glow = (0.5 if open_str else 1.0) * (0.72 if string_arp[si] else 1.0)

            self._add_column(layer, int(round(fx)), int(round(y - 30)), int(round(y - 3)),
                             self.key_color, (0.06 + 0.28 * lvl) * glow)
            self._stamp(layer, fx, y, self.finger_glow, self.key_color, 0.42 * lvl * glow)
            self._stamp(layer, fx, y, self.spark, self.accent, 0.55 * lvl * glow)

            x0 = max(0, int(fx) - 4)
            x1 = min(LOCAL_W, int(fx) + 4)
            band = np.broadcast_to((self.accent * lvl * 0.55 * glow).astype(np.uint16),
                                   (3, x1 - x0, 3))
            _clip_add(layer[int(y) - 1:int(y) + 2, x0:x1, :3], band)

            if not open_str and lvl > 0.02:
                self._draw_wave(layer, si, fx, t, lvl)
            self.vib[si] = max(string_val[si], self.vib[si] * 0.94)

    def _draw_wave(self, layer, si, xa, t, energy):
        xb = float(BRIDGE_X)
        x0 = int(xa) + 6
        if xb - x0 < 12:
            return
        xs = np.arange(x0, int(xb))
        u = (xs - xa) / max(1.0, xb - xa)
        env = np.sin(np.pi * u)
        y = _string_y(si, xs) + (VIB_AMP[si] + 0.6) * energy * env \
            * np.sin(2 * np.pi * (xs - xa) / VIB_WL[si] - t * VIB_OMEGA[si])
        yi = np.clip(np.round(y).astype(np.int32), 0, LOCAL_H - 1)
        col = np.clip(self.accent * (0.35 + 0.5 * energy), 0, 255).astype(np.uint16)
        cur = layer[yi, xs, :3].astype(np.uint16)
        layer[yi, xs, :3] = np.minimum(cur + col[None, :], 255).astype(np.uint8)

    def _update_pick(self, guitar):
        energy = min(1.0, max((v for v, _ in guitar.values()), default=0.0))
        self.pick_x += (self.pick_target_x - self.pick_x) * 0.10
        self.pick_y += (self.pick_target_y - self.pick_y) * 0.30
        self.pick_tilt += (self.pick_target_tilt - self.pick_tilt) * 0.25
        self.pick_kick *= 0.78
        target = 1.0 if guitar else 0.0
        self.pick_alpha += (target - self.pick_alpha) * 0.14
        self._pick_energy = energy

    def _draw_pick(self, layer):
        if self.pick_alpha < 0.05:
            return
        idx = max(0, min(20, int(round(self.pick_tilt / 2.0)) + 10))
        rgb, alpha = self.pick_sprites[idx]
        sh, sw = alpha.shape
        self._stamp(layer, self.pick_x, self.pick_y + 8 + self.pick_kick,
                    self.contact_glow, self.key_color,
                    0.16 * self.pick_alpha * (0.4 + 0.6 * self._pick_energy))
        ox = int(round(self.pick_x - sw * 0.5))
        oy = int(round(self.pick_y - sh * 0.48 + 6 + self.pick_kick))
        cx0, cy0 = max(0, ox), max(0, oy)
        cx1, cy1 = min(LOCAL_W, ox + sw), min(LOCAL_H, oy + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub_a = (alpha[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
                 * (0.32 + 0.68 * self.pick_alpha))[:, :, None]
        sub_rgb = rgb[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
        roi = layer[cy0:cy1, cx0:cx1]
        roi[:, :, :3] = np.clip(roi[:, :, :3] * (1.0 - sub_a) + sub_rgb * sub_a,
                                0, 255).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (sub_a[:, :, 0] * 255).astype(np.uint8))

    def _draw_body(self, strip, piano):
        if piano <= 0.01:
            return
        sh, sw = self.body_glow.shape
        cx, cy = 2260, int(MID + 190)
        x0, y0 = cx - sw // 2, cy - sh // 2
        cy0, cy1 = max(0, y0), min(LOCAL_H, y0 + sh)
        cx0, cx1 = max(0, x0), min(LOCAL_W, x0 + sw)
        if cy1 <= cy0 or cx1 <= cx0:
            return
        sub = self.body_glow[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * self.accent[None, None, :] * (0.16 * piano)).astype(np.uint16)
        _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)

    def _add_column(self, strip, cx, y0, y1, color, intensity):
        h = y1 - y0
        if h <= 2:
            return
        w = 20
        x0 = max(0, cx - w // 2)
        x1 = min(LOCAL_W, x0 + w)
        if x1 <= x0:
            return
        xx = np.arange(x1 - x0, dtype=np.float32) + (x0 - (cx - w // 2))
        horiz = np.clip(1.0 - np.abs(xx - w / 2) / (w / 2), 0, 1)[None, :, None] ** 1.4
        vert = np.linspace(0.35, 1.0, h, dtype=np.float32)[:, None, None]
        add = color[None, None, :] * (horiz * vert) * intensity
        _clip_add(strip[y0:y1, x0:x1, :3], np.clip(add, 0, 255).astype(np.uint16))

    def _stamp(self, strip, cx, cy, sprite, color, intensity):
        sh = sprite.shape[0]
        x0, y0 = int(round(cx - sh / 2)), int(round(cy - sh / 2))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(LOCAL_W, x0 + sh), min(LOCAL_H, y0 + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * color[None, None, :] * (255.0 * min(1.0, intensity))).astype(np.uint16)
        _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)

    def _blend(self, frame, x, y, rgb, alpha, mul=1.0):
        sh, sw = alpha.shape
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(W, x + sw), min(H, y + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        a = (alpha[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * mul)[:, :, None]
        roi = frame[cy0:cy1, cx0:cx1]
        roi[:, :, :3] = np.clip(roi[:, :, :3] * (1.0 - a)
                                + rgb[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * a,
                                0, 255).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (a[:, :, 0] * 255).astype(np.uint8))

    def _badge_sprite(self, pitch, si):
        key = (pitch, si)
        sprite = self.badge_sprites.get(key)
        if sprite is not None:
            return sprite
        note = f"{'CCDDEFFGGAAB'[pitch % 12]}{'#' if pitch % 12 in (1, 3, 6, 8, 10) else ''}{pitch // 12 - 1}"
        fret = pitch - OPEN_PITCHES[si]
        if fret <= 0:
            detail = "open string"
        else:
            detail = f"{_fret_ordinal(fret)} fret · {STRING_NAMES[si]} string"
        f_note = load_font("sans", 14)
        f_detail = load_font("sans", 11)
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        nw = int(probe.textlength(note, font=f_note))
        dw = int(probe.textlength(detail, font=f_detail))
        pad_x, gap, h = 10, 8, 29
        w = pad_x * 2 + nw + gap + dw
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=(7, 10, 17, 214),
                            outline=tuple(int(c) for c in self.accent) + (105,), width=1)
        d.text((pad_x, 6), note, font=f_note, fill=(236, 243, 255, 240))
        d.text((pad_x + nw + gap, 9), detail, font=f_detail,
               fill=(204, 214, 232, 225))
        arr = np.asarray(img, dtype=np.float32)
        sprite = (arr[:, :, :3], arr[:, :, 3] / 255.0, w, h)
        self.badge_sprites[key] = sprite
        return sprite

    def _update_badge(self, t, guitar):
        if guitar:
            pitch = max(guitar, key=lambda p: guitar[p][0])
            if pitch != self.badge_pitch and t - self.badge_since >= self.badge_hold:
                self.badge_pitch = pitch
                self.badge_si = _string_index(pitch)
                self.badge_since = t
            target = 1.0
        else:
            target = 0.0
        self.badge_alpha += (target - self.badge_alpha) * 0.22

    def _draw_badge(self, frame):
        if self.badge_pitch is None or self.badge_alpha < 0.03:
            return
        rgb, alpha, w, h = self._badge_sprite(self.badge_pitch, self.badge_si)
        fx = self.finger_x[self.badge_si]
        if fx is None:
            fx = _stop_x(self.badge_pitch, self.badge_si)
        sx, sy = self._project(fx, _string_y(self.badge_si, fx))
        x = int(round(sx + 18))
        y = int(round(sy + 16))
        if (x + w > PANEL_X - 6 and x < PANEL_X + PANEL_W + 6
                and y + h > PANEL_Y - 6 and y < PANEL_Y + PANEL_H + 6):
            x = int(round(sx - w - 18))
        x = max(16, min(W - w - 16, x))
        y = max(16, min(H - h - 16, y))
        self._blend(frame, x, y, rgb, alpha, min(1.0, self.badge_alpha))

    def _project(self, x, y):
        return (self.M[0, 0] * x + self.M[0, 1] * y + self.M[0, 2],
                self.M[1, 0] * x + self.M[1, 1] * y + self.M[1, 2])

    def _spawn(self, note, si):
        count = 2 + int(note["vel"] / 60)
        y = _string_y(si, self.pick_x)
        for _ in range(count):
            self.particles.append({
                "x": self.pick_x + self.rng.uniform(-10, 10),
                "y": y + self.rng.uniform(-4, 8),
                "vx": self.rng.uniform(-26, 26),
                "vy": -self.rng.uniform(40, 110),
                "life": self.rng.uniform(0.4, 0.85),
                "age": 0.0,
            })

    def _update_particles(self, strip, dt):
        alive = []
        sprite = self.spark
        sh = sprite.shape[0]
        for p in self.particles:
            p["age"] += dt
            if p["age"] >= p["life"]:
                continue
            p["x"] += p["vx"] * dt
            p["vy"] += 30 * dt
            p["y"] += p["vy"] * dt
            life = 1.0 - p["age"] / p["life"]
            x, y = int(p["x"]), int(p["y"])
            x0, y0 = x - sh // 2, y - sh // 2
            x1, y1 = x0 + sh, y0 + sh
            if x1 <= 0 or y1 <= 0 or x0 >= LOCAL_W or y0 >= LOCAL_H:
                continue
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(LOCAL_W, x1), min(LOCAL_H, y1)
            sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            add = (sub[:, :, None] * self.accent[None, None, :]
                   * (life ** 1.4) * 190).astype(np.uint16)
            _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)
            alive.append(p)
        self.particles = alive


class PipeWriter:
    """Writes frames to a subprocess stdin on a background thread."""

    def __init__(self, stream, maxsize=4):
        self.stream = stream
        self.q = queue.Queue(maxsize=maxsize)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, data):
        while not self.error:
            try:
                self.q.put(data, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self):
        while True:
            try:
                self.q.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self.thread.join()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            if self.error is not None:
                continue
            try:
                self.stream.write(item)
            except (BrokenPipeError, ValueError, OSError) as exc:
                self.error = exc


def stream_section(section, notes, stream, seed=0):
    gen = GuitarStripGenerator(section, notes, seed)
    writer = PipeWriter(stream)
    try:
        for frame in gen.frames():
            writer.write(frame)
    finally:
        writer.close()
        try:
            stream.close()
        except Exception:
            pass
