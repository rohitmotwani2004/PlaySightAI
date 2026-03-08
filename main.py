import cv2
import os
import math
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from utils.math_utils import CourtTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SHOT_QUALITY_COLORS = {
    "GOOD":    (0, 255, 80),
    "AVERAGE": (0, 200, 255),
    "BAD":     (0, 60, 255),
}

SHOT_COLORS = {
    "SMASH":   (0, 60,  255),
    "CLEAR":   (255, 180, 0),
    "DRIVE":   (255, 0,  200),
    "DROP":    (0,  200, 255),
    "NETSHOT": (80, 255, 80),
    "LIFT":    (255, 255, 0),
}

# Coaching tips: (shot, quality) → advice
COACHING_TIPS = {
    ("SMASH",   "GOOD"):    "Excellent! Jump smash — high contact point.",
    ("SMASH",   "AVERAGE"): "Extend arm fully; snap wrist harder at contact.",
    ("SMASH",   "BAD"):     "Too low. Jump higher and meet shuttle above head.",
    ("DRIVE",   "GOOD"):    "Flat fast drive — great racket face control.",
    ("DRIVE",   "AVERAGE"): "Snap wrist at contact for more pace.",
    ("DRIVE",   "BAD"):     "Drive too loopy. Keep racket face vertical, punch through.",
    ("LIFT",    "GOOD"):    "Deep lift — opponent pushed to baseline.",
    ("LIFT",    "AVERAGE"): "Bend knees more; generate upward power from legs.",
    ("LIFT",    "BAD"):     "Lift too short. Get lower and swing fully through.",
    ("NETSHOT", "GOOD"):    "Tight tumbling netshot — perfect touch.",
    ("NETSHOT", "AVERAGE"): "Add slight slice to make shuttle tumble closer to tape.",
    ("NETSHOT", "BAD"):     "Too high. Gentle push with flat racket face.",
    ("CLEAR",   "GOOD"):    "High deep clear — excellent defensive stroke.",
    ("CLEAR",   "AVERAGE"): "Aim for deeper rear corner; more shoulder rotation.",
    ("CLEAR",   "BAD"):     "Clear too short — vulnerable to attack. Swing fully.",
    ("DROP",    "GOOD"):    "Disguised drop — deceptive and accurate.",
    ("DROP",    "AVERAGE"): "Delay wrist action for better deception.",
    ("DROP",    "BAD"):     "Landed mid-court. Slow swing at last moment.",
}

# Pixel → real-world scale for shuttle speed (approx, tune per camera setup)
# Assumes court_length=13.4m mapped to map_h pixels in CourtTransformer
COURT_LENGTH_M = 13.4


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def angle_between(a, b, c) -> float:
    """Angle at point b formed by a-b-c, in degrees. Returns 0 if any point is missing."""
    if any(p is None or (p[0] < 1 and p[1] < 1) for p in [a, b, c]):
        return 0.0
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))


# ─────────────────────────────────────────────────────────────────────────────
# SHOT DISPLAY CACHE
# ─────────────────────────────────────────────────────────────────────────────
class ShotDisplayCache:
    """
    Decouples detection from display.
    Confirmed shots stay on screen for DISPLAY_DURATION seconds with a
    smooth fade-out in the last 0.5 s.
    """
    DISPLAY_DURATION = 2.5

    def __init__(self):
        self._cache: dict = {}

    def update(self, pid: int, shot: str, quality: str, tip: str):
        if shot:
            self._cache[pid] = {'shot': shot, 'quality': quality,
                                'tip': tip, 'ts': time.time()}

    def get(self, pid: int) -> dict:
        entry = self._cache.get(pid)
        if not entry:
            return {}
        age = time.time() - entry['ts']
        if age > self.DISPLAY_DURATION:
            del self._cache[pid]
            return {}
        fade_start = self.DISPLAY_DURATION - 0.5
        alpha = 1.0 if age < fade_start else max(0.0, 1.0 - (age - fade_start) / 0.5)
        return {**entry, 'alpha': alpha}

    def clear(self, pid: int):
        self._cache.pop(pid, None)


# ─────────────────────────────────────────────────────────────────────────────
# SHOT HISTORY (per player timeline)
# ─────────────────────────────────────────────────────────────────────────────
class ShotHistory:
    """Stores last N confirmed shots per player for the timeline strip."""
    MAX = 5

    def __init__(self):
        self._hist: dict = {}   # pid -> deque of (shot, quality, ts)

    def log(self, pid: int, shot: str, quality: str):
        if not shot: return
        self._hist.setdefault(pid, deque(maxlen=self.MAX))
        self._hist[pid].append((shot, quality, time.time()))

    def get(self, pid: int) -> list:
        return list(self._hist.get(pid, []))


# ─────────────────────────────────────────────────────────────────────────────
# 1. HIGHLIGHT MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class HighlightManager:
    def __init__(self, output_dir="highlights", fps=30):
        self.output_dir = output_dir
        self.fps        = fps
        os.makedirs(output_dir, exist_ok=True)
        self.frame_buffer       = deque(maxlen=int(fps * 2))
        self.is_recording       = False
        self.out_video          = None
        self.record_frames_left = 0

    def add_to_buffer(self, frame):
        self.frame_buffer.append(frame.copy())
        if self.is_recording and self.out_video:
            self.out_video.write(frame)
            self.record_frames_left -= 1
            if self.record_frames_left <= 0:
                self.stop_recording()

    def start_highlight(self, shot_type, quality, player_id, frame_idx):
        if self.is_recording or not self.frame_buffer:
            return
        filename = f"{self.output_dir}/{shot_type}_{quality}_ID{player_id}_F{frame_idx}.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        h, w     = self.frame_buffer[0].shape[:2]
        self.out_video = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
        for f in self.frame_buffer:
            self.out_video.write(f)
        self.is_recording       = True
        self.record_frames_left = int(self.fps * 2)
        print(f"🎬 [{quality}] {shot_type} — ID{player_id} @ Frame {frame_idx}")

    def stop_recording(self):
        if self.out_video:
            self.out_video.release()
        self.out_video    = None
        self.is_recording = False


# ─────────────────────────────────────────────────────────────────────────────
# 2. COURT DETECTOR  (95% accuracy model)
# ─────────────────────────────────────────────────────────────────────────────
class CourtDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, frame):
        return self.model(frame, verbose=False, conf=0.25)[0]

    def get_court_box(self, results):
        if results and len(results.boxes) > 0:
            box = results.boxes.xyxy.cpu().numpy()[0]
            return (int(box[0]-60), int(box[1]-60), int(box[2]+60), int(box[3]+60))
        return None

    def get_court_corners(self, results):
        if results and results.keypoints is not None and len(results.keypoints.xy) > 0:
            pts = results.keypoints.xy.cpu().numpy()[0]
            valid = [p for p in pts if p[0] > 10 and p[1] > 10]
            if len(valid) >= 4:
                return np.array(valid, dtype=np.float32)
        return None

    def draw_court(self, frame, results):
        if results is None:
            return frame
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            kpts = results.keypoints.xy.cpu().numpy()
            for i, pt in enumerate(kpts[0]):
                x, y = int(pt[0]), int(pt[1])
                if x > 10:
                    cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                    cv2.putText(frame, f"C{i}", (x, y-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHUTTLE TRACKER  (88.4% accuracy model + EMA + speed decay)
# ─────────────────────────────────────────────────────────────────────────────
class ShuttleTracker:
    LOST_DECAY_FRAMES = 5   # after this many lost frames, speed → 0

    def __init__(self, model_path, history_len=25):
        self.model          = YOLO(model_path)
        self.smooth_history = deque(maxlen=history_len)
        self.velocity       = np.array([0.0, 0.0])
        self.speed          = 0.0
        self._alpha         = 0.4
        self._frames_lost   = 0     # FIX: track consecutive frames without detection

    def predict(self, frame):
        return self.model(frame, verbose=False, conf=0.45)[0]

    def update(self, results):
        """Returns smoothed (cx, cy) or None. Decays speed when shuttle is lost."""
        if results and len(results.boxes) > 0:
            self._frames_lost = 0
            confs = results.boxes.conf.cpu().numpy()
            best  = int(np.argmax(confs))
            box   = results.boxes.xyxy.cpu().numpy()[best]
            cx    = float((box[0] + box[2]) / 2)
            cy    = float((box[1] + box[3]) / 2)

            if self.smooth_history:
                px, py = self.smooth_history[-1]
                cx = self._alpha * cx + (1 - self._alpha) * px
                cy = self._alpha * cy + (1 - self._alpha) * py
            self.smooth_history.append((cx, cy))

            if len(self.smooth_history) >= 3:
                self.velocity = (np.array(self.smooth_history[-1])
                                 - np.array(self.smooth_history[-3]))
                self.speed    = float(np.linalg.norm(self.velocity))
            return (int(cx), int(cy))

        # FIX: decay speed when shuttle not found — prevents stale velocity
        self._frames_lost += 1
        if self._frames_lost >= self.LOST_DECAY_FRAMES:
            self.velocity = np.array([0.0, 0.0])
            self.speed    = 0.0
        return None

    @property
    def direction(self):
        if self.speed < 3:  return "STATIC"
        vy = self.velocity[1]
        if vy < -5: return "UP"
        if vy >  5: return "DOWN"
        return "FLAT"

    def distance_to(self, point):
        """Pixel distance from shuttle to any point."""
        if not self.smooth_history:
            return 9999.0
        return float(np.linalg.norm(
            np.array(self.smooth_history[-1]) - np.array(point, dtype=float)
        ))

    def speed_kmh(self, fps: float, px_per_meter: float) -> float:
        """Convert pixel/frame speed to km/h using known scale."""
        if px_per_meter <= 0 or fps <= 0:
            return 0.0
        return self.speed * fps * 3.6 / px_per_meter

    def draw(self, frame, fps=30, px_per_meter=1.0):
        pts = list(self.smooth_history)
        for i in range(1, len(pts)):
            fade = int(220 * i / len(pts))
            cv2.line(frame,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]),   int(pts[i][1])),
                     (0, fade, 255), 2)
        if pts:
            cx, cy = int(pts[-1][0]), int(pts[-1][1])
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            # Speed badge next to shuttle
            spd = self.speed_kmh(fps, px_per_meter)
            if spd > 1:
                label = f"{spd:.0f} km/h"
                cv2.putText(frame, label, (cx + 10, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHOT ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
class ShotAnalyser:
    """
    Biomechanical shot detection. Improvements over previous version:
      - near_shuttle uses WRIST position, not feet
      - _wrist_motion uses a short 4-6 frame swing window, not full 12-frame history
      - CLEAR vs DROP distinguished by shuttle speed (not just wrist dy overlap)
      - DRIVE uses body-relative thresholds (no hardcoded pixels)
      - Elbow angle used for smash/clear quality
      - Per-player shot cooldown prevents same shot firing repeatedly
      - Per-shot CONFIRM_FRAMES + gap tolerance for netshot
    """
    CONFIRM_FRAMES = {
        "SMASH":   3,
        "CLEAR":   3,
        "DRIVE":   3,
        "DROP":    3,
        "NETSHOT": 2,
        "LIFT":    3,
    }
    MAX_GAP = {
        "SMASH":   0, "CLEAR":   0, "DRIVE":   0,
        "DROP":    0, "NETSHOT": 1, "LIFT":    0,
    }
    SHOT_COOLDOWN = 1.5   # seconds — prevent same shot firing multiple times
    WRIST_HIST_LEN    = 12
    SWING_WINDOW      = 5   # FIX: use only last N frames for swing velocity

    def __init__(self, transformer: CourtTransformer):
        self.transformer  = transformer
        self.wrist_hist:  dict = {}
        self.shot_buffer: dict = {}
        self.gap_counter: dict = {}
        self.last_shot_ts: dict = {}   # pid -> timestamp of last confirmed shot

    def analyse(self, player: dict, shuttle: ShuttleTracker):
        pid  = player['id']
        kpts = player['kpts']

        # ── Wrist history ────────────────────────────────────────────────────
        if pid not in self.wrist_hist:
            self.wrist_hist[pid] = deque(maxlen=self.WRIST_HIST_LEN)
        wrist = self._dominant_wrist(kpts)
        if wrist is not None:
            self.wrist_hist[pid].append(tuple(wrist))

        if len(self.wrist_hist.get(pid, [])) < 5:
            return "", "", ""

        # ── Shot cooldown — don't fire same shot again within SHOT_COOLDOWN s ──
        if time.time() - self.last_shot_ts.get(pid, 0) < self.SHOT_COOLDOWN:
            return "", "", ""

        # ── Court zone ───────────────────────────────────────────────────────
        zone = self._get_zone(player['feet'])

        # ── FIX: Shuttle proximity uses wrist position, not feet ─────────────
        wrist_pos = wrist if wrist is not None else player['feet']
        near_shuttle = shuttle.distance_to(wrist_pos) < 200

        # ── Raw detection ────────────────────────────────────────────────────
        raw_shot, raw_quality = self._detect(pid, kpts, shuttle, zone, near_shuttle)

        # ── Multi-frame confirmation with per-shot gap tolerance ─────────────
        if pid not in self.shot_buffer:  self.shot_buffer[pid] = []
        if pid not in self.gap_counter:  self.gap_counter[pid] = 0

        if raw_shot:
            self.gap_counter[pid] = 0
            self.shot_buffer[pid].append((raw_shot, raw_quality))
        else:
            buffered = [s[0] for s in self.shot_buffer[pid]]
            if buffered:
                current_type = max(set(buffered), key=buffered.count)
                if self.gap_counter[pid] < self.MAX_GAP.get(current_type, 0):
                    self.gap_counter[pid] += 1
                    self.shot_buffer[pid].append((current_type,
                                                  self.shot_buffer[pid][-1][1]))
                else:
                    self.shot_buffer[pid] = []
                    self.gap_counter[pid] = 0
            else:
                self.shot_buffer[pid] = []
                self.gap_counter[pid] = 0

        if self.shot_buffer[pid]:
            shots    = [s[0] for s in self.shot_buffer[pid]]
            dominant = max(set(shots), key=shots.count)
            needed   = self.CONFIRM_FRAMES.get(dominant, 3)
            if len(self.shot_buffer[pid]) >= needed:
                qualities = [s[1] for s in self.shot_buffer[pid] if s[0] == dominant]
                quality   = max(set(qualities), key=qualities.count)
                tip       = COACHING_TIPS.get((dominant, quality), "")
                self.shot_buffer[pid] = []
                self.gap_counter[pid] = 0
                self.last_shot_ts[pid] = time.time()   # set cooldown
                return dominant, quality, tip

        return "", "", ""

    # ── helpers ──────────────────────────────────────────────────────────────
    def _dominant_wrist(self, k):
        if len(k) < 17: return None
        wr, wl = k[10], k[9]
        if wr[0] > 5 and wl[0] > 5:
            return wr if wr[1] < wl[1] else wl
        if wr[0] > 5: return wr
        if wl[0] > 5: return wl
        return None

    def _body_ref(self, k):
        """Nose-to-waist distance as body scale reference."""
        nose_y  = k[0][1]
        waist_y = ((k[11][1] + k[12][1]) / 2
                   if k[11][1] > 0 and k[12][1] > 0 else k[12][1])
        ref = abs(waist_y - nose_y)
        return ref if ref > 40 else 100

    def _is_jumping(self, k, p_ref):
        waist_y = (k[11][1] + k[12][1]) / 2
        ankles  = [a[1] for a in [k[15], k[16]] if a[1] > 0]
        if not ankles: return False
        return min(ankles) < (waist_y + p_ref * 0.35)

    def _swing_motion(self, pid):
        """
        FIX: Use only the last SWING_WINDOW frames (not full 12-frame history).
        This captures the actual racket swing, not slow positional drift.
        Returns (dx, dy) where dy > 0 = wrist moving DOWN in image.
        """
        h = self.wrist_hist[pid]
        if len(h) < 3: return 0.0, 0.0
        # Compare current to SWING_WINDOW frames ago (or oldest available)
        window = min(self.SWING_WINDOW, len(h) - 1)
        idx    = len(h) - 1 - window
        dx     = abs(h[-1][0] - h[idx][0])
        dy     = h[-1][1] - h[idx][1]
        return float(dx), float(dy)

    def _elbow_angle(self, k) -> float:
        """
        Angle at dominant elbow (shoulder → elbow → wrist).
        180° = fully extended (good for smash/clear).
        ~90° = bent (bad for power shots).
        """
        # Right side: shoulder=6, elbow=8, wrist=10
        # Left side:  shoulder=5, elbow=7, wrist=9
        right = angle_between(k[6], k[8], k[10]) if all(k[i][0] > 5 for i in [6,8,10]) else 0
        left  = angle_between(k[5], k[7], k[9])  if all(k[i][0] > 5 for i in [5,7,9])  else 0
        return max(right, left)   # use whichever arm has data

    def _get_zone(self, feet) -> str:
        if self.transformer.matrix is None:
            return "UNKNOWN"
        pos = self.transformer.transform_point(float(feet[0]), float(feet[1]))
        if pos is None:
            return "UNKNOWN"
        net_y  = self.transformer.margin + self.transformer.map_h // 2
        half_h = self.transformer.map_h / 2.0
        dist   = abs(float(pos[1]) - net_y) / half_h if half_h > 0 else 0.5
        if dist < 0.22: return "NET"
        if dist < 0.60: return "MID"
        return "BACK"

    def _detect(self, pid, k, shuttle: ShuttleTracker,
                zone: str, near_shuttle: bool):
        if len(k) < 17: return "", ""

        p_ref   = self._body_ref(k)
        wrist   = self._dominant_wrist(k)
        if wrist is None: return "", ""

        nose_y   = k[0][1]
        avg_sh   = (k[5][1] + k[6][1]) / 2
        avg_wa   = (k[11][1] + k[12][1]) / 2
        jumping  = self._is_jumping(k, p_ref)
        dx, dy   = self._swing_motion(pid)     # FIX: short swing window
        s_dir    = shuttle.direction
        s_spd    = shuttle.speed
        elbow_a  = self._elbow_angle(k)        # FIX: elbow extension

        # ── SMASH ─────────────────────────────────────────────────────────
        # High wrist OR jump + strong downward swing
        if (wrist[1] <= nose_y + p_ref * 0.15 or jumping) and dy > p_ref * 0.16:
            shot = "SMASH"
            # FIX: elbow angle improves quality grading
            if jumping and wrist[1] < nose_y and elbow_a > 155:
                quality = "GOOD"
            elif wrist[1] <= nose_y and elbow_a > 130:
                quality = "AVERAGE"
            else:
                quality = "BAD"
            if near_shuttle and s_dir == "DOWN" and s_spd > 12:
                if quality == "BAD":       quality = "AVERAGE"
                elif quality == "AVERAGE": quality = "GOOD"
            return shot, quality

        # ── CLEAR ─────────────────────────────────────────────────────────
        # FIX: differentiated from DROP by requiring shuttle going UP fast
        # Both share wrist < avg_sh + downward swing, so shuttle direction is key
        if (wrist[1] < avg_sh
                and dy > p_ref * 0.12
                and zone in ("MID", "BACK", "UNKNOWN")
                and (s_dir in ("UP", "STATIC") or not near_shuttle)):
            shot = "CLEAR"
            if wrist[1] < nose_y and elbow_a > 145 and dx > p_ref * 0.08:
                quality = "GOOD"
            elif wrist[1] < avg_sh:
                quality = "AVERAGE"
            else:
                quality = "BAD"
            return shot, quality

        # ── DROP ──────────────────────────────────────────────────────────
        # FIX: differentiated from CLEAR — shuttle goes DOWN slowly
        if (wrist[1] < avg_sh
                and 0 < dy < p_ref * 0.14
                and zone in ("MID", "BACK", "UNKNOWN")
                and (s_dir in ("DOWN", "STATIC") or not near_shuttle)):
            shot = "DROP"
            if near_shuttle and s_dir == "DOWN" and s_spd < 14:
                quality = "GOOD"
            elif near_shuttle:
                quality = "AVERAGE"
            else:
                quality = "BAD"
            return shot, quality

        # ── DRIVE ─────────────────────────────────────────────────────────
        # FIX: thresholds are body-relative (were hardcoded 30px / 50px)
        if (avg_sh - p_ref * 0.30 < wrist[1] < avg_wa + p_ref * 0.50
                and dx > p_ref * 0.22
                and abs(dy) < p_ref * 0.12):
            shot = "DRIVE"
            if near_shuttle and s_dir == "FLAT" and s_spd > 10:
                quality = "GOOD"
            elif dx > p_ref * 0.32:
                quality = "AVERAGE"
            else:
                quality = "BAD"
            return shot, quality

        # ── NETSHOT ───────────────────────────────────────────────────────
        # Small controlled wrist motion at net
        at_net = zone in ("NET", "UNKNOWN")
        if (at_net
                and abs(dx) < p_ref * 0.28
                and abs(dy) < p_ref * 0.22
                and wrist[1] > avg_sh - p_ref * 0.4
                and wrist[1] < avg_wa + p_ref * 0.5
                and not jumping):
            shot = "NETSHOT"
            if near_shuttle and s_spd < 10:
                quality = "GOOD"
            elif near_shuttle:
                quality = "AVERAGE"
            elif s_spd < 6:
                quality = "AVERAGE"
            else:
                quality = "BAD"
            return shot, quality

        # ── LIFT ──────────────────────────────────────────────────────────
        # Wrist below waist, strong upward swing
        if wrist[1] > avg_wa and dy < -(p_ref * 0.14):
            shot = "LIFT"
            if near_shuttle and s_dir == "UP":
                quality = "GOOD"
            elif near_shuttle:
                quality = "AVERAGE"
            else:
                quality = "BAD"
            return shot, quality

        return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLAYER DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class PlayerDetector:
    SKELETON = [
        (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),
        (6,8),(8,10),(5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16)
    ]

    def __init__(self, model_path='yolov8n-pose.pt'):
        self.model = YOLO(model_path)

    def detect_active(self, frame, court_box):
        res = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                               verbose=False, conf=0.25)[0]
        players = []
        if court_box is None or res.boxes is None or res.boxes.id is None:
            return players, res
        cx1, cy1, cx2, cy2 = court_box
        for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
            p_id = int(res.boxes.id[i])
            px1, py1, px2, py2 = map(int, box)
            fx, fy = (px1 + px2) // 2, py2
            if (cx1 < fx < cx2) and (cy1 < fy < cy2):
                kpts = res.keypoints.xy[i].cpu().numpy()
                players.append({'id': p_id, 'box': (px1,py1,px2,py2),
                                 'feet': (fx,fy), 'kpts': kpts,
                                 'shot': '', 'quality': '', 'tip': ''})
        return players, res

    def draw_all(self, frame, players,
                 display_cache: ShotDisplayCache,
                 shot_history: ShotHistory):
        for p in players:
            # skeleton
            for a, b in self.SKELETON:
                pt1 = tuple(map(int, p['kpts'][a]))
                pt2 = tuple(map(int, p['kpts'][b]))
                if pt1[0] > 5 and pt2[0] > 5:
                    cv2.line(frame, pt1, pt2, (0, 200, 0), 2)

            # cached display (2.5 s persistence)
            cached  = display_cache.get(p['id'])
            shot    = cached.get('shot',    '')
            quality = cached.get('quality', '')
            tip     = cached.get('tip',     '')
            alpha   = cached.get('alpha',   1.0)

            shot_color = SHOT_COLORS.get(shot, (200, 200, 200))
            qual_color = SHOT_QUALITY_COLORS.get(quality, (200, 200, 200))
            box_color  = qual_color if quality else (180, 180, 180)

            # bounding box
            cv2.rectangle(frame, (p['box'][0], p['box'][1]),
                          (p['box'][2], p['box'][3]), box_color, 2)

            # ID badge
            self._pill(frame, f"ID {p['id']}",
                       (p['box'][0], p['box'][1] - 28),
                       (70,70,70), (255,255,255), scale=0.50, alpha=1.0)

            if shot:
                box_cx = (p['box'][0] + p['box'][2]) // 2
                # large shot name badge
                self._pill_centred(frame, shot,
                                   (box_cx, p['box'][1] - 58),
                                   shot_color, (0,0,0),
                                   scale=0.92, thickness=2, alpha=alpha)
                # quality badge
                self._pill_centred(frame, quality,
                                   (box_cx, p['box'][1] - 28),
                                   qual_color, (255,255,255),
                                   scale=0.52, thickness=1, alpha=alpha)
                # coaching tip
                if tip:
                    self._tip_banner(frame, tip,
                                     (p['box'][0], p['box'][3] + 6),
                                     p['box'][2] - p['box'][0], alpha)

            # ── shot history timeline below tip ───────────────────────────
            hist = shot_history.get(p['id'])
            if hist:
                self._draw_timeline(frame, hist,
                                    (p['box'][0], p['box'][3] + 36),
                                    p['box'][2] - p['box'][0])
        return frame

    # ── drawing helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _blend_rect(frame, x1, y1, x2, y2, color, alpha):
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(frame.shape[1]-1,x2), min(frame.shape[0]-1,y2)
        if x2<=x1 or y2<=y1: return
        roi = frame[y1:y2, x1:x2]
        ov  = roi.copy(); ov[:] = color
        cv2.addWeighted(ov, alpha, roi, 1-alpha, 0, roi)
        frame[y1:y2, x1:x2] = roi

    def _pill(self, frame, text, origin, bg, fg, scale=0.6, thickness=1,
              pad_x=8, pad_y=4, alpha=1.0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        bx1,by1 = x, y-th-pad_y
        bx2,by2 = x+tw+pad_x*2, y+bl+pad_y
        r = max(1, (by2-by1)//2)
        self._blend_rect(frame, bx1+r, by1, bx2-r, by2, bg, alpha*0.82)
        for cx,cy in [(bx1+r,by1+r),(bx2-r,by1+r),(bx1+r,by2-r),(bx2-r,by2-r)]:
            ov = frame.copy()
            cv2.circle(ov,(cx,cy),r,bg,-1)
            cv2.addWeighted(ov,alpha*0.82,frame,1-alpha*0.82,0,frame)
        cv2.putText(frame, text, (x+pad_x,y), font, scale, fg, thickness, cv2.LINE_AA)

    def _pill_centred(self, frame, text, centre, bg, fg, scale=0.75,
                      thickness=2, pad_x=12, pad_y=5, alpha=1.0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
        cx, cy = centre
        x  = cx - tw//2 - pad_x
        y  = cy + th//2
        bx1,by1 = x,          cy-th//2-pad_y
        bx2,by2 = x+tw+pad_x*2, cy+th//2+bl+pad_y
        r = max(1, (by2-by1)//2)
        self._blend_rect(frame, bx1+r, by1, bx2-r, by2, bg, alpha*0.85)
        for ox,oy in [(bx1+r,by1+r),(bx2-r,by1+r),(bx1+r,by2-r),(bx2-r,by2-r)]:
            ov = frame.copy()
            cv2.circle(ov,(ox,oy),r,bg,-1)
            cv2.addWeighted(ov,alpha*0.85,frame,1-alpha*0.85,0,frame)
        cv2.putText(frame, text, (x+pad_x,y), font, scale, fg, thickness, cv2.LINE_AA)

    def _tip_banner(self, frame, tip, origin, width, alpha=1.0):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        (tw, th), _ = cv2.getTextSize(tip, font, scale, 1)
        x, y = origin
        self._blend_rect(frame, x, y, x+max(width,tw+16), y+th+10,
                         (20,20,20), alpha*0.72)
        cv2.putText(frame, tip, (x+6, y+th+4), font, scale,
                    (255,228,80), 1, cv2.LINE_AA)

    def _draw_timeline(self, frame, hist, origin, width):
        """
        Draw the last N shots as a horizontal mini-badge strip
        e.g.  SMASH · LIFT · DRIVE · NETSHOT · CLEAR
        Oldest left → newest right. Fades older entries.
        """
        if not hist: return
        x, y   = origin
        slot_w = max(1, width // len(hist))
        for i, (shot, quality, _) in enumerate(hist):
            age_alpha = 0.4 + 0.6 * (i + 1) / len(hist)  # older = more faded
            sc = SHOT_COLORS.get(shot, (160,160,160))
            bg = tuple(int(c * 0.4) for c in sc)
            fg = sc
            self._blend_rect(frame, x + i*slot_w, y,
                             x + (i+1)*slot_w - 2, y + 18,
                             bg, age_alpha * 0.75)
            abbrev = shot[:3]
            cv2.putText(frame, abbrev,
                        (x + i*slot_w + 3, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, fg, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 6. STATS OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
class StatsOverlay:
    def __init__(self):
        self.records: dict = {}

    def log(self, pid, shot, quality):
        if not shot or not quality: return
        self.records.setdefault(pid, {})
        self.records[pid].setdefault(shot, {"GOOD":0,"AVERAGE":0,"BAD":0})
        self.records[pid][shot][quality] += 1

    def draw(self, frame):
        if not self.records: return frame
        x, y  = 10, 30
        lines = 1 + sum(1 + len(s) for s in self.records.values())
        PlayerDetector._blend_rect(frame, x-5, y-22, x+300,
                                   y+lines*18+5, (20,20,20), 0.72)
        cv2.putText(frame, "── SHOT STATS ──", (x,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,180), 1)
        y += 20
        for pid, shots in self.records.items():
            cv2.putText(frame, f"Player {pid}", (x,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80,220,255), 1)
            y += 17
            for shot, counts in shots.items():
                total = sum(counts.values())
                gp = int(100*counts["GOOD"]   /total)
                ap = int(100*counts["AVERAGE"]/total)
                bp = int(100*counts["BAD"]    /total)
                # Mini coloured bar
                bar_x = x + 160
                bar_w = 100
                cv2.rectangle(frame, (bar_x,y-10), (bar_x+bar_w,y+2),
                              (50,50,50), -1)
                cv2.rectangle(frame, (bar_x,y-10),
                              (bar_x+int(bar_w*gp/100),y+2), (0,200,60), -1)
                cv2.rectangle(frame, (bar_x+int(bar_w*gp/100),y-10),
                              (bar_x+int(bar_w*(gp+ap)/100),y+2), (0,160,220), -1)
                cv2.rectangle(frame, (bar_x+int(bar_w*(gp+ap)/100),y-10),
                              (bar_x+bar_w,y+2), (0,40,220), -1)

                cv2.putText(frame,
                            f"  {shot:<9} G:{gp}% A:{ap}% B:{bp}% ({total})",
                            (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210,210,170), 1)
                y += 16
        return frame

    def print_summary(self):
        print("\n" + "═"*55)
        print("  MATCH SHOT SUMMARY")
        print("═"*55)
        for pid, shots in self.records.items():
            print(f"\n  Player {pid}")
            print(f"  {'Shot':<10} {'Good':>6} {'Avg':>6} {'Bad':>6} {'Total':>7}")
            print("  " + "─"*40)
            for shot, counts in shots.items():
                total = sum(counts.values())
                print(f"  {shot:<10} {counts['GOOD']:>6} "
                      f"{counts['AVERAGE']:>6} {counts['BAD']:>6} {total:>7}")
        print("═"*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    court_detector  = CourtDetector("models/court_model.pt")
    player_detector = PlayerDetector("yolov8n-pose.pt")
    shuttle_tracker = ShuttleTracker("models/shuttle_best.pt")
    transformer     = CourtTransformer()
    shot_analyser   = ShotAnalyser(transformer)
    display_cache   = ShotDisplayCache()
    shot_history    = ShotHistory()
    stats_overlay   = StatsOverlay()
    highlight_mgr   = HighlightManager()

    cap = cv2.VideoCapture("data/raw/badmintonnew.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    highlight_mgr.fps = fps

    # Pixel-per-meter scale (updated once homography is ready)
    px_per_meter  = 1.0

    heatmap_accum = None
    frame_cnt     = 0
    saved_box     = None
    is_homography = False
    c_res         = None
    last_highlight: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_cnt += 1
        disp = frame.copy()

        # ── Court every 30 frames ──────────────────────────────────────────
        if frame_cnt % 30 == 1 or saved_box is None:
            c_res     = court_detector.predict(frame)
            saved_box = court_detector.get_court_box(c_res)
            corners   = court_detector.get_court_corners(c_res)
            if corners is not None:
                is_homography = transformer.calculate_matrix(corners)
                # Compute pixel/meter scale from minimap
                if is_homography and transformer.map_h > 0:
                    px_per_meter = transformer.map_h / COURT_LENGTH_M

        # ── Players ───────────────────────────────────────────────────────
        players, _ = player_detector.detect_active(frame, saved_box)

        # ── Shuttle ───────────────────────────────────────────────────────
        s_res       = shuttle_tracker.predict(frame)
        shuttle_pos = shuttle_tracker.update(s_res)

        # ── Shot analysis ─────────────────────────────────────────────────
        for p in players:
            shot, quality, tip = shot_analyser.analyse(p, shuttle_tracker)
            p['shot']    = shot
            p['quality'] = quality
            p['tip']     = tip

            display_cache.update(p['id'], shot, quality, tip)

            if shot:
                shot_history.log(p['id'], shot, quality)
                stats_overlay.log(p['id'], shot, quality)
                now      = time.time()
                cooldown = 3.5 if quality == "GOOD" else 5.0
                if now - last_highlight.get(p['id'], 0) > cooldown:
                    highlight_mgr.start_highlight(shot, quality, p['id'], frame_cnt)
                    last_highlight[p['id']] = now

        # ── Draw ──────────────────────────────────────────────────────────
        disp = court_detector.draw_court(disp, c_res)
        disp = player_detector.draw_all(disp, players, display_cache, shot_history)
        disp = shuttle_tracker.draw(disp, fps=fps, px_per_meter=px_per_meter)
        disp = stats_overlay.draw(disp)
        highlight_mgr.add_to_buffer(disp)

        # ── Mini-map + heatmap ─────────────────────────────────────────────
        if is_homography:
            t_data = []
            for p in players:
                pos = transformer.transform_point(
                    float(p['feet'][0]), float(p['feet'][1]))
                if pos is not None:
                    cached = display_cache.get(p['id'])
                    s_lbl  = cached.get('shot', '')
                    if s_lbl == "SMASH": s_lbl = "SMASH!"
                    t_data.append({'id': p['id'], 'pos': pos, 'shot': s_lbl})

            if shuttle_pos is not None:
                s_map = transformer.transform_point(
                    float(shuttle_pos[0]), float(shuttle_pos[1]))
                if s_map is not None:
                    t_data.append({'id': 'shuttle', 'pos': s_map, 'shot': ''})

            minimap = transformer.draw_minimap(t_data)
            cv2.imshow("Minimap", minimap)

            if heatmap_accum is None:
                heatmap_accum = np.zeros(
                    (minimap.shape[0], minimap.shape[1]), dtype=np.float32)
            for d in t_data:
                if d['id'] != 'shuttle':
                    px, py = int(d['pos'][0]), int(d['pos'][1])
                    if 0 <= px < minimap.shape[1] and 0 <= py < minimap.shape[0]:
                        cv2.circle(heatmap_accum, (px, py), 15, 1, -1)

        # ── Display ───────────────────────────────────────────────────────
        cv2.imshow("PlaySight AI", cv2.resize(disp, (1080, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── End of video ──────────────────────────────────────────────────────
   

    if heatmap_accum is not None:
        norm = cv2.normalize(heatmap_accum, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite("match_heatmap.png",
                    cv2.applyColorMap(norm, cv2.COLORMAP_JET))
        print("✅ Heatmap saved → match_heatmap.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()