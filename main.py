import cv2
import os
import csv
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
COURT_LENGTH_M = 13.4
QUALITY_SCORE  = {"GOOD": 1.0, "AVERAGE": 0.5, "BAD": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def angle_between(a, b, c) -> float:
    """Angle at joint b (a→b→c) in degrees [0-180]. 0 if any point missing."""
    if any(p is None or (p[0] < 1 and p[1] < 1) for p in [a, b, c]):
        return 0.0
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))

def norm_angle(deg):
    """Normalise joint angle [0-180] → [0.0-1.0]."""
    return round(float(np.clip(deg / 180.0, 0.0, 1.0)), 4)

def norm_pos(x, y, w, h):
    return round(float(np.clip(x/w, 0, 1)), 4), round(float(np.clip(y/h, 0, 1)), 4)

def norm_dist(d, max_d):
    return round(float(np.clip(d/max_d, 0, 1)), 4) if max_d > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class CSVLogger:
    """
    Writes 3 CSV files:

    court_keypoints.csv     — court corners per detection (normalised 0-1)
    player_biomechanics.csv — per player per frame: all joint angles + kinematics
    shot_events.csv         — per confirmed shot: full biomechanics + shot metrics

    All numeric values normalised to [0.0, 1.0]:
      • Joint angles  : value / 180
      • Positions     : pixel / frame_dimension
      • Distances     : value / frame_diagonal
      • Speed         : value / max_observed
      • Reaction time : value / MAX_REACTION_S cap
      • Flags         : 0 or 1
      • Quality score : BAD=0.0, AVERAGE=0.5, GOOD=1.0
    """

    # ── Column definitions ───────────────────────────────────────────────────
    COURT_COLS = (
        ["frame", "timestamp_s"] +
        [f"kpt_{i}_{ax}" for i in range(20) for ax in ("x", "y")]
    )

    BIO_COLS = [
        # ── Identity & time ─────────────────────────────────────────────────
        "frame", "timestamp_s", "player_id",
        # ── Position on court ───────────────────────────────────────────────
        "pos_x_norm",           # feet x / frame_width
        "pos_y_norm",           # feet y / frame_height
        "court_zone_norm",      # NET=0.0, MID=0.5, BACK=1.0
        "shuttle_dist_norm",    # wrist→shuttle / frame_diagonal
        # ── Upper body joint angles (value/180) ─────────────────────────────
        "right_shoulder_angle", # neck→r_shoulder→r_elbow
        "left_shoulder_angle",  # neck→l_shoulder→l_elbow
        "right_elbow_angle",    # r_shoulder→r_elbow→r_wrist
        "left_elbow_angle",     # l_shoulder→l_elbow→l_wrist
        "right_wrist_angle",    # r_elbow→r_wrist→r_shoulder (arm alignment)
        "left_wrist_angle",
        # ── Lower body joint angles ──────────────────────────────────────────
        "right_hip_angle",      # r_shoulder→r_hip→r_knee
        "left_hip_angle",
        "right_knee_angle",     # r_hip→r_knee→r_ankle
        "left_knee_angle",
        "right_ankle_angle",    # r_knee→r_ankle→r_hip
        "left_ankle_angle",
        # ── Posture metrics ──────────────────────────────────────────────────
        "torso_lean_norm",      # forward lean vs vertical (0=upright, 1=max lean)
        "body_symmetry_norm",   # shoulder height diff / body_ref (0=perfect)
        "stance_width_norm",    # ankle-to-ankle / body_ref*1.5
        # ── Dynamics ────────────────────────────────────────────────────────
        "wrist_speed_norm",     # swing displacement / body_ref*2
        "elbow_extension_norm", # dominant elbow angle/180 (1=fully extended)
    ]

    SHOT_COLS = BIO_COLS + [
        # ── Shot identification ──────────────────────────────────────────────
        "shot_type",
        "quality",              # GOOD / AVERAGE / BAD
        "quality_score",        # 1.0 / 0.5 / 0.0
        # ── Shuttle metrics at contact ───────────────────────────────────────
        "shuttle_speed_kmh",    # raw km/h for readability
        "shuttle_speed_norm",   # normalised by max observed this session
        # ── Timing ──────────────────────────────────────────────────────────
        "reaction_time_s",      # seconds since player's last shot
        "reaction_time_norm",   # / MAX_REACTION_S (0=instant, 1=very slow)
        # ── Contact quality ──────────────────────────────────────────────────
        "contact_height_norm",  # 1-(wrist_y/frame_h): 1=top, 0=bottom
        # ── Recovery ────────────────────────────────────────────────────────
        "recovery_score",       # 0-1: how centred player is after shot
    ]

    MAX_REACTION_S = 10.0

    def __init__(self, output_dir="csv_output"):
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        self._court_f = open(f"{output_dir}/court_keypoints_{ts}.csv",
                             "w", newline="", encoding="utf-8")
        self._bio_f   = open(f"{output_dir}/player_biomechanics_{ts}.csv",
                             "w", newline="", encoding="utf-8")
        self._shot_f  = open(f"{output_dir}/shot_events_{ts}.csv",
                             "w", newline="", encoding="utf-8")

        self._court_w = csv.DictWriter(self._court_f, fieldnames=self.COURT_COLS)
        self._bio_w   = csv.DictWriter(self._bio_f,   fieldnames=self.BIO_COLS)
        self._shot_w  = csv.DictWriter(self._shot_f,  fieldnames=self.SHOT_COLS)

        for w in (self._court_w, self._bio_w, self._shot_w):
            w.writeheader()

        self._max_shuttle_speed = 1.0
        self._last_shot_ts: dict = {}
        self._center_x = 0.5
        self._center_y = 0.5

        print(f"📊 CSV output → {output_dir}/")
        print(f"   court_keypoints_{ts}.csv")
        print(f"   player_biomechanics_{ts}.csv")
        print(f"   shot_events_{ts}.csv\n")

    def close(self):
        for f in (self._court_f, self._bio_f, self._shot_f):
            f.close()
        print("✅ CSV files saved.")

    # ── 1. Court keypoints ────────────────────────────────────────────────────
    def log_court(self, frame_idx, results, frame_w, frame_h):
        if results is None or results.keypoints is None: return
        if len(results.keypoints.xy) == 0: return
        kpts = results.keypoints.xy.cpu().numpy()[0]
        row  = {"frame": frame_idx, "timestamp_s": round(time.time(), 3)}
        for i in range(20):
            if i < len(kpts) and kpts[i][0] > 1:
                nx, ny = norm_pos(kpts[i][0], kpts[i][1], frame_w, frame_h)
            else:
                nx, ny = 0.0, 0.0
            row[f"kpt_{i}_x"] = nx
            row[f"kpt_{i}_y"] = ny
        self._court_w.writerow(row)
        self._court_f.flush()

    # ── 2. Player biomechanics ────────────────────────────────────────────────
    def log_biomechanics(self, frame_idx, player, shuttle,
                         zone, frame_w, frame_h, wrist_speed, body_ref):
        k   = player['kpts']
        pid = player['id']
        fx, fy = player['feet']
        frame_diag = math.hypot(frame_w, frame_h)

        px_n, py_n = norm_pos(fx, fy, frame_w, frame_h)
        zone_map   = {"NET": 0.0, "MID": 0.5, "BACK": 1.0, "UNKNOWN": 0.5}
        zone_n     = zone_map.get(zone, 0.5)

        wrist    = self._dominant_wrist(k)
        wp       = tuple(wrist) if wrist is not None else (fx, fy)
        s_dist_n = norm_dist(shuttle.distance_to(wp), frame_diag)

        # Helper: safe keypoint access
        def a(i):
            return k[i] if len(k) > i and k[i][0] > 1 else np.array([0.0, 0.0])

        # Joint angles  (COCO layout)
        r_sh  = norm_angle(angle_between(a(0),  a(6),  a(8)))
        l_sh  = norm_angle(angle_between(a(0),  a(5),  a(7)))
        r_el  = norm_angle(angle_between(a(6),  a(8),  a(10)))
        l_el  = norm_angle(angle_between(a(5),  a(7),  a(9)))
        r_wr  = norm_angle(angle_between(a(8),  a(10), a(6)))
        l_wr  = norm_angle(angle_between(a(7),  a(9),  a(5)))
        r_hip = norm_angle(angle_between(a(6),  a(12), a(14)))
        l_hip = norm_angle(angle_between(a(5),  a(11), a(13)))
        r_kn  = norm_angle(angle_between(a(12), a(14), a(16)))
        l_kn  = norm_angle(angle_between(a(11), a(13), a(15)))
        r_an  = norm_angle(angle_between(a(14), a(16), a(12)))
        l_an  = norm_angle(angle_between(a(13), a(15), a(11)))

        # Torso lean vs vertical
        sh_mid = (a(5) + a(6)) / 2
        hi_mid = (a(11) + a(12)) / 2
        if sh_mid[0] > 1 and hi_mid[0] > 1:
            tv  = sh_mid - hi_mid
            cos = np.dot(tv, [0,-1]) / (np.linalg.norm(tv) + 1e-6)
            torso_lean = norm_angle(math.degrees(math.acos(np.clip(cos, -1, 1))))
        else:
            torso_lean = 0.5

        # Body symmetry: shoulder height diff
        body_sym = norm_dist(abs(float(a(5)[1]) - float(a(6)[1])), body_ref) \
                   if a(5)[0] > 1 and a(6)[0] > 1 else 0.0

        # Stance width: ankle separation
        stance_n = norm_dist(float(np.linalg.norm(a(15) - a(16))), body_ref * 1.5) \
                   if a(15)[0] > 1 and a(16)[0] > 1 else 0.0

        # Jump
        waist_y = (float(a(11)[1]) + float(a(12)[1])) / 2
        ankles  = [float(a(i)[1]) for i in [15, 16] if a(i)[0] > 1]
        jump_flag, jump_h_n = 0, 0.0
        if ankles and body_ref > 0:
            if min(ankles) < waist_y + body_ref * 0.35:
                jump_flag = 1
                rise      = (waist_y + body_ref * 0.35) - min(ankles)
                jump_h_n  = norm_dist(rise, body_ref * 0.6)

        wrist_spd_n  = norm_dist(wrist_speed, body_ref * 2.0)
        elbow_ext    = max(r_el, l_el)

        row = {
            "frame": frame_idx, "timestamp_s": round(time.time(), 3),
            "player_id": pid,
            "pos_x_norm": px_n, "pos_y_norm": py_n,
            "court_zone_norm":      zone_n,
            "shuttle_dist_norm":    s_dist_n,
            "right_shoulder_angle": r_sh,  "left_shoulder_angle": l_sh,
            "right_elbow_angle":    r_el,  "left_elbow_angle":    l_el,
            "right_wrist_angle":    r_wr,  "left_wrist_angle":    l_wr,
            "right_hip_angle":      r_hip, "left_hip_angle":      l_hip,
            "right_knee_angle":     r_kn,  "left_knee_angle":     l_kn,
            "right_ankle_angle":    r_an,  "left_ankle_angle":    l_an,
            "torso_lean_norm":      torso_lean,
            "body_symmetry_norm":   round(body_sym, 4),
            "stance_width_norm":    round(stance_n, 4),
            "wrist_speed_norm":     round(wrist_spd_n, 4),
            "elbow_extension_norm": elbow_ext,
        }
        self._bio_w.writerow(row)
        self._bio_f.flush()
        return row

    # ── 3. Shot events ────────────────────────────────────────────────────────
    def log_shot(self, bio_row, shot, quality, tip,
                 shuttle, fps, px_per_meter, frame_h):
        pid = bio_row["player_id"]

        spd_kmh = shuttle.speed_kmh(fps, px_per_meter)
        self._max_shuttle_speed = max(self._max_shuttle_speed, spd_kmh)
        spd_norm = norm_dist(spd_kmh, self._max_shuttle_speed)

        now      = time.time()
        last_ts  = self._last_shot_ts.get(pid, None)
        react_s  = round(now - last_ts, 3) if last_ts else self.MAX_REACTION_S
        react_n  = norm_dist(min(react_s, self.MAX_REACTION_S), self.MAX_REACTION_S)
        self._last_shot_ts[pid] = now

        # Contact height: higher wrist = better for power shots
        # pos_y_norm is feet; use as proxy (wrist_y not stored separately)
        # 1 - pos_y gives top-of-frame = 1
        contact_h_n = round(1.0 - bio_row["pos_y_norm"], 4)

        # Recovery: distance of player from court centre
        dx = abs(bio_row["pos_x_norm"] - self._center_x)
        dy = abs(bio_row["pos_y_norm"] - self._center_y)
        recovery = round(max(0.0, 1.0 - math.hypot(dx, dy) / 0.5), 4)

        row = {
            **bio_row,
            "shot_type":           shot,
            "quality":             quality,
            "quality_score":       QUALITY_SCORE.get(quality, 0.5),
            "shuttle_speed_kmh":   round(spd_kmh, 2),
            "shuttle_speed_norm":  spd_norm,
            "reaction_time_s":     react_s,
            "reaction_time_norm":  round(react_n, 4),
            "contact_height_norm": contact_h_n,
            "recovery_score":      recovery,
        }
        self._shot_w.writerow(row)
        self._shot_f.flush()

    def set_court_centre(self, transformer):
        try:
            total_w = transformer.map_w + 2 * transformer.margin
            total_h = transformer.map_h + 2 * transformer.margin
            self._center_x = (transformer.margin + transformer.map_w/2) / total_w
            self._center_y = (transformer.margin + transformer.map_h/2) / total_h
        except Exception:
            pass

    @staticmethod
    def _dominant_wrist(k):
        if len(k) < 17: return None
        wr, wl = k[10], k[9]
        if wr[0] > 5 and wl[0] > 5: return wr if wr[1] < wl[1] else wl
        if wr[0] > 5: return wr
        if wl[0] > 5: return wl
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SHOT DISPLAY CACHE
# ─────────────────────────────────────────────────────────────────────────────
class ShotDisplayCache:
    DISPLAY_DURATION = 2.5
    def __init__(self): self._cache: dict = {}

    def update(self, pid, shot, quality, tip):
        if shot:
            self._cache[pid] = {'shot':shot,'quality':quality,
                                'tip':tip,'ts':time.time()}

    def get(self, pid) -> dict:
        entry = self._cache.get(pid)
        if not entry: return {}
        age = time.time() - entry['ts']
        if age > self.DISPLAY_DURATION:
            del self._cache[pid]; return {}
        fade = self.DISPLAY_DURATION - 0.5
        alpha = 1.0 if age < fade else max(0.0, 1.0-(age-fade)/0.5)
        return {**entry, 'alpha': alpha}

    def clear(self, pid): self._cache.pop(pid, None)


# ─────────────────────────────────────────────────────────────────────────────
# SHOT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class ShotHistory:
    MAX = 5
    def __init__(self): self._hist: dict = {}
    def log(self, pid, shot, quality):
        if not shot: return
        self._hist.setdefault(pid, deque(maxlen=self.MAX))
        self._hist[pid].append((shot, quality, time.time()))
    def get(self, pid) -> list: return list(self._hist.get(pid, []))


# ─────────────────────────────────────────────────────────────────────────────
# 1. HIGHLIGHT MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class HighlightManager:
    def __init__(self, output_dir="highlights", fps=30):
        self.output_dir=output_dir; self.fps=fps
        os.makedirs(output_dir, exist_ok=True)
        self.frame_buffer=deque(maxlen=int(fps*2))
        self.is_recording=False; self.out_video=None; self.record_frames_left=0

    def add_to_buffer(self, frame):
        self.frame_buffer.append(frame.copy())
        if self.is_recording and self.out_video:
            self.out_video.write(frame); self.record_frames_left -= 1
            if self.record_frames_left <= 0: self.stop_recording()

    def start_highlight(self, shot_type, quality, player_id, frame_idx):
        if self.is_recording or not self.frame_buffer: return
        fn=f"{self.output_dir}/{shot_type}_{quality}_ID{player_id}_F{frame_idx}.mp4"
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'); h,w=self.frame_buffer[0].shape[:2]
        self.out_video=cv2.VideoWriter(fn,fourcc,self.fps,(w,h))
        for f in self.frame_buffer: self.out_video.write(f)
        self.is_recording=True; self.record_frames_left=int(self.fps*2)
        print(f"🎬 [{quality}] {shot_type} — ID{player_id} @ Frame {frame_idx}")

    def stop_recording(self):
        if self.out_video: self.out_video.release()
        self.out_video=None; self.is_recording=False


# ─────────────────────────────────────────────────────────────────────────────
# 2. COURT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class CourtDetector:
    def __init__(self, model_path): self.model=YOLO(model_path)
    def predict(self, frame): return self.model(frame,verbose=False,conf=0.25)[0]

    def get_court_box(self, results):
        if results and len(results.boxes)>0:
            box=results.boxes.xyxy.cpu().numpy()[0]
            return (int(box[0]-60),int(box[1]-60),int(box[2]+60),int(box[3]+60))
        return None

    def get_court_corners(self, results):
        if results and results.keypoints is not None and len(results.keypoints.xy)>0:
            pts=results.keypoints.xy.cpu().numpy()[0]
            valid=[p for p in pts if p[0]>10 and p[1]>10]
            if len(valid)>=4: return np.array(valid,dtype=np.float32)
        return None

    def draw_court(self, frame, results):
        if results is None: return frame
        if results.keypoints is not None and len(results.keypoints.xy)>0:
            kpts=results.keypoints.xy.cpu().numpy()
            for i,pt in enumerate(kpts[0]):
                x,y=int(pt[0]),int(pt[1])
                if x>10:
                    cv2.circle(frame,(x,y),6,(0,255,0),-1)
                    cv2.putText(frame,f"C{i}",(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHUTTLE TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class ShuttleTracker:
    LOST_DECAY_FRAMES=5
    def __init__(self, model_path, history_len=25):
        self.model=YOLO(model_path)
        self.smooth_history=deque(maxlen=history_len)
        self.velocity=np.array([0.0,0.0]); self.speed=0.0
        self._alpha=0.4; self._frames_lost=0

    def predict(self, frame): return self.model(frame,verbose=False,conf=0.45)[0]

    def update(self, results):
        if results and len(results.boxes)>0:
            self._frames_lost=0
            confs=results.boxes.conf.cpu().numpy(); best=int(np.argmax(confs))
            box=results.boxes.xyxy.cpu().numpy()[best]
            cx=float((box[0]+box[2])/2); cy=float((box[1]+box[3])/2)
            if self.smooth_history:
                px,py=self.smooth_history[-1]
                cx=self._alpha*cx+(1-self._alpha)*px
                cy=self._alpha*cy+(1-self._alpha)*py
            self.smooth_history.append((cx,cy))
            if len(self.smooth_history)>=3:
                self.velocity=(np.array(self.smooth_history[-1])
                               -np.array(self.smooth_history[-3]))
                self.speed=float(np.linalg.norm(self.velocity))
            return (int(cx),int(cy))
        self._frames_lost+=1
        if self._frames_lost>=self.LOST_DECAY_FRAMES:
            self.velocity=np.array([0.0,0.0]); self.speed=0.0
        return None

    @property
    def direction(self):
        if self.speed<3: return "STATIC"
        vy=self.velocity[1]
        if vy<-5: return "UP"
        if vy>5:  return "DOWN"
        return "FLAT"

    def distance_to(self, point):
        if not self.smooth_history: return 9999.0
        return float(np.linalg.norm(
            np.array(self.smooth_history[-1])-np.array(point,dtype=float)))

    def speed_kmh(self, fps, px_per_meter):
        if px_per_meter<=0 or fps<=0: return 0.0
        return self.speed*fps*3.6/px_per_meter

    def draw(self, frame, fps=30, px_per_meter=1.0):
        pts=list(self.smooth_history)
        for i in range(1,len(pts)):
            fade=int(220*i/len(pts))
            cv2.line(frame,(int(pts[i-1][0]),int(pts[i-1][1])),
                     (int(pts[i][0]),int(pts[i][1])),(0,fade,255),2)
        if pts:
            cx,cy=int(pts[-1][0]),int(pts[-1][1])
            cv2.circle(frame,(cx,cy),7,(0,0,255),-1)
            spd=self.speed_kmh(fps,px_per_meter)
            if spd>1:
                cv2.putText(frame,f"{spd:.0f} km/h",(cx+10,cy-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,220,255),1,cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHOT ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
class ShotAnalyser:
    CONFIRM_FRAMES={"SMASH":3,"CLEAR":3,"DRIVE":3,"DROP":3,"NETSHOT":2,"LIFT":3}
    MAX_GAP={"SMASH":0,"CLEAR":0,"DRIVE":0,"DROP":0,"NETSHOT":1,"LIFT":0}
    SHOT_COOLDOWN=1.5; WRIST_HIST_LEN=12; SWING_WINDOW=5

    def __init__(self, transformer):
        self.transformer=transformer
        self.wrist_hist:dict={}; self.shot_buffer:dict={}
        self.gap_counter:dict={}; self.last_shot_ts:dict={}

    def analyse(self, player, shuttle):
        pid=player['id']; kpts=player['kpts']
        if pid not in self.wrist_hist:
            self.wrist_hist[pid]=deque(maxlen=self.WRIST_HIST_LEN)
        wrist=self._dominant_wrist(kpts)
        if wrist is not None: self.wrist_hist[pid].append(tuple(wrist))
        if len(self.wrist_hist.get(pid,[]))<5: return "","",""
        if time.time()-self.last_shot_ts.get(pid,0)<self.SHOT_COOLDOWN: return "","",""
        zone=self._get_zone(player['feet'])
        wp=wrist if wrist is not None else player['feet']
        near_shuttle=shuttle.distance_to(wp)<200
        raw_shot,raw_quality=self._detect(pid,kpts,shuttle,zone,near_shuttle)
        if pid not in self.shot_buffer: self.shot_buffer[pid]=[]
        if pid not in self.gap_counter: self.gap_counter[pid]=0
        if raw_shot:
            self.gap_counter[pid]=0; self.shot_buffer[pid].append((raw_shot,raw_quality))
        else:
            buffered=[s[0] for s in self.shot_buffer[pid]]
            if buffered:
                cur=max(set(buffered),key=buffered.count)
                if self.gap_counter[pid]<self.MAX_GAP.get(cur,0):
                    self.gap_counter[pid]+=1
                    self.shot_buffer[pid].append((cur,self.shot_buffer[pid][-1][1]))
                else: self.shot_buffer[pid]=[]; self.gap_counter[pid]=0
            else: self.shot_buffer[pid]=[]; self.gap_counter[pid]=0
        if self.shot_buffer[pid]:
            shots=[s[0] for s in self.shot_buffer[pid]]
            dominant=max(set(shots),key=shots.count)
            needed=self.CONFIRM_FRAMES.get(dominant,3)
            if len(self.shot_buffer[pid])>=needed:
                qualities=[s[1] for s in self.shot_buffer[pid] if s[0]==dominant]
                quality=max(set(qualities),key=qualities.count)
                tip=COACHING_TIPS.get((dominant,quality),"")
                self.shot_buffer[pid]=[]; self.gap_counter[pid]=0
                self.last_shot_ts[pid]=time.time()
                return dominant,quality,tip
        return "","",""

    def get_zone(self, feet): return self._get_zone(feet)

    def get_wrist_speed(self, pid, body_ref):
        h=self.wrist_hist.get(pid)
        if not h or len(h)<3: return 0.0
        window=min(self.SWING_WINDOW,len(h)-1)
        dx=abs(h[-1][0]-h[len(h)-1-window][0])
        dy=abs(h[-1][1]-h[len(h)-1-window][1])
        return float(math.hypot(dx,dy))

    def get_body_ref(self, k):
        nose_y=k[0][1]
        waist_y=((k[11][1]+k[12][1])/2 if k[11][1]>0 and k[12][1]>0 else k[12][1])
        ref=abs(waist_y-nose_y); return ref if ref>40 else 100

    def _dominant_wrist(self,k):
        if len(k)<17: return None
        wr,wl=k[10],k[9]
        if wr[0]>5 and wl[0]>5: return wr if wr[1]<wl[1] else wl
        if wr[0]>5: return wr
        if wl[0]>5: return wl
        return None

    def _body_ref(self,k): return self.get_body_ref(k)

    def _is_jumping(self,k,p_ref):
        waist_y=(k[11][1]+k[12][1])/2
        ankles=[a[1] for a in [k[15],k[16]] if a[1]>0]
        if not ankles: return False
        return min(ankles)<(waist_y+p_ref*0.35)

    def _swing_motion(self,pid):
        h=self.wrist_hist[pid]
        if len(h)<3: return 0.0,0.0
        window=min(self.SWING_WINDOW,len(h)-1); idx=len(h)-1-window
        return float(abs(h[-1][0]-h[idx][0])),float(h[-1][1]-h[idx][1])

    def _elbow_angle(self,k):
        r=angle_between(k[6],k[8],k[10]) if all(k[i][0]>5 for i in [6,8,10]) else 0
        l=angle_between(k[5],k[7],k[9])  if all(k[i][0]>5 for i in [5,7,9])  else 0
        return max(r,l)

    def _get_zone(self,feet):
        if self.transformer.matrix is None: return "UNKNOWN"
        pos=self.transformer.transform_point(float(feet[0]),float(feet[1]))
        if pos is None: return "UNKNOWN"
        net_y=self.transformer.margin+self.transformer.map_h//2
        half_h=self.transformer.map_h/2.0
        dist=abs(float(pos[1])-net_y)/half_h if half_h>0 else 0.5
        if dist<0.22: return "NET"
        if dist<0.60: return "MID"
        return "BACK"

    def _detect(self,pid,k,shuttle,zone,near_shuttle):
        if len(k)<17: return "",""
        p_ref=self._body_ref(k); wrist=self._dominant_wrist(k)
        if wrist is None: return "",""
        nose_y=k[0][1]; avg_sh=(k[5][1]+k[6][1])/2; avg_wa=(k[11][1]+k[12][1])/2
        jumping=self._is_jumping(k,p_ref); dx,dy=self._swing_motion(pid)
        s_dir=shuttle.direction; s_spd=shuttle.speed; elbow_a=self._elbow_angle(k)

        if (wrist[1]<=nose_y+p_ref*0.15 or jumping) and dy>p_ref*0.16:
            shot="SMASH"
            if jumping and wrist[1]<nose_y and elbow_a>155: quality="GOOD"
            elif wrist[1]<=nose_y and elbow_a>130:           quality="AVERAGE"
            else:                                             quality="BAD"
            if near_shuttle and s_dir=="DOWN" and s_spd>12:
                quality="GOOD" if quality=="AVERAGE" else quality
            return shot,quality

        if (wrist[1]<avg_sh and dy>p_ref*0.12
                and zone in ("MID","BACK","UNKNOWN") and s_dir in ("UP","STATIC")):
            shot="CLEAR"
            if wrist[1]<nose_y and elbow_a>145 and dx>p_ref*0.08: quality="GOOD"
            elif wrist[1]<avg_sh: quality="AVERAGE"
            else:                 quality="BAD"
            return shot,quality

        if (wrist[1]<avg_sh and 0<dy<p_ref*0.14
                and zone in ("MID","BACK","UNKNOWN") and s_dir in ("DOWN","STATIC")):
            shot="DROP"
            if near_shuttle and s_dir=="DOWN" and s_spd<14: quality="GOOD"
            elif near_shuttle: quality="AVERAGE"
            else:              quality="BAD"
            return shot,quality

        if (avg_sh-p_ref*0.30<wrist[1]<avg_wa+p_ref*0.50
                and dx>p_ref*0.22 and abs(dy)<p_ref*0.12):
            shot="DRIVE"
            if near_shuttle and s_dir=="FLAT" and s_spd>10: quality="GOOD"
            elif dx>p_ref*0.32: quality="AVERAGE"
            else:               quality="BAD"
            return shot,quality

        at_net=zone in ("NET","UNKNOWN")
        if (at_net and abs(dx)<p_ref*0.28 and abs(dy)<p_ref*0.22
                and wrist[1]>avg_sh-p_ref*0.4
                and wrist[1]<avg_wa+p_ref*0.5 and not jumping):
            shot="NETSHOT"
            if near_shuttle and s_spd<10: quality="GOOD"
            elif near_shuttle:             quality="AVERAGE"
            elif s_spd<6:                  quality="AVERAGE"
            else:                          quality="BAD"
            return shot,quality

        if wrist[1]>avg_wa and dy<-(p_ref*0.14):
            shot="LIFT"
            if near_shuttle and s_dir=="UP": quality="GOOD"
            elif near_shuttle:               quality="AVERAGE"
            else:                            quality="BAD"
            return shot,quality

        return "",""


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER ID MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class PlayerIDManager:
    """
    Converts volatile ByteTrack IDs → stable player IDs (1, 2, 3, 4).

    Strategy:
    ─────────
    Players are assigned a stable ID on their FIRST appearance based on
    court position:

      Singles (2 players):
        P1 = top half of court  (lower y in image, far side)
        P2 = bottom half        (higher y in image, near side)

      Doubles (3-4 players):
        P1 = top-left,  P2 = top-right
        P3 = bottom-left, P4 = bottom-right

    Re-identification:
    ─────────────────
    If ByteTrack loses a player and assigns a new tracker ID, we match by
    proximity — the new detection closest to each stable player's last known
    position is re-assigned that stable ID. This keeps IDs persistent even
    through occlusions, fast movement or camera cuts.

    Max players capped at 4 (doubles). Any extra detections are ignored.
    """
    MAX_PLAYERS      = 4
    LOST_TIMEOUT_S   = 3.0    # after this, a slot is freed for reassignment
    REASSIGN_DIST_PX = 150    # max pixel dist to re-assign a known player

    # Stable ID colours so each player always shows the same colour
    ID_COLORS = {
        1: (255, 100, 0),    # blue
        2: (0,   200, 255),  # yellow
        3: (180, 0,   255),  # purple
        4: (0,   255, 120),  # green
    }

    def __init__(self):
        # stable_id → {'last_pos': (fx,fy), 'tracker_id': int, 'last_seen': ts}
        self._slots: dict = {}
        self._tracker_to_stable: dict = {}   # tracker_id → stable_id
        self._next_id = 1                     # next stable ID to assign

    def update(self, raw_players: list) -> list:
        """
        Takes raw player list (with ByteTrack IDs) from detect_active.
        Returns same list but with ['id'] replaced by stable 1-4 IDs.
        Also enforces MAX_PLAYERS cap.
        """
        now = time.time()

        # ── Step 1: expire slots that haven't been seen recently ───────────
        expired = [sid for sid, slot in self._slots.items()
                   if now - slot['last_seen'] > self.LOST_TIMEOUT_S]
        for sid in expired:
            old_tid = self._slots[sid]['tracker_id']
            self._tracker_to_stable.pop(old_tid, None)
            del self._slots[sid]

        # ── Step 2: match existing tracker IDs we already know ─────────────
        matched_stable  = set()
        unmatched_raw   = []

        for p in raw_players:
            tid = p['id']
            if tid in self._tracker_to_stable:
                sid = self._tracker_to_stable[tid]
                self._slots[sid]['last_pos']  = p['feet']
                self._slots[sid]['last_seen'] = now
                matched_stable.add(sid)
                p['id'] = sid
            else:
                unmatched_raw.append(p)

        # ── Step 3: try to re-identify unmatched detections ─────────────────
        # Match by proximity to known slots that weren't updated this frame
        unmatched_slots = [sid for sid in self._slots if sid not in matched_stable]

        for p in unmatched_raw[:]:
            best_sid  = None
            best_dist = self.REASSIGN_DIST_PX
            for sid in unmatched_slots:
                lx, ly = self._slots[sid]['last_pos']
                fx, fy = p['feet']
                dist   = math.hypot(fx - lx, fy - ly)
                if dist < best_dist:
                    best_dist = dist
                    best_sid  = sid

            if best_sid is not None:
                # Re-assign: update tracker_id mapping
                old_tid = self._slots[best_sid]['tracker_id']
                self._tracker_to_stable.pop(old_tid, None)
                self._slots[best_sid]['tracker_id'] = p['id']
                self._slots[best_sid]['last_pos']   = p['feet']
                self._slots[best_sid]['last_seen']  = now
                self._tracker_to_stable[p['id']]    = best_sid
                p['id'] = best_sid
                unmatched_slots.remove(best_sid)
                unmatched_raw.remove(p)

        # ── Step 4: new players — assign next stable ID ──────────────────────
        for p in unmatched_raw:
            if len(self._slots) >= self.MAX_PLAYERS:
                p['id'] = -1   # cap exceeded — mark for exclusion
                continue

            # Assign stable ID based on position
            sid = self._position_based_id(p['feet'])
            if sid in self._slots:
                # Position slot taken → take next available number
                used = set(self._slots.keys())
                sid  = next(i for i in range(1, self.MAX_PLAYERS+1) if i not in used)

            self._slots[sid] = {
                'tracker_id': p['id'],
                'last_pos':   p['feet'],
                'last_seen':  now,
            }
            self._tracker_to_stable[p['id']] = sid
            p['id'] = sid

        # ── Step 5: remove capped players and enforce max ───────────────────
        players_out = [p for p in raw_players if p['id'] != -1]
        # Sort by stable ID for consistent ordering
        players_out.sort(key=lambda p: p['id'])
        return players_out

    def _position_based_id(self, feet) -> int:
        """
        Assign stable ID based on court position.
        This gives predictable IDs: top players get lower numbers.
        """
        fx, fy = feet
        # We don't know frame size here, use relative position of existing slots
        if not self._slots:
            return 1   # first player always gets P1

        # Find slots in same half (top/bottom split by average y of known players)
        known_ys = [s['last_pos'][1] for s in self._slots.values()]
        avg_y    = sum(known_ys) / len(known_ys)

        top_half   = fy < avg_y
        used_ids   = set(self._slots.keys())

        # Top players: prefer 1, 2. Bottom: prefer 3, 4
        preferred  = [1, 2] if top_half else [3, 4]
        for pid in preferred:
            if pid not in used_ids:
                return pid

        # Fallback: any free slot
        for pid in range(1, self.MAX_PLAYERS + 1):
            if pid not in used_ids:
                return pid
        return 1

    def color(self, stable_id: int) -> tuple:
        return self.ID_COLORS.get(stable_id, (200, 200, 200))


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLAYER DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class PlayerDetector:
    SKELETON=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
              (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

    def __init__(self,model_path='yolov8n-pose.pt'):
        self.model      = YOLO(model_path)
        self.id_manager = PlayerIDManager()

    def detect_active(self,frame,court_box):
        res=self.model.track(frame,persist=True,tracker="bytetrack.yaml",
                             verbose=False,conf=0.25)[0]
        raw_players=[]
        if court_box is None or res.boxes is None or res.boxes.id is None:
            return raw_players,res
        cx1,cy1,cx2,cy2=court_box
        for i,box in enumerate(res.boxes.xyxy.cpu().numpy()):
            p_id=int(res.boxes.id[i]); px1,py1,px2,py2=map(int,box)
            fx,fy=(px1+px2)//2,py2
            if (cx1<fx<cx2) and (cy1<fy<cy2):
                kpts=res.keypoints.xy[i].cpu().numpy()
                raw_players.append({'id':p_id,'box':(px1,py1,px2,py2),
                                     'feet':(fx,fy),'kpts':kpts,
                                     'shot':'','quality':'','tip':''})
        # Stabilise IDs: volatile ByteTrack IDs → persistent 1-4
        players = self.id_manager.update(raw_players)
        return players,res

    def draw_all(self,frame,players,display_cache,shot_history):
        for p in players:
            for a,b in self.SKELETON:
                pt1=tuple(map(int,p['kpts'][a])); pt2=tuple(map(int,p['kpts'][b]))
                if pt1[0]>5 and pt2[0]>5: cv2.line(frame,pt1,pt2,(0,200,0),2)
            cached=display_cache.get(p['id'])
            shot=cached.get('shot',''); quality=cached.get('quality','')
            tip=cached.get('tip','');   alpha=cached.get('alpha',1.0)
            shot_col  = SHOT_COLORS.get(shot,(200,200,200))
            qual_col  = SHOT_QUALITY_COLORS.get(quality,(200,200,200))
            # Stable player colour for bounding box when no shot active
            pid_col   = self.id_manager.color(p['id'])
            box_col   = qual_col if quality else pid_col
            cv2.rectangle(frame,(p['box'][0],p['box'][1]),(p['box'][2],p['box'][3]),box_col,2)
            # ID badge uses stable player colour
            self._pill(frame,f"P{p['id']}",(p['box'][0],p['box'][1]-28),
                       pid_col,(0,0,0),scale=0.55)
            if shot:
                bcx=(p['box'][0]+p['box'][2])//2
                self._pill_centred(frame,shot,(bcx,p['box'][1]-58),shot_col,(0,0,0),scale=0.92,thickness=2,alpha=alpha)
                self._pill_centred(frame,quality,(bcx,p['box'][1]-28),qual_col,(255,255,255),scale=0.52,thickness=1,alpha=alpha)
                if tip: self._tip_banner(frame,tip,(p['box'][0],p['box'][3]+6),p['box'][2]-p['box'][0],alpha)
            hist=shot_history.get(p['id'])
            if hist: self._draw_timeline(frame,hist,(p['box'][0],p['box'][3]+36),p['box'][2]-p['box'][0])
        return frame

    @staticmethod
    def _blend_rect(frame,x1,y1,x2,y2,color,alpha):
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(frame.shape[1]-1,x2),min(frame.shape[0]-1,y2)
        if x2<=x1 or y2<=y1: return
        roi=frame[y1:y2,x1:x2]; ov=roi.copy(); ov[:]=color
        cv2.addWeighted(ov,alpha,roi,1-alpha,0,roi); frame[y1:y2,x1:x2]=roi

    def _pill(self,frame,text,origin,bg,fg,scale=0.6,thickness=1,pad_x=8,pad_y=4,alpha=1.0):
        font=cv2.FONT_HERSHEY_SIMPLEX; (tw,th),bl=cv2.getTextSize(text,font,scale,thickness)
        x,y=origin; bx1,by1=x,y-th-pad_y; bx2,by2=x+tw+pad_x*2,y+bl+pad_y; r=max(1,(by2-by1)//2)
        self._blend_rect(frame,bx1+r,by1,bx2-r,by2,bg,alpha*0.82)
        for cx,cy in [(bx1+r,by1+r),(bx2-r,by1+r),(bx1+r,by2-r),(bx2-r,by2-r)]:
            ov=frame.copy(); cv2.circle(ov,(cx,cy),r,bg,-1); cv2.addWeighted(ov,alpha*0.82,frame,1-alpha*0.82,0,frame)
        cv2.putText(frame,text,(x+pad_x,y),font,scale,fg,thickness,cv2.LINE_AA)

    def _pill_centred(self,frame,text,centre,bg,fg,scale=0.75,thickness=2,pad_x=12,pad_y=5,alpha=1.0):
        font=cv2.FONT_HERSHEY_SIMPLEX; (tw,th),bl=cv2.getTextSize(text,font,scale,thickness)
        cx,cy=centre; x=cx-tw//2-pad_x; y=cy+th//2
        bx1,by1=x,cy-th//2-pad_y; bx2,by2=x+tw+pad_x*2,cy+th//2+bl+pad_y; r=max(1,(by2-by1)//2)
        self._blend_rect(frame,bx1+r,by1,bx2-r,by2,bg,alpha*0.85)
        for ox,oy in [(bx1+r,by1+r),(bx2-r,by1+r),(bx1+r,by2-r),(bx2-r,by2-r)]:
            ov=frame.copy(); cv2.circle(ov,(ox,oy),r,bg,-1); cv2.addWeighted(ov,alpha*0.85,frame,1-alpha*0.85,0,frame)
        cv2.putText(frame,text,(x+pad_x,y),font,scale,fg,thickness,cv2.LINE_AA)

    def _tip_banner(self,frame,tip,origin,width,alpha=1.0):
        font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.42; (tw,th),_=cv2.getTextSize(tip,font,scale,1)
        x,y=origin; self._blend_rect(frame,x,y,x+max(width,tw+16),y+th+10,(20,20,20),alpha*0.72)
        cv2.putText(frame,tip,(x+6,y+th+4),font,scale,(255,228,80),1,cv2.LINE_AA)

    def _draw_timeline(self,frame,hist,origin,width):
        if not hist: return
        x,y=origin; slot_w=max(1,width//len(hist))
        for i,(shot,quality,_) in enumerate(hist):
            age_a=0.4+0.6*(i+1)/len(hist); sc=SHOT_COLORS.get(shot,(160,160,160))
            bg=tuple(int(c*0.4) for c in sc)
            self._blend_rect(frame,x+i*slot_w,y,x+(i+1)*slot_w-2,y+18,bg,age_a*0.75)
            cv2.putText(frame,shot[:3],(x+i*slot_w+3,y+13),cv2.FONT_HERSHEY_SIMPLEX,0.36,sc,1,cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 6. STATS OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
class StatsOverlay:
    def __init__(self): self.records:dict={}

    def log(self,pid,shot,quality):
        if not shot or not quality: return
        self.records.setdefault(pid,{})
        self.records[pid].setdefault(shot,{"GOOD":0,"AVERAGE":0,"BAD":0})
        self.records[pid][shot][quality]+=1

    def draw(self,frame):
        if not self.records: return frame
        x,y=10,30; lines=1+sum(1+len(s) for s in self.records.values())
        PlayerDetector._blend_rect(frame,x-5,y-22,x+300,y+lines*18+5,(20,20,20),0.72)
        cv2.putText(frame,"── SHOT STATS ──",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,180,180),1)
        y+=20
        for pid,shots in self.records.items():
            cv2.putText(frame,f"Player {pid}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,(80,220,255),1); y+=17
            for shot,counts in shots.items():
                total=sum(counts.values())
                gp=int(100*counts["GOOD"]/total); ap=int(100*counts["AVERAGE"]/total); bp=int(100*counts["BAD"]/total)
                bar_x=x+160; bar_w=100
                cv2.rectangle(frame,(bar_x,y-10),(bar_x+bar_w,y+2),(50,50,50),-1)
                cv2.rectangle(frame,(bar_x,y-10),(bar_x+int(bar_w*gp/100),y+2),(0,200,60),-1)
                cv2.rectangle(frame,(bar_x+int(bar_w*gp/100),y-10),(bar_x+int(bar_w*(gp+ap)/100),y+2),(0,160,220),-1)
                cv2.putText(frame,f"  {shot:<9} G:{gp}% A:{ap}% B:{bp}% ({total})",
                            (x,y),cv2.FONT_HERSHEY_SIMPLEX,0.40,(210,210,170),1); y+=16
        return frame

    def print_summary(self):
        print("\n"+"═"*55); print("  MATCH SHOT SUMMARY"); print("═"*55)
        for pid,shots in self.records.items():
            print(f"\n  Player {pid}")
            print(f"  {'Shot':<10}{'Good':>6}{'Avg':>6}{'Bad':>6}{'Total':>7}"); print("  "+"─"*38)
            for shot,counts in shots.items():
                total=sum(counts.values())
                print(f"  {shot:<10}{counts['GOOD']:>6}{counts['AVERAGE']:>6}{counts['BAD']:>6}{total:>7}")
        print("═"*55+"\n")


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
    csv_logger      = CSVLogger(output_dir="csv_output")

    cap          = cv2.VideoCapture("data/raw/new badminton.mp4")
    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    highlight_mgr.fps = fps
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    px_per_meter = 1.0
    heatmap_accum = None; frame_cnt=0; saved_box=None
    is_homography=False; c_res=None; last_highlight:dict={}

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame_cnt+=1; disp=frame.copy()

        if frame_cnt%30==1 or saved_box is None:
            c_res=court_detector.predict(frame)
            saved_box=court_detector.get_court_box(c_res)
            corners=court_detector.get_court_corners(c_res)
            if corners is not None:
                is_homography=transformer.calculate_matrix(corners)
                if is_homography and transformer.map_h>0:
                    px_per_meter=transformer.map_h/COURT_LENGTH_M
                    csv_logger.set_court_centre(transformer)
            csv_logger.log_court(frame_cnt, c_res, frame_w, frame_h)

        players,_=player_detector.detect_active(frame,saved_box)
        s_res=shuttle_tracker.predict(frame)
        shuttle_pos=shuttle_tracker.update(s_res)

        for p in players:
            shot,quality,tip=shot_analyser.analyse(p,shuttle_tracker)
            p['shot']=shot; p['quality']=quality; p['tip']=tip
            display_cache.update(p['id'],shot,quality,tip)

            zone        = shot_analyser.get_zone(p['feet'])
            body_ref    = shot_analyser.get_body_ref(p['kpts'])
            wrist_speed = shot_analyser.get_wrist_speed(p['id'], body_ref)

            bio_row = csv_logger.log_biomechanics(
                frame_idx=frame_cnt, player=p, shuttle=shuttle_tracker,
                zone=zone, frame_w=frame_w, frame_h=frame_h,
                wrist_speed=wrist_speed, body_ref=body_ref)

            if shot:
                shot_history.log(p['id'],shot,quality)
                stats_overlay.log(p['id'],shot,quality)
                csv_logger.log_shot(
                    bio_row=bio_row, shot=shot, quality=quality, tip=tip,
                    shuttle=shuttle_tracker, fps=fps,
                    px_per_meter=px_per_meter, frame_h=frame_h)
                now=time.time()
                if now-last_highlight.get(p['id'],0)>(3.5 if quality=="GOOD" else 5.0):
                    highlight_mgr.start_highlight(shot,quality,p['id'],frame_cnt)
                    last_highlight[p['id']]=now

        disp=court_detector.draw_court(disp,c_res)
        disp=player_detector.draw_all(disp,players,display_cache,shot_history)
        disp=shuttle_tracker.draw(disp,fps=fps,px_per_meter=px_per_meter)
        disp=stats_overlay.draw(disp)
        highlight_mgr.add_to_buffer(disp)

        if is_homography:
            t_data=[]
            for p in players:
                pos=transformer.transform_point(float(p['feet'][0]),float(p['feet'][1]))
                if pos is not None:
                    cached=display_cache.get(p['id']); s_lbl=cached.get('shot','')
                    if s_lbl=="SMASH": s_lbl="SMASH!"
                    t_data.append({'id':p['id'],'pos':pos,'shot':s_lbl})
            if shuttle_pos is not None:
                s_map=transformer.transform_point(float(shuttle_pos[0]),float(shuttle_pos[1]))
                if s_map is not None: t_data.append({'id':'shuttle','pos':s_map,'shot':''})
            minimap=transformer.draw_minimap(t_data)
            cv2.imshow("Minimap",minimap)
            if heatmap_accum is None:
                heatmap_accum=np.zeros((minimap.shape[0],minimap.shape[1]),dtype=np.float32)
            for d in t_data:
                if d['id']!='shuttle':
                    px,py=int(d['pos'][0]),int(d['pos'][1])
                    if 0<=px<minimap.shape[1] and 0<=py<minimap.shape[0]:
                        cv2.circle(heatmap_accum,(px,py),15,1,-1)

        cv2.imshow("PlaySight AI",cv2.resize(disp,(1080,720)))
        if cv2.waitKey(1)&0xFF==ord('q'): break

    stats_overlay.print_summary()
    csv_logger.close()
    if heatmap_accum is not None:
        norm=cv2.normalize(heatmap_accum,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite("match_heatmap.png",cv2.applyColorMap(norm,cv2.COLORMAP_JET))
        print("✅ Heatmap saved → match_heatmap.png")
    cap.release(); cv2.destroyAllWindows()


if __name__=="__main__":
    main()
