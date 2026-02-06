import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
import math
import torch
import warnings
from dataclasses import dataclass
import os
import hamer
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
import threading

warnings.filterwarnings("ignore")

HAMER_CONFIDENCE_THRESH = 0.5
PINCH_CLOSE_THRESH = 0.05
PINCH_OPEN_THRESH = 0.10
WRIST_FRAME_SMOOTH_ALPHA = 0.6
BOX_LOCK_THRESH = 2.0
DEPTH_SAMPLE_SIZE = 11

@dataclass
class HandState:
    position: np.ndarray
    orientation: np.ndarray
    pinch_dist: float
    is_pinched: bool
    bbox: list
    joints_3d: np.ndarray
    fps: float
    rpy: np.ndarray = None

class OneEuroFilter:
    def __init__(self, x0, t0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=np.float64)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        x = np.array(x, dtype=np.float64)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class RealsenseCamera:
    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        self.serial = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if serial_number:
            self.config.enable_device(serial_number)
            
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        self.align = rs.align(rs.stream.color)
        self.spat_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()
        
        self.profile = self.pipeline.start(self.config)
        self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # Threading support
        self.running = False
        self.thread = None
        self.latest_frames = (None, None, None)
        self.lock = threading.Lock()
        
        print(f"[INFO] RealSense Camera Started (Serial: {serial_number if serial_number else 'Default'})")

    def start_async(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"[INFO] Camera {self.serial} started async mode.")

    def _update_loop(self):
        while self.running:
            frames = self.get_frames()
            if frames[0] is not None:
                with self.lock:
                    self.latest_frames = frames
            else:
                time.sleep(0.001)

    def get_latest_frames(self):
        with self.lock:
            return self.latest_frames

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None, None
                
            filtered_depth = self.spat_filter.process(depth_frame)
            filtered_depth = self.temp_filter.process(filtered_depth)
            
            return np.asanyarray(color_frame.get_data()), \
                   np.asanyarray(filtered_depth.get_data()), \
                   filtered_depth.as_depth_frame()
        except RuntimeError:
            return None, None, None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pipeline.stop()

class RobustTracker:
    def __init__(self, hand_cam_serial=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using Device: {self.device}")
        
        self.hand_cam = RealsenseCamera(serial_number=hand_cam_serial)
        self.intrinsics = self.hand_cam.intrinsics
        
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5
        )
        
        print("[INFO] Loading HaMeR Model...")
        old_cwd = os.getcwd()
        try:
            hamer_pkg = os.path.dirname(hamer.__file__)
            hamer_root = os.path.dirname(hamer_pkg)
            os.chdir(hamer_root)
            self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        finally:
            os.chdir(old_cwd)
            
        self.model = self.model.to(self.device).eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).float()
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).float()
        
        self.prev_box = None
        self.pinch_state = False
        self.filter_pos = None
        self.filter_rot = None

        self.box_filter = OneEuroFilter(np.zeros(3), 0, min_cutoff=0.01, beta=0.005)
        self.z_filter = OneEuroFilter(0.5, 0, min_cutoff=0.5, beta=0.05)
        
        self.frame_count = 0
        self.skip_rate = 3
        self.last_hamer_joints_local = None
        self.locked_box = None

    def get_mediapipe_box(self, img_rgb):
        h, w = img_rgb.shape[:2]
        img_flipped = cv2.flip(img_rgb, 1)
        results = self.mp_hands.process(img_flipped)
        
        if not results.multi_hand_landmarks:
            self.prev_box = None
            return None
            
        target_idx = -1
        for i, handedness in enumerate(results.multi_handedness):
            if handedness.classification[0].label == "Left":
                target_idx = i
                break
                
        if target_idx == -1:
            self.prev_box = None
            return None 

        lm = results.multi_hand_landmarks[target_idx]
        for pt in lm.landmark: pt.x = 1.0 - pt.x
        
        x_list = [pt.x * w for pt in lm.landmark]
        y_list = [pt.y * h for pt in lm.landmark]
        
        min_x, max_x = min(x_list), max(x_list)
        min_y, max_y = min(y_list), max(y_list)
        
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        box_size = max(max_x - min_x, max_y - min_y) * 1.5 
        
        if self.prev_box is not None:
            alpha = 0.6
            cx = self.prev_box[0] * alpha + cx * (1-alpha)
            cy = self.prev_box[1] * alpha + cy * (1-alpha)
            box_size = self.prev_box[2] * alpha + box_size * (1-alpha)
            
        self.prev_box = [cx, cy, box_size]
        
        s = int(box_size)
        cx, cy = int(cx), int(cy)
        x = max(0, cx - s//2)
        y = max(0, cy - s//2)
        w_box, h_box = min(w - x, s), min(h - y, s)
        
        return [x, y, w_box, h_box], lm

    def estimate_rigid_orientation(self, joints_3d):
        wrist = joints_3d[0]
        v1 = joints_3d[5] - wrist
        v1 /= np.linalg.norm(v1)
        v2 = joints_3d[17] - wrist
        v2 /= np.linalg.norm(v2)
        
        z_vec = np.cross(v1, v2)
        norm_z = np.linalg.norm(z_vec)
        if norm_z < 1e-6: return np.eye(3)
        z_vec /= norm_z
        
        y_vec = np.cross(z_vec, v1)
        y_vec /= np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, z_vec) 
        
        v_forward = joints_3d[9] - wrist
        proj_forward = v_forward - np.dot(v_forward, z_vec) * z_vec
        proj_forward /= (np.linalg.norm(proj_forward) + 1e-9)
        
        final_y = np.cross(z_vec, proj_forward)
        return np.column_stack((proj_forward, final_y, z_vec))

    def get_robust_wrist_depth(self, depth_frame, x, y):
        h, w = depth_frame.get_height(), depth_frame.get_width()
        half = DEPTH_SAMPLE_SIZE // 2
        
        valid_depths = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    d = depth_frame.get_distance(px, py)
                    if d > 0 and d < 2.0: valid_depths.append(d)
        
        if not valid_depths: return 0.5
            
        valid_depths.sort()
        n = len(valid_depths)
        q1, q3 = valid_depths[int(n * 0.25)], valid_depths[int(n * 0.75)]
        clean_depths = [v for v in valid_depths if (q1 - 1.5*(q3-q1)) <= v <= (q3 + 1.5*(q3-q1))]
        
        return np.median(clean_depths) if clean_depths else np.median(valid_depths)

    def process_frame(self) -> HandState:
        t_now = time.time()
        self.frame_count += 1
        
        img_bgr, _, depth_frame_obj = self.hand_cam.get_frames()
        if img_bgr is None: return None, None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        box_data = self.get_mediapipe_box(img_rgb)
        if not box_data: 
            self.last_hamer_joints_local = None
            return img_bgr, None

        bbox_raw, mp_lm = box_data
        cx_raw = bbox_raw[0] + bbox_raw[2]/2
        cy_raw = bbox_raw[1] + bbox_raw[3]/2
        s_raw = max(bbox_raw[2], bbox_raw[3])
        
        box_smooth = self.box_filter(t_now, np.array([cx_raw, cy_raw, s_raw]))
        
        final_box = box_smooth 
        if self.locked_box is None:
            self.locked_box = box_smooth
        else:
            dist = np.linalg.norm(box_smooth[:2] - self.locked_box[:2])
            size_diff = abs(box_smooth[2] - self.locked_box[2])
            if dist > BOX_LOCK_THRESH or size_diff > BOX_LOCK_THRESH:
                self.locked_box = box_smooth
            final_box = self.locked_box
            
        s, cx, cy = int(final_box[2]), int(final_box[0]), int(final_box[1])
        x, y = max(0, cx - s//2), max(0, cy - s//2)
        w_box, h_box = min(w - x, s), min(h - y, s)
        
        should_run_hamer = (self.frame_count % self.skip_rate == 0) or (self.last_hamer_joints_local is None)
        pred_joints_centered = None
        
        if should_run_hamer:
            crop = img_rgb[y:y+h_box, x:x+w_box]
            if crop.size > 0:
                crop_input = cv2.flip(crop, 1)
                try:
                    _inp = cv2.resize(crop_input, (256, 256))
                    _inp = torch.from_numpy(_inp).float().to(self.device) / 255.0
                    _inp = _inp.permute(2, 0, 1).unsqueeze(0)
                    _inp = (_inp - self.mean) / self.std
                    
                    with torch.no_grad():
                        out = self.model({'img': _inp})
                        
                    pred_joints = out['pred_keypoints_3d'][0].cpu().numpy()
                    pred_joints[:, 0] *= -1 
                    
                    wrist_local = pred_joints[0].copy()
                    pred_joints_centered = pred_joints - wrist_local
                    self.last_hamer_joints_local = pred_joints_centered
                except Exception: pass
        else:
            pred_joints_centered = self.last_hamer_joints_local
            
        if pred_joints_centered is None: return img_bgr, None

        wrist_px_x = int(mp_lm.landmark[0].x * w)
        wrist_px_y = int(mp_lm.landmark[0].y * h)
        
        z_wrist_raw = self.get_robust_wrist_depth(depth_frame_obj, wrist_px_x, wrist_px_y)
        z_wrist_m = self.z_filter(t_now, z_wrist_raw)
        
        wrist_pt_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [wrist_px_x, wrist_px_y], z_wrist_m)
        pos_3d = np.array(wrist_pt_3d)
        
        R = self.estimate_rigid_orientation(pred_joints_centered)
        
        if self.filter_rot is None: self.filter_rot = OneEuroFilter(R, t_now, min_cutoff=0.5, beta=0.01)
        R_smooth = self.filter_rot(t_now, R)
        
        if self.filter_pos is None: self.filter_pos = OneEuroFilter(pos_3d, t_now, min_cutoff=1.0, beta=0.02)
        pos_smooth = self.filter_pos(t_now, pos_3d)

        R_pos_map = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        pos_rob = R_pos_map @ pos_smooth
        
        R_rot_map = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        R_hand_local = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        R_rob = R_rot_map @ R_smooth @ R_hand_local
        
        pos_rob += np.array([0.3, -0.1, 0.2])
        
        pinch_dist = np.linalg.norm(pred_joints_centered[4] - pred_joints_centered[8])
        if self.pinch_state:
            if pinch_dist > PINCH_OPEN_THRESH: self.pinch_state = False
        else:
            if pinch_dist < PINCH_CLOSE_THRESH: self.pinch_state = True
            
        color = (0, 255, 0) if should_run_hamer else (0, 255, 255)
        cv2.circle(img_bgr, (wrist_px_x, wrist_px_y), 5, color, -1)
        draw_wrist_frame(img_bgr, wrist_px_x, wrist_px_y, R_smooth)

        rpy_raw = rotationMatrixToEulerAngles(R_smooth)
        
        return img_bgr, HandState(
            position=pos_rob,
            orientation=R_rob,
            pinch_dist=pinch_dist,
            is_pinched=self.pinch_state,
            bbox=[x,y,w_box,h_box],
            joints_3d=pred_joints_centered,
            fps=0,
            rpy=rpy_raw
        )

    def run(self):
        print("[INFO] Starting Clean Tracker...")
        cv2.namedWindow("DexTel Control", cv2.WINDOW_NORMAL)
        try:
            while True:
                t_start = time.time()
                img, state = self.process_frame()
                if img is None: break
                
                final_display = img
                fps = 1.0 / (time.time() - t_start)
                if state: 
                    draw_ui_overlay(final_display, state, f"FPS: {fps:.1f} | Hand Tracking", (0, 255, 0))
                
                cv2.imshow("DexTel Control", final_display)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.hand_cam.stop()
            cv2.destroyAllWindows()

def draw_wrist_frame(image, u, v, R, axis_len=60):
    origin = (u, v)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)] 
    for i in range(3):
        vec = R[:, i]
        end_pt = (int(u + vec[0] * axis_len), int(v + vec[1] * axis_len))
        cv2.line(image, origin, end_pt, colors[i], 3, cv2.LINE_AA)
    cv2.circle(image, origin, 5, (255, 255, 255), -1)

def draw_ui_overlay(image, state: HandState, status_text: str, status_color: tuple):
    h, w = image.shape[:2]
    overlay = image.copy()
    s = h / 480.0
    
    def si(v): return int(v * s)
    def sf(v): return v * s 
    
    bar_h = si(50)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), -1)
    
    if state:
        panel_w, panel_h = si(200), si(120)
        margin = si(20)
        cv2.rectangle(overlay, (margin, h - panel_h - margin), (margin + panel_w, h - margin), (20, 20, 20), -1)
        cv2.rectangle(overlay, (w - panel_w - margin, h - panel_h - margin), (w - margin, h - margin), (20, 20, 20), -1)

    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, "DexTel", (si(20), si(35)), font, sf(0.8), (255, 255, 255), si(1), cv2.LINE_AA)
    cv2.putText(image, status_text, (si(140), si(35)), font, sf(0.6), status_color, si(1), cv2.LINE_AA)
    
    if state:
        status = "GRIPPED" if state.is_pinched else "RELEASED"
        col = (0, 200, 100) if state.is_pinched else (200, 200, 200)
        
        grip_box_w, grip_box_h = si(130), si(30)
        grip_margin = si(20)
        
        box_tl = (w - grip_box_w - grip_margin, si(10))
        box_br = (w - grip_margin, si(10) + grip_box_h)
        
        cv2.rectangle(image, box_tl, box_br, col, -1)
        
        ts = cv2.getTextSize(status, font, sf(0.6), si(1))[0]
        text_x = box_tl[0] + (grip_box_w - ts[0]) // 2
        text_y = box_tl[1] + (grip_box_h + ts[1]) // 2
        cv2.putText(image, status, (text_x, text_y), font, sf(0.6), (0,0,0), si(1))
    
        info_y_base = h - si(120) - margin + si(30)
        line_step = si(25)
        
        cv2.putText(image, "POSITION", (margin + si(10), info_y_base), font, sf(0.5), (150, 150, 150), si(1))
        cv2.putText(image, f"X {state.position[0]:.3f}", (margin + si(10), info_y_base + line_step), font, sf(0.6), (255,255,255), si(1))
        cv2.putText(image, f"Y {state.position[1]:.3f}", (margin + si(10), info_y_base + line_step*2), font, sf(0.6), (255,255,255), si(1))
        cv2.putText(image, f"Z {state.position[2]:.3f}", (margin + si(10), info_y_base + line_step*3), font, sf(0.6), (255,255,255), si(1))
        
        bar_w = si(300)
        cx, cy = w // 2, h - si(30)
        
        cv2.line(image, (cx - bar_w//2, cy), (cx + bar_w//2, cy), (100,100,100), si(4)) 
        
        rmax = 0.15
        for thresh, c in [(PINCH_CLOSE_THRESH, (0,0,255)), (PINCH_OPEN_THRESH, (0,255,0))]:
            off = int((thresh/rmax) * bar_w)
            x_line = cx - bar_w//2 + off
            cv2.line(image, (x_line, cy - si(8)), (x_line, cy + si(8)), c, si(2))
            
        val_off = int((min(state.pinch_dist, rmax)/rmax) * bar_w)
        cv2.circle(image, (cx - bar_w//2 + val_off, cy), si(8), (0,255,255), -1)

        orient_x_base = w - si(200) - margin + si(10)
        r_deg = np.degrees(state.rpy)
        cv2.putText(image, "RAW ORIENTATION", (orient_x_base, info_y_base), font, sf(0.5), (150, 150, 150), si(1))
        cv2.putText(image, f"R {r_deg[0]:.0f}", (orient_x_base, info_y_base + line_step), font, sf(0.6), (255,255,255), si(1))
        cv2.putText(image, f"P {r_deg[1]:.0f}", (orient_x_base + si(60), info_y_base + line_step), font, sf(0.6), (255,255,255), si(1))
        cv2.putText(image, f"Y {r_deg[2]:.0f}", (orient_x_base + si(120), info_y_base + line_step), font, sf(0.6), (255,255,255), si(1))

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if not sy < 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrist-cam-ip", type=str, default=None, help="Robot IP for Wrist Camera")
    parser.add_argument("--hand-cam-serial", type=str, default=None, help="Serial Number for Hand Tracking Camera")
    args = parser.parse_args()
    
    RobustTracker(wrist_cam_ip=args.wrist_cam_ip, hand_cam_serial=args.hand_cam_serial).run()
