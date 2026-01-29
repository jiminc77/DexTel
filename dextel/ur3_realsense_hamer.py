import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
import math
import torch
import warnings
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import os
import hamer
from hamer.models import load_hamer, DEFAULT_CHECKPOINT

warnings.filterwarnings("ignore")

HAMER_CONFIDENCE_THRESH = 0.5

@dataclass
class HandState:
    position: np.ndarray
    orientation: np.ndarray
    pinch_dist: float
    is_pinched: bool
    bbox: list
    joints_3d: np.ndarray
    fps: float

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

class RobustTracker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using Device: {self.device}")
        
        self.init_realsense()
        
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

    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        profile = self.pipeline.start(config)
        stream = profile.get_stream(rs.stream.depth)
        self.intrinsics = stream.as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)
        
        self.spat_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
            
        filtered_depth = self.spat_filter.process(depth_frame)
        filtered_depth = self.temp_filter.process(filtered_depth)
        
        img = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(filtered_depth.get_data())
        
        return img, depth, filtered_depth.as_depth_frame()

    def get_mediapipe_box(self, img_rgb):
        h, w = img_rgb.shape[:2]
        img_flipped = cv2.flip(img_rgb, 1)
        results = self.mp_hands.process(img_flipped)
        
        if not results.multi_hand_landmarks:
            self.prev_box = None
            return None
            
        target_idx = -1
        
        for i, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            
            if label == "Left":
                target_idx = i
                break
                
        if target_idx == -1:
            self.prev_box = None
            return None 

        lm = results.multi_hand_landmarks[target_idx]
        
        for pt in lm.landmark:
            pt.x = 1.0 - pt.x
        
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
        x = int(cx - s/2)
        y = int(cy - s/2)
        x = max(0, x)
        y = max(0, y)
        w_box = min(w - x, s)
        h_box = min(h - y, s)
        
        return [x, y, w_box, h_box], lm

    def estimate_rigid_orientation(self, joints_3d):
        wrist = joints_3d[0]
        index_mcp = joints_3d[5]
        pinky_mcp = joints_3d[17]
        
        v1 = index_mcp - wrist
        v1 /= np.linalg.norm(v1)
        v2 = pinky_mcp - wrist
        v2 /= np.linalg.norm(v2)
        
        z_vec = np.cross(v1, v2)
        norm_z = np.linalg.norm(z_vec)
        if norm_z < 1e-6: return np.eye(3)
        z_vec /= norm_z
        
        x_vec_raw = v1 
        y_vec = np.cross(z_vec, x_vec_raw)
        y_vec /= np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, z_vec) 
        
        middle_mcp = joints_3d[9]
        v_forward = middle_mcp - wrist
        
        proj_forward = v_forward - np.dot(v_forward, z_vec) * z_vec
        proj_forward /= (np.linalg.norm(proj_forward) + 1e-9)
        
        final_x = proj_forward
        final_z = z_vec
        final_y = np.cross(final_z, final_x)
        
        return np.column_stack((final_x, final_y, final_z))

    def process_frame(self) -> HandState:
        t_now = time.time()
        img_bgr, depth_img, depth_frame_obj = self.get_frames()
        if img_bgr is None: return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        box_data = self.get_mediapipe_box(img_rgb)
        if not box_data: return img_bgr, None

        bbox, mp_lm = box_data
        x, y, w_box, h_box = bbox
        
        crop = img_rgb[y:y+h_box, x:x+w_box]
        if crop.size == 0: return img_bgr, None
        
        crop_input = cv2.flip(crop, 1)
        
        valid_hamer = False
        try:
            _inp = cv2.resize(crop_input, (256, 256))
            _inp = torch.from_numpy(_inp).float().to(self.device) / 255.0
            _inp = _inp.permute(2, 0, 1).unsqueeze(0)
            _inp = (_inp - self.mean) / self.std
            
            with torch.no_grad():
                out = self.model({'img': _inp})
                
            pred_joints = out['pred_keypoints_3d'][0].cpu().numpy()
            valid_hamer = True
        except:
            pass
            
        if valid_hamer:
            pred_joints[:, 0] *= -1
            
            wrist_px_x = int(mp_lm.landmark[0].x * w)
            wrist_px_y = int(mp_lm.landmark[0].y * h)
            
            d_list = []
            pad = 2
            for dy in range(-pad, pad+1):
                for dx in range(-pad, pad+1):
                    d = depth_frame_obj.get_distance(wrist_px_x+dx, wrist_px_y+dy)
                    if d > 0: d_list.append(d)
            z_wrist_m = np.median(d_list) if d_list else 0.5
            
            wrist_local = pred_joints[0].copy()
            joints_centered = pred_joints - wrist_local
            
            R = self.estimate_rigid_orientation(joints_centered)
            if self.filter_rot is None:
                self.filter_rot = OneEuroFilter(R, t_now, min_cutoff=0.5, beta=0.05)
            R_smooth = self.filter_rot(t_now, R)
            
            wrist_pt_3d =  rs.rs2_deproject_pixel_to_point(self.intrinsics, [wrist_px_x, wrist_px_y], z_wrist_m)
            pos_3d = np.array(wrist_pt_3d)
            
            if self.filter_pos is None:
                self.filter_pos = OneEuroFilter(pos_3d, t_now, min_cutoff=1.0, beta=0.1)
            pos_smooth = self.filter_pos(t_now, pos_3d)
            
            thumb_tip = joints_centered[4]
            index_tip = joints_centered[8]
            pinch_dist = np.linalg.norm(thumb_tip - index_tip)
            
            if self.pinch_state:
                if pinch_dist > PINCH_OPEN_THRESH: self.pinch_state = False
            else:
                if pinch_dist < PINCH_CLOSE_THRESH: self.pinch_state = True
                
            draw_wrist_frame(img_bgr, wrist_px_x, wrist_px_y, R_smooth)

            state = HandState(
                position=pos_smooth,
                orientation=R_smooth,
                pinch_dist=pinch_dist,
                is_pinched=self.pinch_state,
                bbox=[x,y,w_box,h_box],
                joints_3d=joints_centered,
                fps=0
            )
            return img_bgr, state
            
        return img_bgr, None

    def run(self):
        print("[INFO] Starting Clean Tracker...")
        try:
            while True:
                t_start = time.time()
                img, state = self.process_frame()
                if img is None: break
                
                fps = 1.0 / (time.time() - t_start)
                if state:
                    draw_ui_overlay(img, state, fps)
                
                cv2.imshow("DexTel Control", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def draw_wrist_frame(image, u, v, R, axis_len=60):
    origin = (u, v)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)] # RGB
    for i in range(3):
        vec = R[:, i]
        end_pt = (int(u + vec[0] * axis_len), int(v + vec[1] * axis_len))
        cv2.line(image, origin, end_pt, colors[i], 3, cv2.LINE_AA)
    cv2.circle(image, origin, 5, (255, 255, 255), -1)


def draw_ui_overlay(image, state: HandState, fps: float):
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # 1. Status Bar
    cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, "DexTel", (20, 35), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"{fps:.0f} FPS", (120, 35), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    
    status = "GRIPPED" if state.is_pinched else "RELEASED"
    col = (0, 200, 100) if state.is_pinched else (200, 200, 200)
    
    cv2.rectangle(image, (w-150, 10), (w-20, 40), col, -1)
    ts = cv2.getTextSize(status, font, 0.6, 1)[0]
    cv2.putText(image, status, (w-85-ts[0]//2, 25+ts[1]//2), font, 0.6, (0,0,0), 1)
    
    # 2. Info Panel
    cv2.rectangle(overlay, (20, h-140), (220, h-20), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    cv2.putText(image, "POSITION", (30, h-110), font, 0.5, (150, 150, 150), 1)
    cv2.putText(image, f"X {state.position[0]:.3f}", (30, h-85), font, 0.6, (255,255,255), 1)
    cv2.putText(image, f"Y {state.position[1]:.3f}", (30, h-60), font, 0.6, (255,255,255), 1)
    cv2.putText(image, f"Z {state.position[2]:.3f}", (30, h-35), font, 0.6, (255,255,255), 1)
    
    # 3. Pinch Bar
    cx, cy, cw = w//2, h-30, 300
    cv2.line(image, (cx-cw//2, cy), (cx+cw//2, cy), (100,100,100), 4) # Rail
    
    rmax = 0.15
    for thresh, col in [(PINCH_CLOSE_THRESH, (0,0,255)), (PINCH_OPEN_THRESH, (0,255,0))]:
        off = int((thresh/rmax)*cw)
        cv2.line(image, (cx-cw//2+off, cy-8), (cx-cw//2+off, cy+8), col, 2)
        
    val_off = int((min(state.pinch_dist, rmax)/rmax)*cw)
    cv2.circle(image, (cx-cw//2+val_off, cy), 8, (0,255,255), -1)

if __name__ == "__main__":
    RobustTracker().run()
