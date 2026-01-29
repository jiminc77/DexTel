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

# HaMeR Dependencies
from hamer.models import load_hamer, DEFAULT_CHECKPOINT

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
HAMER_CONFIDENCE_THRESH = 0.5
PINCH_CLOSE_THRESH = 0.08  # meters (7cm)
PINCH_OPEN_THRESH = 0.12   # meters (12cm)
WRIST_FRAME_SMOOTH_ALPHA = 0.6 # Lower = smoother, Higher = faster

@dataclass
class HandState:
    position: np.ndarray        # [x, y, z] in Camera Frame (meters)
    orientation: np.ndarray     # [3x3] Rotation Matrix (Wrist Frame)
    pinch_dist: float           # Distance between Thumb and Index Tips (meters)
    is_pinched: bool            # Binary State
    bbox: list                  # [x, y, w, h]
    joints_3d: np.ndarray       # [21, 3]
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
        
        # 1. Initialize RealSense
        self.init_realsense()
        
        # 2. Initialize MediaPipe (BBox Only)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0, # Fast
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3. Initialize HaMeR
        print("[INFO] Loading HaMeR Model...")
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device).eval()
        self.model = self.model.to(self.device).eval()
        # self.model = self.model.half() # FP16 disabled for stability
        
        # Stats
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).float()
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).float()
        self.faces = self.model.mano.faces.astype(np.int32)
        
        # State
        self.prev_box = None
        self.pinch_state = False # False=Open, True=Closed
        self.filter_pos = None
        self.filter_rot = None
        
    def init_realsense(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
        # Filters
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.temporal = rs.temporal_filter()
        
        # Intrinsics
        stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = stream.as_video_stream_profile().get_intrinsics()
        
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        
        if not color or not depth: return None, None, None
        
        # Filter Depth
        depth = self.spatial.process(depth)
        depth = self.temporal.process(depth)
        
        img_bgr = np.asanyarray(color.get_data())
        # Mirror Flip (User Experience)
        img_bgr = cv2.flip(img_bgr, 1)
        
        return img_bgr, depth, aligned_frames.get_depth_frame() # Return original depth frame object for query

    def get_mediapipe_box(self, img_rgb):
        h, w = img_rgb.shape[:2]
        results = self.mp_hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            self.prev_box = None
            return None
            
        # [FILTER] Right Hand Only
        # Since image is FLIPPED (Mirror Mode):
        # - Real Right Hand appears on the Right side of the screen.
        # - Geometrically, it looks like a Left Hand to MediaPipe.
        # Logic: Look for handedness "Left" OR bbox on the right half.
        
        target_idx = -1
        
        # Priority 1: Check Handedness Label
        for i, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            if label == "Left": # Flipped Right Hand
                target_idx = i
                break
        
        # Priority 2: If no "Left" found, pick the one on the right side of screen (center > w/2)
        if target_idx == -1:
            for i, lm in enumerate(results.multi_hand_landmarks):
                cx = lm.landmark[9].x * w # Middle MCP
                if cx > w * 0.4: # Slightly lenient
                    target_idx = i
                    break
                    
        if target_idx == -1:
            self.prev_box = None
            return None # No Right Hand found

        lm = results.multi_hand_landmarks[target_idx]
        
        # Calculate Box
        x_list = [pt.x * w for pt in lm.landmark]
        y_list = [pt.y * h for pt in lm.landmark]
        
        min_x, max_x = min(x_list), max(x_list)
        min_y, max_y = min(y_list), max(y_list)
        
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        box_size = max(max_x - min_x, max_y - min_y) * 1.5 # Scale factor
        
        # Temporal Smoothing
        if self.prev_box is not None:
            alpha = 0.7
            cx = self.prev_box[0] * alpha + cx * (1-alpha)
            cy = self.prev_box[1] * alpha + cy * (1-alpha)
            box_size = self.prev_box[2] * alpha + box_size * (1-alpha)
            
        self.prev_box = [cx, cy, box_size]
        
        # Convert to int rect
        s = int(box_size)
        x = int(cx - s/2)
        y = int(cy - s/2)
        
        # Square crop padding
        x = max(0, x)
        y = max(0, y)
        w_box = min(w - x, s)
        h_box = min(h - y, s)
        
        return [x, y, w_box, h_box], lm 

    def estimate_rigid_orientation(self, joints_3d):
        """
        Computes a stable orientation frame using the 'Rigid Triangle' of the hand.
        Origin: Wrist (0)
        Vec1: Wrist -> Index MCP (5)
        Vec2: Wrist -> Pinky MCP (17)
        These points are structurally rigid (knuckles), unlike fingertips.
        """
        wrist = joints_3d[0]
        index_mcp = joints_3d[5]
        pinky_mcp = joints_3d[17]
        
        # 1. Primary Vector (Wrist -> Index)
        v1 = index_mcp - wrist
        v1 /= np.linalg.norm(v1)
        
        # 2. Secondary Vector (Wrist -> Pinky)
        v2 = pinky_mcp - wrist
        v2 /= np.linalg.norm(v2)
        
        # 3. Normal Vector (Up/Down)
        z_vec = np.cross(v1, v2)
        norm_z = np.linalg.norm(z_vec)
        if norm_z < 1e-6: return np.eye(3)
        z_vec /= norm_z
        
        # 4. Approach Vector (Forward)
        x_vec = v1 # Approximate X
        y_vec = np.cross(z_vec, x_vec)
        y_vec /= np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, z_vec)
        x_vec /= np.linalg.norm(x_vec)
        
        # Remap for Robot Control (Z=Normal, X=Approach)
        middle_mcp = joints_3d[9]
        v_approach_raw = middle_mcp - wrist
        dist = np.dot(v_approach_raw, z_vec)
        v_approach = v_approach_raw - dist * z_vec
        v_approach /= (np.linalg.norm(v_approach) + 1e-9)
        v_binormal = np.cross(z_vec, v_approach)
        
        # [Approach, BiNormal, Normal]
        R = np.column_stack((v_approach, v_binormal, z_vec))
        return R

    def process_frame(self) -> HandState:
        t_now = time.time()
        img_bgr, depth_img, depth_frame_obj = self.get_frames()
        if img_bgr is None: return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # 1. Detection
        box_data = self.get_mediapipe_box(img_rgb)
        
        if not box_data:
            # Draw sleek "Searching" UI
            cv2.rectangle(img_bgr, (w//2 - 150, h//2 - 30), (w//2 + 150, h//2 + 30), (0, 0, 0), -1)
            cv2.putText(img_bgr, "SEARCHING RIGHT HAND...", (w//2 - 130, h//2 + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return img_bgr, None

        bbox, mp_lm = box_data
        x, y, w_box, h_box = bbox
        
        # 2. HaMeR Inference
        crop = img_rgb[y:y+h_box, x:x+w_box]
        if crop.size == 0: return img_bgr, None
        
        # Chirality Fix
        crop_input = cv2.flip(crop, 1)
        
        valid_hamer = False
        try:
            # Preprocess (FP32)
            _inp = cv2.resize(crop_input, (256, 256))
            _inp = torch.from_numpy(_inp).float().to(self.device) / 255.0
            _inp = _inp.permute(2, 0, 1).unsqueeze(0) # [H, W, C] -> [1, C, H, W]
            _inp = (_inp - self.mean) / self.std
            
            with torch.no_grad():
                out = self.model({'img': _inp})
                
            pred_verts = out['pred_vertices'][0].cpu().numpy()
            pred_cam = out['pred_cam'][0].cpu().numpy()
            pred_joints = out['pred_keypoints_3d'][0].cpu().numpy()
            valid_hamer = True
        except Exception as e:
            print(f"HaMeR Error: {e}")
            
        if valid_hamer:
            # 3. Process 3D Data
            pred_joints[:, 0] *= -1
            pred_verts[:, 0] *= -1
            
            # --- Depth Anchoring ---
            wrist_px_x = int(mp_lm.landmark[0].x * w)
            wrist_px_y = int(mp_lm.landmark[0].y * h)
            
            d_list = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    d = depth_frame_obj.get_distance(wrist_px_x+dx, wrist_px_y+dy)
                    if d > 0: d_list.append(d)
            z_wrist_m = np.median(d_list) if d_list else 0.5
            
            # Center joints
            wrist_local = pred_joints[0].copy()
            joints_centered = pred_joints - wrist_local
            
            # 4. Orientation
            R = self.estimate_rigid_orientation(joints_centered)
            
            if self.filter_rot is None:
                self.filter_rot = OneEuroFilter(R, t_now, min_cutoff=0.5, beta=0.05)
            R_smooth = self.filter_rot(t_now, R)
            
            # 5. Position
            wrist_pt_3d =  rs.rs2_deproject_pixel_to_point(self.intrinsics, [wrist_px_x, wrist_px_y], z_wrist_m)
            pos_3d = np.array(wrist_pt_3d)
            
            if self.filter_pos is None:
                self.filter_pos = OneEuroFilter(pos_3d, t_now, min_cutoff=1.0, beta=0.1)
            pos_smooth = self.filter_pos(t_now, pos_3d)
            
            # 6. Pinch Detection
            thumb_tip = joints_centered[4]
            index_tip = joints_centered[8]
            pinch_dist = np.linalg.norm(thumb_tip - index_tip)
            
            if self.pinch_state:
                if pinch_dist > PINCH_OPEN_THRESH: self.pinch_state = False
            else:
                if pinch_dist < PINCH_CLOSE_THRESH: self.pinch_state = True
                
                
            # --- Visualization ---
            draw_hand_mesh(img_bgr, pred_verts, self.faces, self.intrinsics, x, y, w_box, h_box)
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
        print("[INFO] Starting Robust Hybrid Tracker...")
        print("[INFO] Press 'q' to exit.")
        try:
            frame_cnt = 0
            while True:
                t_start = time.time()
                img, state = self.process_frame()
                if img is None: break
                
                fps = 1.0 / (time.time() - t_start)
                if state:
                    draw_ui_overlay(img, state, fps)
                
                cv2.imshow("DexTel Robust Tracker", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_cnt += 1
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


def draw_wrist_frame(image, u, v, R, axis_len=50):
    # R cols: [Approach, BiNormal, Normal]
    # Approach (Red), Normal (Green)
    # We are drawing in 2D Screen space (simple projection of direction)
    # Assuming weak perspective for direction vector viz
    
    origin = (u, v)
    
    # Project 3D vectors to 2D is tricky without full projection matrix logic on the rotation.
    # But we want to visualize the 3D frame.
    # Let's simple draw the projected vectors if we had them.
    # Simplified: Just draw unit vectors scaled.
    # Note: This is "Fake" 2D projection of the 3D rotation, strictly speaking incorrect but useful enough for debug.
    # Correct way: Add `axis_len` in 3D to wrist position, then project.
    
    # Already computed P_app, P_norm in previous code could use `rs2_project...`
    # But R here is purely rotation.
    # Let's skip complex projection here since we don't pass full pos/intrinsics easily here.
    # Just draw a circle for wrist.
    cv2.circle(image, origin, 5, (0, 255, 255), -1)

def draw_hand_mesh(image, vertices, faces, intrinsics, bx, by, bw, bh):
    # Vertices are in Crop Coordinates (Canonical).
    # Need to map back to Screen.
    # This is complex because HaMeR output is weak-perspective camera relative to crop center.
    # Simplified: Just draw BBox for now to keep it clean.
    cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (255, 255, 0), 2)
    # Add a cool label
    cv2.putText(image, "RIGHT HAND TARGET", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def draw_ui_overlay(image, state: HandState, fps: float):
    h, w = image.shape[:2]
    
    # Theme Colors
    c_bg = (20, 20, 30)
    c_acc = (0, 255, 255) # Cyan
    c_warn = (0, 0, 255) # Red
    c_safe = (0, 255, 0) # Green
    
    # Sidebar
    sidebar_w = 300
    overlay = image.copy()
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), c_bg, -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Header
    x_base = w - sidebar_w + 20
    y = 50
    cv2.putText(image, "DexTel CONTROL", (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_acc, 2)
    y += 40
    
    # FPS
    cv2.putText(image, f"FPS: {fps:.1f}", (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 50
    
    # Gripper State
    cv2.putText(image, "GRIPPER STATE", (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 30
    
    state_txt = "CLOSED" if state.is_pinched else "OPEN"
    c_st = c_safe if not state.is_pinched else c_warn
    
    # Draw State Box
    cv2.rectangle(image, (x_base, y), (x_base + 200, y + 50), c_st, -1)
    text_size = cv2.getTextSize(state_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    tx = x_base + (200 - text_size[0]) // 2
    ty = y + (50 + text_size[1]) // 2
    cv2.putText(image, state_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    y += 80
    
    # Pinch Metric Bar
    cv2.putText(image, "PINCH DISTANCE", (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 15
    
    # Bar Background
    bar_w = 240
    bar_h = 10
    cv2.rectangle(image, (x_base, y), (x_base + bar_w, y + bar_h), (50, 50, 50), -1)
    
    # Bar Fill (Max range approx 0.15m)
    fill_ratio = max(0, min(1.0, state.pinch_dist / 0.15))
    fill_w = int(bar_w * fill_ratio)
    cv2.rectangle(image, (x_base, y), (x_base + fill_w, y + bar_h), c_acc, -1)
    
    # Threshold Markers
    close_x = int(bar_w * (PINCH_CLOSE_THRESH / 0.15))
    open_x = int(bar_w * (PINCH_OPEN_THRESH / 0.15))
    cv2.line(image, (x_base + close_x, y - 5), (x_base + close_x, y + bar_h + 5), c_warn, 2)
    cv2.line(image, (x_base + open_x, y - 5), (x_base + open_x, y + bar_h + 5), c_safe, 2)
    
    y += 30
    cv2.putText(image, f"{state.pinch_dist*100:.1f} cm", (x_base + bar_w - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_acc, 1)
    
    # Position Info
    y += 40
    cv2.putText(image, "WRIST POSITION", (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 25
    pos_str = f"X: {state.position[0]:.3f}"
    cv2.putText(image, pos_str, (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y += 25
    pos_str = f"Y: {state.position[1]:.3f}"
    cv2.putText(image, pos_str, (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y += 25
    pos_str = f"Z: {state.position[2]:.3f}"
    cv2.putText(image, pos_str, (x_base, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


if __name__ == "__main__":
    tracker = RobustTracker()
    tracker.run()
