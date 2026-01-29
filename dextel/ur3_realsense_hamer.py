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
            max_num_hands=2,
            model_complexity=1, # Higher accuracy
            min_detection_confidence=0.4,
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
            
        # [FILTERING] Strict LEFT HAND Only
        # Mirror Logic: Real Left Hand -> Image Left Side -> MP Label "Right"
        
        target_idx = -1
        best_score = -1
        
        for i, handedness in enumerate(results.multi_handedness):
            score = handedness.classification[0].score
            label = handedness.classification[0].label
            
            # Check Spatial Position
            lm = results.multi_hand_landmarks[i]
            x_wrist = lm.landmark[0].x 
            is_left_side = x_wrist < 0.6 # Left side of screen (Physical Left)
            
            # Primary: Label "Right" (Physical Left)
            if label == "Right":
                if score > best_score:
                    best_score = score
                    target_idx = i
            
            # Secondary: If no label match, check spatial
            elif target_idx == -1 and is_left_side:
                 target_idx = i
        
        if target_idx == -1:
            # Fallback: Track closest to previous
            if self.prev_box is not None:
                min_dist = float('inf')
                prev_cx = self.prev_box[0] / w
                prev_cy = self.prev_box[1] / h
                
                for i, lm in enumerate(results.multi_hand_landmarks):
                    cx = lm.landmark[9].x
                    cy = lm.landmark[9].y
                    dist = (cx - prev_cx)**2 + (cy - prev_cy)**2
                    if dist < min_dist:
                        min_dist = dist
                        target_idx = i
            else:
                return None 

        if target_idx == -1: return None

        lm = results.multi_hand_landmarks[target_idx]
        
        # Calculate Box
        x_list = [pt.x * w for pt in lm.landmark]
        y_list = [pt.y * h for pt in lm.landmark]
        
        min_x, max_x = min(x_list), max(x_list)
        min_y, max_y = min(y_list), max(y_list)
        
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        box_size = max(max_x - min_x, max_y - min_y) * 1.5 
        
        if self.prev_box is not None:
            alpha = 0.6 # Smoother
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
        """
        Computes a stable orientation frame using the 'Rigid Triangle' of the hand.
        Origin: Wrist (0)
        Vec1: Wrist -> Index MCP (5)
        Vec2: Wrist -> Pinky MCP (17)
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
        
        # 3. Normal Vector (Up/Down) - Z axis
        z_vec = np.cross(v1, v2)
        norm_z = np.linalg.norm(z_vec)
        if norm_z < 1e-6: return np.eye(3)
        z_vec /= norm_z
        
        # 4. Approach Vector (Forward) - X axis
        # Use v1 (Wrist->Index) as rough X, then orthog against Z
        x_vec_raw = v1 
        y_vec = np.cross(z_vec, x_vec_raw)
        y_vec /= np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, z_vec) # Precise X
        
        # Re-Map for Robot End-Effector Convention
        # We want: 
        #   Z (Green) = Normal out of back of hand
        #   X (Red)   = Approach (Wrist -> Fingers)
        #   Y (Blue)  = Bi-normal (Side)
        
        # Currently:
        #   z_vec = Normal (Back of hand) -> Correct
        #   x_vec = Wrist->Index (Side-ish) -> We want Forward
        
        # Let's derive "Forward" from Middle Finger
        middle_mcp = joints_3d[9]
        v_forward = middle_mcp - wrist
        
        # Project onto plane normal to Z
        proj_forward = v_forward - np.dot(v_forward, z_vec) * z_vec
        proj_forward /= (np.linalg.norm(proj_forward) + 1e-9)
        
        final_x = proj_forward # Approach
        final_z = z_vec        # Normal
        final_y = np.cross(final_z, final_x) # Bi-normal
        
        # R = [X, Y, Z]
        R = np.column_stack((final_x, final_y, final_z))
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
            # Minimal "Searching" Indicator
            # cv2.putText(img_bgr, "Searching Left Hand...", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
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
            _inp = _inp.permute(2, 0, 1).unsqueeze(0)
            _inp = (_inp - self.mean) / self.std
            
            with torch.no_grad():
                out = self.model({'img': _inp})
                
            pred_verts = out['pred_vertices'][0].cpu().numpy()
            pred_cam = out['pred_cam'][0].cpu().numpy()
            pred_joints = out['pred_keypoints_3d'][0].cpu().numpy()
            valid_hamer = True
        except Exception as e:
            pass
            
        if valid_hamer:
            # 3. Process 3D Data
            pred_joints[:, 0] *= -1
            pred_verts[:, 0] *= -1
            
            # --- Depth Anchoring ---
            wrist_px_x = int(mp_lm.landmark[0].x * w)
            wrist_px_y = int(mp_lm.landmark[0].y * h)
            
            d_list = []
            pad = 2
            for dy in range(-pad, pad+1):
                for dx in range(-pad, pad+1):
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
    # R cols: [X, Y, Z] -> [Red, Blue, Green]
    origin = (u, v)
    
    # Simple 2D projection of 3D rotation columns
    # We assume 'y' in 3D is roughly 'y' on screen for visualization sake
    # (Not perfect but clear enough for orientation check)
    
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)] # BGR: Red, Blue, Green
    labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        # Project 3D vector to 2D
        # x_2d = x_3d, y_2d = -y_3d (image CS is y-down, 3D is y-up usually? No, camera frame y-down)
        # Just direct projection
        vec = R[:, i]
        end_pt = (int(u + vec[0] * axis_len), int(v + vec[1] * axis_len))
        
        cv2.line(image, origin, end_pt, colors[i], 3, cv2.LINE_AA)
        
    cv2.circle(image, origin, 6, (255, 255, 255), -1)
    cv2.circle(image, origin, 4, (0, 0, 0), 1)

def draw_hand_mesh(image, vertices, faces, intrinsics, bx, by, bw, bh):
    # Minimal BBox
    cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (255, 255, 255), 1)
    cv2.rectangle(image, (bx-1, by-1), (bx+bw+1, by+bh+1), (0, 0, 0), 1)

def draw_ui_overlay(image, state: HandState, fps: float):
    h, w = image.shape[:2]
    
    # --- Premium Minimalist UI ---
    
    # 1. Top Bar (Status)
    # Transparent black strip
    bar_h = 60
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Text
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Logo / Title
    cv2.putText(image, "DexTel", (20, 40), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    
    # FPS
    cv2.putText(image, f"{fps:.0f} FPS", (160, 40), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Gripper Status Indicator (Pill Shape)
    status_txt = "GRIPPED" if state.is_pinched else "RELEASED"
    status_color = (0, 200, 100) if state.is_pinched else (200, 200, 200) # Green vs Gray
    
    # Draw pill
    pill_x, pill_y, pill_w, pill_h = w - 180, 15, 160, 30
    cv2.rectangle(image, (pill_x, pill_y), (pill_x+pill_w, pill_y+pill_h), status_color, -1)
    
    # Centered Text
    txt_size = cv2.getTextSize(status_txt, font, 0.6, 1)[0]
    tx = pill_x + (pill_w - txt_size[0]) // 2
    ty = pill_y + (pill_h + txt_size[1]) // 2
    cv2.putText(image, status_txt, (tx, ty), font, 0.6, (0, 0, 0) if state.is_pinched else (50, 50, 50), 1, cv2.LINE_AA)
    
    # 2. Floating Info Panel (Bottom Left)
    panel_w, panel_h = 240, 120
    px, py = 20, h - 20 - panel_h
    
    overlay = image.copy()
    cv2.rectangle(overlay, (px, py), (px+panel_w, py+panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Position
    cv2.putText(image, "WRIST POSITION", (px+15, py+30), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(image, f"X {state.position[0]:.3f}", (px+15, py+55), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"Y {state.position[1]:.3f}", (px+15, py+80), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"Z {state.position[2]:.3f}", (px+15, py+105), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 3. Dynamic Pinch Bar (Bottom Center)
    kp_w = 400
    kx = (w - kp_w) // 2
    ky = h - 40
    
    # Track
    cv2.line(image, (kx, ky), (kx+kp_w, ky), (100, 100, 100), 4)
    
    # Threshold Ticks
    range_max = 0.15 # 15cm
    close_x = kx + int((PINCH_CLOSE_THRESH / range_max) * kp_w)
    open_x = kx + int((PINCH_OPEN_THRESH / range_max) * kp_w)
    
    cv2.line(image, (close_x, ky-10), (close_x, ky+10), (0, 0, 255), 2) # Close Mark
    cv2.line(image, (open_x, ky-10), (open_x, ky+10), (0, 255, 0), 2)   # Open Mark
    
    # Current Value (Circle)
    curr_x = kx + int((min(range_max, state.pinch_dist) / range_max) * kp_w)
    val_color = (0, 255, 255) # Cyan
    cv2.circle(image, (curr_x, ky), 8, val_color, -1)
    cv2.circle(image, (curr_x, ky), 10, (255, 255, 255), 2)

if __name__ == "__main__":
    tracker = RobustTracker()
    tracker.run()
