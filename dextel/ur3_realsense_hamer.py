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
            
        # We assume Single Hand (Right Hand of User -> appears on Right side of Mirrored Screen?)
        # Actually in Mirror Mode:
        # User raises RIGHT hand.
        # Image shows hand on RIGHT side.
        # MediaPipe sees localized hand.
        
        lm = results.multi_hand_landmarks[0]
        
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
        
        # 1. Primary Vector (Wrist -> Index) - Serves as rough 'Forward' or 'Side'
        v1 = index_mcp - wrist
        v1 /= np.linalg.norm(v1)
        
        # 2. Secondary Vector (Wrist -> Pinky)
        v2 = pinky_mcp - wrist
        v2 /= np.linalg.norm(v2)
        
        # 3. Normal Vector (Up/Down) - Cross product defines the palm plane normal
        # Right hand: Index(5) is right, Pinky(17) is left. 
        # Cross(Index-Wrist, Pinky-Wrist) -> Points DOWN (into palm) or UP?
        # Let's verify standard Right Hand Rule.
        # Index (Thumb side approx) x Pinky -> Normal points out of back of hand?
        # Testing needed. Let's define:
        # Z_local = Normalize(Cross(v1, v2)) -> 'Normal'
        z_vec = np.cross(v1, v2)
        norm_z = np.linalg.norm(z_vec)
        if norm_z < 1e-6: return np.eye(3)
        z_vec /= norm_z
        
        # 4. Approach Vector (Forward) - Direction fingers point
        # Project 'Middle Finger' direction onto the plane defined by Normal
        # Or simpler: Cross(Normal, Wrist->Index) ? No.
        # Let's Define X_local as Wrist->Index (approx)
        # Y_local = Cross(Z, X)
        # Then orthonormalize.
        
        x_vec = v1 # Approximate X
        y_vec = np.cross(z_vec, x_vec)
        y_vec /= np.linalg.norm(y_vec)
        
        # Recalculate X to ensure orthogonality
        x_vec = np.cross(y_vec, z_vec)
        x_vec /= np.linalg.norm(x_vec)
        
        # Output Matrix: Columns [X, Y, Z] or [Normal, Approach, ...]?
        # User wants: "Hand Open/Close", "Wrist Rotate".
        # Let's standardize:
        # Z = Normal (Green Arrow)
        # X = Approach (Red Arrow, pointing out of fingers)
        
        # Re-mapping to requested viz style:
        # Normal (Green) = Back of hand normal = z_vec
        # Approach (Red) = Towards fingers.
        # Current X_vec is Wrist->Index. Middle finger is a bit more 'centered'.
        
        middle_mcp = joints_3d[9]
        v_approach_raw = middle_mcp - wrist
        
        # Project v_approach_raw onto plane perpendicular to Normal (Z)
        dist = np.dot(v_approach_raw, z_vec)
        v_approach = v_approach_raw - dist * z_vec
        v_approach /= (np.linalg.norm(v_approach) + 1e-9)
        
        # Recalculate Bi-Normal
        v_binormal = np.cross(z_vec, v_approach)
        
        # Rotation Matrix [Approach, BiNormal, Normal]
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
            cv2.putText(img_bgr, "No Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return img_bgr, None

        bbox, mp_lm = box_data
        x, y, w_box, h_box = bbox
        
        # 2. HaMeR Inference
        # Crop
        crop = img_rgb[y:y+h_box, x:x+w_box]
        if crop.size == 0: return img_bgr, None
        
        # Chirality Fix: Result is Mirrored. HaMeR needs Right Hand.
        # Flip crop horizontally before inference -> Look like Right Hand.
        crop_input = cv2.flip(crop, 1)
        
        # Infer
        valid_hamer = False
        try:
            # Preprocess
            _inp = cv2.resize(crop_input, (256, 256))
            _inp = torch.from_numpy(_inp).float().to(self.device) / 255.0
            # _inp = _inp.half()
            _inp = _inp.permute(2, 0, 1).unsqueeze(0)
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
            # Un-Flip X coordinates because we flipped input
            pred_joints[:, 0] *= -1
            pred_verts[:, 0] *= -1
            
            # --- Depth Anchoring ---
            # Get Wrist Depth from RealSense
            # Map Wrist 3D to Pixel
            # We already have MP wrist pixel roughly.
            # But better to use the BBox center or similar?
            # Actually, we have the original image coordinates from MP.
            wrist_px_x = int(mp_lm.landmark[0].x * w)
            wrist_px_y = int(mp_lm.landmark[0].y * h)
            
            # Use kernel for robust depth
            d_list = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    d = depth_frame_obj.get_distance(wrist_px_x+dx, wrist_px_y+dy)
                    if d > 0: d_list.append(d)
            
            if d_list:
                z_wrist_m = np.median(d_list)
            else:
                z_wrist_m = 0.5 # Default fallback
            
            # Scale HaMeR output (which is normalized) to Metric
            # Heuristic: HaMeR canonical hand size is approx 0.2m?
            # Better: Use the fact that HaMeR outputs weak-perspective.
            # But simpler: Just place the WRIST at (0,0,0) in local, 
            # and offset by z_wrist_m.
            # The pred_joints are in 'camera frame' of the crop?
            # Actually pred_cam gives us (scale, trans_x, trans_y).
            # Let's just use the relative shape and anchor Wrist to RealSense Z.
            
            # Center joints on wrist
            wrist_local = pred_joints[0].copy()
            joints_centered = pred_joints - wrist_local
            
            # 4. Orientation (Rigid Triangle)
            R = self.estimate_rigid_orientation(joints_centered)
            
            # Smooth Orientation
            if self.filter_rot is None:
                self.filter_rot = OneEuroFilter(R, t_now, min_cutoff=0.5, beta=0.05)
            R_smooth = self.filter_rot(t_now, R)
            
            # 5. Position
            # We need X, Y in meters.
            # De-project wrist pixel
            wrist_pt_3d =  rs.rs2_deproject_pixel_to_point(self.intrinsics, [wrist_px_x, wrist_px_y], z_wrist_m)
            pos_3d = np.array(wrist_pt_3d)
            
            if self.filter_pos is None:
                self.filter_pos = OneEuroFilter(pos_3d, t_now, min_cutoff=1.0, beta=0.1)
            pos_smooth = self.filter_pos(t_now, pos_3d)
            
            # 6. Pinch Detection
            thumb_tip = joints_centered[4]
            index_tip = joints_centered[8]
            pinch_dist = np.linalg.norm(thumb_tip - index_tip)
            
            # Hysteresis
            if self.pinch_state: # Currently Closed
                if pinch_dist > PINCH_OPEN_THRESH: self.pinch_state = False
            else: # Currently Open
                if pinch_dist < PINCH_CLOSE_THRESH: self.pinch_state = True
                
            # --- Visualization ---
            # Draw Skeleton (Projected)
            # We need to project the 3D joints back to 2D image.
            # Since we have hybrid data, let's just project using the bounding box relative coords?
            # Or simplified: Draw MediaPipe skeleton but with HaMeR colors? 
            # User wants to debug robustness.
            # Let's draw the 3D Axes at the wrist.
            
            wrist_2d = (wrist_px_x, wrist_px_y)
            
            # Project Approach (Red) and Normal (Green)
            axis_len = 0.1 # 10cm
            
            # Approach
            pt_app = pos_smooth + R_smooth[:, 0] * axis_len
            px_app = rs.rs2_project_point_to_pixel(self.intrinsics, pt_app)
            cv2.arrowedLine(img_bgr, wrist_2d, (int(px_app[0]), int(px_app[1])), (0, 0, 255), 3) # Red
            
            # Normal
            pt_norm = pos_smooth + R_smooth[:, 2] * axis_len
            px_norm = rs.rs2_project_point_to_pixel(self.intrinsics, pt_norm)
            cv2.arrowedLine(img_bgr, wrist_2d, (int(px_norm[0]), int(px_norm[1])), (0, 255, 0), 3) # Green
            
            # Status Text
            color_status = (0, 0, 255) if self.pinch_state else (0, 255, 0)
            status_txt = "CLOSE" if self.pinch_state else "OPEN"
            
            cv2.putText(img_bgr, f"GRIPPER: {status_txt}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_status, 3)
            cv2.putText(img_bgr, f"Pinch Dist: {pinch_dist:.3f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_bgr, f"Z-Depth: {z_wrist_m:.3f}m", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw BBox
            cv2.rectangle(img_bgr, (x, y), (x+w_box, y+h_box), (255, 255, 0), 2)
            
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
            while True:
                img, state = self.process_frame()
                if img is None: break
                
                cv2.imshow("DexTel Robust Tracker", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RobustTracker()
    tracker.run()
