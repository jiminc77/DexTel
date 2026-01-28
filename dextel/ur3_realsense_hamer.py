import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
import math
import threading
import queue
import warnings
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

HAMER_AVAILABLE = True
SYNC_MODE = True  # DEBUG: Run synchronously to verify HaMeR output
print("[INFO] Loading HaMeR dependencies...")
import torch
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
print("[INFO] HaMeR dependencies loaded successfully.")


@dataclass
class HandPose:
    position: np.ndarray
    approach: np.ndarray
    normal: np.ndarray
    joints_3d: Optional[np.ndarray] = None
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    bbox: Optional[list] = None
    timestamp: float = 0.0
    gripper_state: str = "OPEN"
    debug_crop: Optional[np.ndarray] = None  # For visual debugging


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t = float(t)
        x = np.array(x, dtype=np.float64)

        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

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


class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Depth: 848x480 @ 30 FPS (Standard D455 High FPS mode)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
        # Color: 1280x720 @ 30 FPS
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        print(f"[INFO] Initializing RealSense D455: Color {width}x{height}@{fps}, Depth 848x480@{fps}...")
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Spatial Filter
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 3)

        # Temporal Filter
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

        # Intrinsics
        try:
            color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        except Exception as e:
            raise RuntimeError(f"Failed to get intrinsics: {e}")

        print("[INFO] RealSense initialized successfully.")

        self.last_depth_quality = {"valid_pixels": 0, "total_pixels": 0, "median_depth": 0.0}

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError as e:
            raise RuntimeError(f"Timeout waiting for frames: {e}")

        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Received incomplete frames (missing depth or color).")

        filtered_depth = self.spatial.process(depth_frame)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = filtered_depth.as_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        return color_image, filtered_depth, color_frame

    def get_pixel_depth(self, u, v, depth_frame, kernel_size=5):
        depths = []
        half_k = kernel_size // 2

        w, h = self.intrinsics.width, self.intrinsics.height

        for dy in range(-half_k, half_k + 1):
            for dx in range(-half_k, half_k + 1):
                x, y = int(u + dx), int(v + dy)
                if 0 <= x < w and 0 <= y < h:
                    d = depth_frame.get_distance(x, y)
                    if d > 0:
                        depths.append(d)

        if not depths:
            return None

        return np.median(depths)

    def deproject_pixel_to_point(self, u, v, depth):
        if depth <= 0:
            return None
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
        return np.array(point)

    def get_depth_quality_metrics(self, depth_frame) -> Dict[str, float]:
        depth_image = np.asanyarray(depth_frame.get_data())

        valid_mask = depth_image > 0
        valid_pixels = np.sum(valid_mask)
        total_pixels = depth_image.size

        if valid_pixels > 0:
            median_depth = np.median(depth_image[valid_mask]) / 1000.0  # Convert to meters
        else:
            median_depth = 0.0

        self.last_depth_quality = {
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "median_depth": median_depth
        }

        return self.last_depth_quality

    def stop(self):
        self.pipeline.stop()


class MediaPipeBBoxDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=0
        )

        self.prev_bbox = None
        self.bbox_alpha = 0.7

    def detect_bbox(self, image_rgb) -> Optional[Tuple[list, float, Any]]:
        h, w, _ = image_rgb.shape
        results = self.hand_detector.process(image_rgb)

        if not results.multi_hand_landmarks:
            self.prev_bbox = None
            return None

        landmarks = results.multi_hand_landmarks[0]
        h, w, _ = image_rgb.shape
        results = self.hand_detector.process(image_rgb)

        if not results.multi_hand_landmarks:
            self.prev_bbox = None
            return None

        landmarks = results.multi_hand_landmarks[0]
        confidence = results.multi_handedness[0].classification[0].score

        # [NORMALIZATION] Use palm-based stable box logic from ur3_hamer.py
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]

        x0, y0 = wrist.x * w, wrist.y * h
        x9, y9 = middle_mcp.x * w, middle_mcp.y * h
        
        raw_cx, raw_cy = (x0 + x9) / 2, (y0 + y9) / 2
        palm_len = np.linalg.norm(np.array([x0, y0]) - np.array([x9, y9]))
        
        BOX_SCALE = 4.5
        raw_size = palm_len * BOX_SCALE

        if self.prev_bbox is not None:
            prev_cx, prev_cy, prev_size = self.prev_bbox
            S_FACTOR = 0.8
            curr_cx = prev_cx * S_FACTOR + raw_cx * (1 - S_FACTOR)
            curr_cy = prev_cy * S_FACTOR + raw_cy * (1 - S_FACTOR)
            curr_size = prev_size * S_FACTOR + raw_size * (1 - S_FACTOR)
        else:
            curr_cx, curr_cy, curr_size = raw_cx, raw_cy, raw_size

        # Store as [cx, cy, size] for smoothing next frame
        self.prev_bbox = [curr_cx, curr_cy, curr_size]

        # Convert to [x, y, w, h] for compatibility
        half_s = curr_size / 2
        x = int(curr_cx - half_s)
        y = int(curr_cy - half_s)
        size = int(curr_size)

        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        w_box = min(w - x, size)
        h_box = min(h - y, size)

        # Make square as much as possible, or just use w, h
        # HaMeR expects square-ish crops, so we try to provide square
        bbox = [x, y, w_box, h_box]

        return bbox, confidence, landmarks

    def reset(self):
        self.prev_bbox = None


class HaMeRInferenceEngine:
    def __init__(self, device='cuda', use_fp16=True):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and device == 'cuda'

        print(f"[INFO] Loading HaMeR model on {device}...")
        try:
            self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
            self.model = self.model.to(self.device)
            self.model.eval()

            if self.use_fp16:
                self.model = self.model.half()
                print("[INFO] HaMeR model loaded with FP16 precision.")
            else:
                print("[INFO] HaMeR model loaded with FP32 precision.")

            self.mean = torch.tensor(DEFAULT_MEAN, device=self.device).view(3, 1, 1).float()
            self.std = torch.tensor(DEFAULT_STD, device=self.device).view(3, 1, 1).float()

            if self.use_fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()

            self.faces = self.model.mano.faces.astype(np.int32)
            print("[INFO] HaMeR initialized successfully.")

        except Exception as e:
            raise RuntimeError(f"Failed to load HaMeR model: {e}")

    def _preprocess(self, image_crop: np.ndarray) -> torch.Tensor:
        img_resized = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # HWC -> CHW
        img_tensor = img_tensor.permute(2, 0, 1)

        # Normalize with ImageNet statistics
        img_tensor = (img_tensor - self.mean) / self.std

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        if self.use_fp16:
            img_tensor = img_tensor.half()

        return img_tensor

    def _postprocess(self, model_output: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        joints_3d = model_output['pred_keypoints_3d'][0].cpu().float().numpy()
        vertices = model_output['pred_vertices'][0].cpu().float().numpy()
        return {"joints": joints_3d, "vertices": vertices, "faces": self.faces}

    def infer(self, image_crop: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        try:
            img_tensor = self._preprocess(image_crop).to(self.device)

            with torch.no_grad():
                output = self.model({'img': img_tensor})

            return self._postprocess(output)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("[ERROR] CUDA OOM during HaMeR inference. Clearing cache...")
                torch.cuda.empty_cache()
                return None
            else:
                print(f"[ERROR] HaMeR inference failed: {e}")
                return None

        except Exception as e:
            print(f"[ERROR] Unexpected error in HaMeR inference: {e}")
            return None


class AsyncInferenceQueue:
    def __init__(self, inference_fn, max_queue_size=2):
        self.inference_fn = inference_fn
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.result_cache = {
            "data": None,
            "timestamp": 0.0,
            "error": None,
            "frame_id": -1
        }
        self.lock = threading.Lock()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        print("[INFO] AsyncInferenceQueue started.")

    def _worker(self):
        print("[DEBUG] Worker thread started.")
        while self.running:
            try:
                frame_crop, metadata = self.input_queue.get(timeout=1.0)
                # print(f"[DEBUG] Worker got frame {metadata.get('frame_id')}")

                try:
                    result = self.inference_fn(frame_crop)
                    # print("[DEBUG] Inference successful")

                    with self.lock:
                        self.result_cache = {
                            "data": result,
                            "timestamp": time.time(),
                            "error": None,
                            "frame_id": metadata.get("frame_id", -1)
                        }

                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    with self.lock:
                        self.result_cache["error"] = str(e)
                        self.result_cache["timestamp"] = time.time()

            except queue.Empty:
                continue

            except Exception as e:
                print(f"[ERROR] Worker thread loop error: {e}")

    def submit(self, frame_crop: np.ndarray, frame_id: int = -1) -> bool:
        try:
            metadata = {"frame_id": frame_id, "submit_time": time.time()}
            self.input_queue.put_nowait((frame_crop, metadata))
            return True
        except queue.Full:
            # print("[WARN] Async queue full, dropping frame")
            return False

    def get_latest(self) -> Dict[str, Any]:
        with self.lock:
            return self.result_cache.copy()

    def is_result_fresh(self, max_age_ms: float = 200.0) -> bool:
        with self.lock:
            if self.result_cache["data"] is None:
                return False

            age_ms = (time.time() - self.result_cache["timestamp"]) * 1000.0
            return age_ms < max_age_ms

    def stop(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        print("[INFO] AsyncInferenceQueue stopped.")

class RobustFrameEstimator:
    def __init__(self):
        self.prev_normal = None
        self.prev_approach = None

    @staticmethod
    def estimate_wrist_frame(joints_3d: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if joints_3d is None or joints_3d.shape[0] < 21:
            return None, None, None

        if np.isnan(joints_3d).any():
            return None, None, None

        palm_indices = [0, 5, 9, 13, 17]
        palm_points = joints_3d[palm_indices]

        if np.isnan(palm_points).any():
            return None, None, None

        centroid = np.mean(palm_points, axis=0)

        centered_points = palm_points - centroid
        distances = np.linalg.norm(centered_points, axis=1)
        std_dist = np.std(distances)

        if std_dist > 0:
            valid_mask = distances < (np.mean(distances) + 2 * std_dist)
            if np.sum(valid_mask) >= 3:
                palm_points_filtered = palm_points[valid_mask]
                centroid = np.mean(palm_points_filtered, axis=0)
                centered_points = palm_points_filtered - centroid
            else:
                centered_points = palm_points - centroid
        else:
            centered_points = palm_points - centroid

        try:
            u, s, vt = np.linalg.svd(centered_points)
        except np.linalg.LinAlgError:
            return None, None, None

        normal_vec = vt[-1]

        wrist = joints_3d[0]
        ref_vec = np.cross(joints_3d[5] - wrist, joints_3d[17] - wrist)

        if np.dot(normal_vec, ref_vec) > 0:
            normal_vec = -normal_vec

        norm_magnitude = np.linalg.norm(normal_vec)
        if norm_magnitude > 1e-6:
            v_normal = normal_vec / norm_magnitude
        else:
            v_normal = np.array([0., 1., 0.])

        v_approach = joints_3d[9] - joints_3d[0]
        v_approach_norm = np.linalg.norm(v_approach)

        if v_approach_norm > 1e-6:
            v_approach /= v_approach_norm
        else:
            v_approach = np.array([0., 0., 1.])

        v_approach = v_approach - np.dot(v_approach, v_normal) * v_normal
        v_approach_norm = np.linalg.norm(v_approach)

        if v_approach_norm > 1e-6:
            v_approach /= v_approach_norm
        else:
            v_approach = np.array([0., 0., 1.])

        return wrist, v_approach, v_normal

    def estimate_with_temporal_check(self, joints_3d: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        wrist, v_approach, v_normal = self.estimate_wrist_frame(joints_3d)

        if wrist is None:
            return None, None, None

        if self.prev_normal is not None and self.prev_approach is not None:
            normal_dot = np.dot(v_normal, self.prev_normal)
            approach_dot = np.dot(v_approach, self.prev_approach)

            if normal_dot < 0.7 or approach_dot < 0.7:
                # print(f"[WARN] Orientation flip detected (dot={normal_dot:.2f}), allowing update")
                pass

        self.prev_normal = v_normal.copy()
        self.prev_approach = v_approach.copy()

        return wrist, v_approach, v_normal

class PinchDetector:
    def __init__(self):
        self.calibrated_open_dist = None
        self.calibration_samples = []
        self.calibrating = False
        self.gripper_state = "OPEN"

    def start_calibration(self):
        self.calibration_samples = []
        self.calibrating = True
        print("[INFO] Pinch calibration started. Hold hand OPEN for 2 seconds...")

    def update_calibration(self, joints_3d: np.ndarray):
        if not self.calibrating:
            return

        thumb_tip = joints_3d[4]
        index_tip = joints_3d[8]
        dist = np.linalg.norm(thumb_tip - index_tip)

        self.calibration_samples.append(dist)

        if len(self.calibration_samples) >= 60:  # ~2 seconds at 30 FPS
            self.calibrated_open_dist = np.median(self.calibration_samples)
            self.calibrating = False
            print(f"[INFO] Pinch calibration complete. Open distance: {self.calibrated_open_dist:.4f}m")

    def detect(self, joints_3d: np.ndarray) -> str:
        if joints_3d is None or joints_3d.shape[0] < 21:
            return self.gripper_state

        if self.calibrating:
            self.update_calibration(joints_3d)
            return self.gripper_state

        if self.calibrated_open_dist is None:
            thumb_tip = joints_3d[4]
            index_tip = joints_3d[8]
            dist = np.linalg.norm(thumb_tip - index_tip)
            if self.gripper_state == "OPEN" and dist < 0.07:
                self.gripper_state = "CLOSE"
            elif self.gripper_state == "CLOSE" and dist > 0.12:
                self.gripper_state = "OPEN"

            return self.gripper_state

        thumb_tip = joints_3d[4]
        index_tip = joints_3d[8]
        dist = np.linalg.norm(thumb_tip - index_tip)

        threshold_close = self.calibrated_open_dist * 0.4
        threshold_open = self.calibrated_open_dist * 0.6

        if self.gripper_state == "OPEN" and dist < threshold_close:
            self.gripper_state = "CLOSE"
        elif self.gripper_state == "CLOSE" and dist > threshold_open:
            self.gripper_state = "OPEN"

        return self.gripper_state

class HybridHandPoseEstimator:
    def __init__(
        self,
        bbox_detector: MediaPipeBBoxDetector,
        rs_cam: RealSenseCamera,
        use_hamer: bool = True,
        hamer_engine: Optional['HaMeRInferenceEngine'] = None,
        async_queue: Optional[AsyncInferenceQueue] = None
    ):
        self.bbox_detector = bbox_detector
        self.rs_cam = rs_cam
        self.hamer_engine = hamer_engine
        self.async_queue = async_queue

        self.frame_estimator = RobustFrameEstimator()

        self.prev_pose = None
        self.hamer_submit_counter = 0
        self.hamer_interval = 6

        # OneEuroFilters for smoothing
        self.filter_joints = OneEuroFilter(min_cutoff=0.05, beta=2.0)
        self.filter_vertices = OneEuroFilter(min_cutoff=0.05, beta=2.0)

        self.stats = {
            "bbox_ms": 0.0,
            "hamer_ms": 0.0,
            "fusion_ms": 0.0,
            "hamer_submit_rate": 0.0,
            "cache_age_ms": 0.0,
            "error": None
        }

    def _fuse_hamer_with_depth(
        self,
        hamer_joints: np.ndarray,
        hamer_vertices: np.ndarray,
        wrist_depth_rs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        hamer_hand_size = np.linalg.norm(hamer_joints[9] - hamer_joints[0])

        if hamer_hand_size < 1e-6:
            return hamer_joints, hamer_vertices

        real_hand_size = 0.09
        scale = real_hand_size / hamer_hand_size

        scaled_joints = hamer_joints * scale
        scaled_vertices = hamer_vertices * scale

        depth_offset = wrist_depth_rs - scaled_joints[0, 2]
        scaled_joints[:, 2] += depth_offset
        scaled_vertices[:, 2] += depth_offset

        return scaled_joints, scaled_vertices

    def estimate_pose(
        self,
        image_rgb: np.ndarray,
        depth_frame,
        frame_id: int
    ) -> Optional[HandPose]:
        t_start = time.time()

        bbox_result = self.bbox_detector.detect_bbox(image_rgb)

        if bbox_result is None:
            self.bbox_detector.reset()
            return None

        bbox, confidence, landmarks_2d = bbox_result
        self.stats["bbox_ms"] = (time.time() - t_start) * 1000.0

        h, w = image_rgb.shape[:2]
        
        wrist_lm = landmarks_2d.landmark[0]
        wrist_px_x = int(wrist_lm.x * w)
        wrist_px_y = int(wrist_lm.y * h)
        
        wrist_px_x = max(0, min(w - 1, wrist_px_x))
        wrist_px_y = max(0, min(h - 1, wrist_px_y))

        wrist_depth_rs = self.rs_cam.get_pixel_depth(wrist_px_x, wrist_px_y, depth_frame, kernel_size=5)

        if wrist_depth_rs is None or wrist_depth_rs <= 0:
            return None

        # 3. HaMeR inference
        joints_3d = None
        vertices = None
        faces = None
        debug_crop = None
        t_hamer_start = time.time()

        if self.prev_pose is not None:
            prev_cx = self.prev_pose.bbox[0] + self.prev_pose.bbox[2] // 2
            prev_cy = self.prev_pose.bbox[1] + self.prev_pose.bbox[3] // 2
            curr_cx = bbox[0] + bbox[2] // 2
            curr_cy = bbox[1] + bbox[3] // 2
            
            motion = np.linalg.norm(
                np.array([curr_cx, curr_cy]) - np.array([prev_cx, prev_cy])
            )

            if motion > 50: self.hamer_interval = 2
            elif motion > 20: self.hamer_interval = 3
            else: self.hamer_interval = 4

        # Prepare Crop
        x, y, w_box, h_box = bbox
        crop = image_rgb[y:y+h_box, x:x+w_box].copy()
        
        # Save for debug visualization
        debug_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # [CRITICAL] Chirality Fix
        # HaMeR often works better on the "mirrored" version if the hand is right/left ambiguous or dataset bias.
        # We flip the input, then we must flip the output back.
        crop_flipped = cv2.flip(crop, 1)
        
        if SYNC_MODE:
            # Synchronous Inference (Main Thread)
            result_data = self.hamer_engine.infer(crop_flipped)
            if result_data is not None:
                joints_3d = result_data["joints"]
                vertices = result_data["vertices"]
                faces = result_data["faces"]
                self.stats["cache_age_ms"] = 0.0 # Fresh
                self.stats["error"] = None
                self.stats["hamer_submit_rate"] = 0.0 # N/A in sync
            else:
                 self.stats["error"] = "Sync Infer Failed"
        
        else:
            # Async Mode
            if self.hamer_submit_counter % self.hamer_interval == 0:
                self.async_queue.submit(crop_flipped, frame_id)
                self.stats["hamer_submit_rate"] = 30.0 / self.hamer_interval

            self.hamer_submit_counter += 1

            result = self.async_queue.get_latest()

            if result["data"] is not None and result["error"] is None:
                data = result["data"]
                joints_3d = data["joints"]
                vertices = data["vertices"]
                faces = data["faces"]
                self.stats["cache_age_ms"] = (time.time() - result["timestamp"]) * 1000.0
                self.stats["error"] = None
            elif result["error"] is not None:
                 self.stats["error"] = result["error"]
        
        self.stats["hamer_ms"] = (time.time() - t_hamer_start) * 1000.0

        t_fusion_start = time.time()

        if joints_3d is not None:
            # [CRITICAL] Un-Flip 3D X-coordinates
            # Since we flipped the input image, the predicted 3D structure is also flipped.
            # We negate the X component to restore it (assuming canonical centering).
            joints_3d = joints_3d.copy()
            vertices = vertices.copy()
            joints_3d[:, 0] *= -1
            vertices[:, 0] *= -1

            # [SMOOTHING] Apply OneEuroFilter
            t_curr = time.time()
            joints_3d = self.filter_joints(t_curr, joints_3d)
            vertices = self.filter_vertices(t_curr, vertices)

            scaled_joints, scaled_vertices = self._fuse_hamer_with_depth(joints_3d, vertices, wrist_depth_rs)
            wrist, v_approach, v_normal = self.frame_estimator.estimate_with_temporal_check(scaled_joints)

            if wrist is None:
                return None

            pose = HandPose(
                position=wrist,
                approach=v_approach,
                normal=v_normal,
                joints_3d=scaled_joints,
                vertices=scaled_vertices,
                faces=faces,
                bbox=bbox,
                timestamp=time.time(),
                debug_crop=debug_crop
            )
        else:
            return None

        self.stats["fusion_ms"] = (time.time() - t_fusion_start) * 1000.0

        self.prev_pose = pose
        return pose

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()


def draw_wrist_frame(image, wrist, v_approach, v_normal, intrinsics):
    vec_len = 0.1
    wrist_px = rs.rs2_project_point_to_pixel(intrinsics, wrist)

    if np.isnan(wrist_px).any(): return
    wrist_px = (int(wrist_px[0]), int(wrist_px[1]))

    p_approach = wrist + v_approach * vec_len
    pixel_approach = rs.rs2_project_point_to_pixel(intrinsics, p_approach)

    if not np.isnan(pixel_approach).any():
        px, py = int(pixel_approach[0]), int(pixel_approach[1])
        cv2.arrowedLine(image, wrist_px, (px, py), (255, 0, 0), 3, tipLength=0.3)

    p_normal = wrist + v_normal * vec_len
    pixel_normal = rs.rs2_project_point_to_pixel(intrinsics, p_normal)

    if not np.isnan(pixel_normal).any():
        px, py = int(pixel_normal[0]), int(pixel_normal[1])
        cv2.arrowedLine(image, wrist_px, (px, py), (0, 255, 0), 3, tipLength=0.3)

    cv2.circle(image, wrist_px, 5, (0, 255, 255), -1)


def draw_hand_mesh(image, vertices, faces, intrinsics):
    if vertices is None or faces is None: return
    
    pixels = []
    w, h = image.shape[1], image.shape[0]

    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    valid_mask = vertices[:, 2] > 0
    if not np.any(valid_mask): return

    v_z = vertices[:, 2]
    v_x = vertices[:, 0] / v_z * fx + cx
    v_y = vertices[:, 1] / v_z * fy + cy

    points = np.stack([v_x, v_y], axis=1).astype(np.int32)
    
    # Pre-allocate valid points
    pts_2d = points[valid_mask]
    
    # Draw vertices as dots
    for i in range(0, len(pts_2d), 2): # Draw every 2nd vertex
         cv2.circle(image, (pts_2d[i, 0], pts_2d[i, 1]), 1, (200, 200, 200), -1)
    for i in range(0, len(faces), 10):
        face = faces[i]
        pts = points[face]
        # Check bounds
        if np.any(pts < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] >= h): continue
        cv2.polylines(image, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_skeleton_2d(image, joints_3d, intrinsics):
    """Draw 2D skeleton projected from 3D joints."""
    if joints_3d is None: return

    # MediaPipe/HaMeR 21 joint topology
    # 0: Wrist
    # 1-4: Thumb
    # 5-8: Index
    # 9-12: Middle
    # 13-16: Ring
    # 17-20: Pinky
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)     # Pinky
    ]
    
    # Project all points
    points_2d = []
    for i in range(21):
        pt_3d = joints_3d[i]
        px = rs.rs2_project_point_to_pixel(intrinsics, pt_3d)
        points_2d.append((int(px[0]), int(px[1])))
    
    # Draw connections
    for start, end in connections:
        pt1 = points_2d[start]
        pt2 = points_2d[end]
        
        # Check bounds roughly
        if not (0 <= pt1[0] < 3000 and 0 <= pt1[1] < 3000): continue
        if not (0 <= pt2[0] < 3000 and 0 <= pt2[1] < 3000): continue

        cv2.line(image, pt1, pt2, (0, 100, 255), 1)
        
    # Draw joints
    for pt in points_2d:
        cv2.circle(image, pt, 2, (0, 0, 255), -1)


def draw_ui_overlay(
    image: np.ndarray,
    stats: Dict[str, float],
    fps: float,
    gripper_state: str,
    wrist_depth: float,
    debug_crop: Optional[np.ndarray] = None
) -> None:
    h, w = image.shape[:2]
    
    panel_w, panel_h = 320, 160
    x_start = w - panel_w - 10
    y_start = 10
    
    overlay = image.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (0, 0, 0), -1)
    
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_text = (255, 255, 255)
    color_val = (0, 255, 255)
    
    cv2.putText(image, "FPS:", (x_start + 10, y_start + 30), font, 0.6, color_text, 1)
    cv2.putText(image, f"{fps:.1f}", (x_start + 70, y_start + 30), font, 0.6, color_val, 2)
    
    cv2.putText(image, "GPU (HaMeR)", (x_start + 130, y_start + 30), font, 0.5, (0, 255, 0), 1)

    cv2.putText(image, "Wrist Z:", (x_start + 10, y_start + 60), font, 0.6, color_text, 1)
    cv2.putText(image, f"{wrist_depth:.3f} m", (x_start + 90, y_start + 60), font, 0.6, color_val, 2)
    
    color_grip = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
    cv2.putText(image, "Gripper:", (x_start + 10, y_start + 90), font, 0.6, color_text, 1)
    cv2.putText(image, gripper_state, (x_start + 90, y_start + 90), font, 0.7, color_grip, 2)    # Timings
    cv2.putText(image, f"BBox: {stats['bbox_ms']:.1f}  HaMeR: {stats['hamer_ms']:.1f}", 
                (x_start + 10, y_start + 120), font, 0.45, (200, 200, 200), 1)
                
    # HaMeR Status
    age = stats['cache_age_ms']
    
    # Check for explicit error
    if stats.get("error"):
        error_msg = stats["error"]
        cv2.putText(image, f"ERR: {error_msg[:20]}", (x_start + 10, y_start + 140), font, 0.45, (0, 0, 255), 1)
    else:
        color_status = (0, 255, 0) if age < 500 else (0, 0, 255)
        status_text = "OK" if age < 500 else "LAG"
        
        cv2.putText(image, f"Rate: {stats['hamer_submit_rate']:.1f}  Age: {age:.0f}ms [{status_text}]", 
                    (x_start + 10, y_start + 140), font, 0.45, color_status, 1)

    # Draw Debug Crop if available
    if debug_crop is not None:
        try:
            # Resize fit to 100x100
            crop_viz = cv2.resize(debug_crop, (100, 100))
            h_crop, w_crop = crop_viz.shape[:2]
            
            # Bottom Right
            y_crop = h - h_crop - 10
            x_crop = w - w_crop - 10
            
            image[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop] = crop_viz
            cv2.rectangle(image, (x_crop, y_crop), (x_crop+w_crop, y_crop+h_crop), (255, 255, 0), 1)
            cv2.putText(image, "HaMeR Input", (x_crop, y_crop - 5), font, 0.4, (255, 255, 0), 1)
        except Exception:
            pass


def main():
    print("=" * 80)
    print("DexTel Production Hand Tracking System")
    print("Hybrid Pipeline: MediaPipe + HaMeR + RealSense D455")
    print("=" * 80)

    rs_cam = None
    async_queue = None

    try:
        print("\n[1/5] Initializing RealSense D455...")
        rs_cam = RealSenseCamera(fps=30)

        print("[2/5] Initializing MediaPipe BBox Detector...")
        bbox_detector = MediaPipeBBoxDetector()

        print("[3/5] Initializing HaMeR Inference Engine (GPU Only)...")
        try:
            hamer_engine = HaMeRInferenceEngine(device='cuda', use_fp16=False)
            async_queue = AsyncInferenceQueue(hamer_engine.infer, max_queue_size=2)
            print("[INFO] HaMeR mode ENABLED (RTX 5090 FP32)")
        except Exception as e:
            raise RuntimeError(f"FATAL: HaMeR initialization failed. GPU Required. Error: {e}")

        print("[4/5] Initializing Hybrid Hand Pose Estimator...")
        estimator = HybridHandPoseEstimator(
            bbox_detector=bbox_detector,
            rs_cam=rs_cam,
            use_hamer=True,
            hamer_engine=hamer_engine,
            async_queue=async_queue
        )

        # 5. Initialize Filters (TUNED FOR 3090/4090/5090 Responsiveness)
        print("[5/5] Initializing Filters and Pinch Detector...")
        t0 = time.time()
        # Beta increased for faster tracking
        filter_pos = OneEuroFilter(t0, np.zeros(3), min_cutoff=0.1, beta=0.5, d_cutoff=1.0)
        # Rotation needs to be VERY fast to feel responsive
        filter_app = OneEuroFilter(t0, np.zeros(3), min_cutoff=0.1, beta=2.0, d_cutoff=1.0)
        filter_norm = OneEuroFilter(t0, np.zeros(3), min_cutoff=0.1, beta=2.0, d_cutoff=1.0)
        filter_pinch = OneEuroFilter(t0, 0.0, min_cutoff=2.0, beta=1.0, d_cutoff=1.0)

        pinch_detector = PinchDetector()

        print("\n" + "=" * 80)
        print("System Ready! Controls:")
        print("  'q' - Quit")
        print("  'c' - Calibrate pinch detection (hold hand OPEN for 2 seconds)")
        print("=" * 80 + "\n")

        frame_count = 0

        while True:
            t_start = time.time()

            try:
                color_img, depth_frame, _ = rs_cam.get_frames()
            except RuntimeError as e:
                print(f"[ERROR] Camera stream failed: {e}")
                break

            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # rgb_img = cv2.flip(rgb_img, 1) # Disabled to fix chirality
            # color_img = cv2.flip(color_img, 1) # Disabled to fix chirality

            pose = estimator.estimate_pose(rgb_img, depth_frame, frame_count)

            if pose is not None:
                t_now = time.time()

                # Apply filters
                wrist = filter_pos(t_now, pose.position)
                v_app = filter_app(t_now, pose.approach)
                v_norm = filter_norm(t_now, pose.normal)

                # Detect pinch
                gripper_state = pinch_detector.detect(pose.joints_3d)

                # Visualize (Mesh enabled for debugging)
                draw_hand_mesh(color_img, pose.vertices, pose.faces, rs_cam.intrinsics)
                draw_skeleton_2d(color_img, pose.joints_3d, rs_cam.intrinsics) # Synced Skeleton
                draw_wrist_frame(color_img, wrist, v_app, v_norm, rs_cam.intrinsics)

                stats = estimator.get_stats()
                fps = 1.0 / (time.time() - t_start) if (time.time() - t_start) > 0 else 0
                draw_ui_overlay(color_img, stats, fps, gripper_state, wrist[2], pose.debug_crop)

            else:
                cv2.putText(color_img, "No Hand Detected", (450, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                fps = 1.0 / (time.time() - t_start) if (time.time() - t_start) > 0 else 0
                cv2.putText(color_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("DexTel - HaMeR Hybrid Hand Tracking", color_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested by user.")
                break
            elif key == ord('c'):
                pinch_detector.start_calibration()

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n[INFO] Cleaning up...")
        if async_queue is not None:
            async_queue.stop()
        if rs_cam is not None:
            rs_cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
