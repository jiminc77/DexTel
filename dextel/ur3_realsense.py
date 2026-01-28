import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import time
import math

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=np.float64)
        self.dx_prev = np.array(dx0, dtype=np.float64) if isinstance(dx0, (list, tuple, np.ndarray)) else np.full_like(self.x_prev, dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        
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

class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=60):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Depth: 848x480 @ 60 FPS
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
        # Color: 1280x720 @ 60 FPS
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
        self.spatial.set_option(rs.option.holes_fill, 0)
        
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

        # Apply Filters to Depth
        filtered_depth = self.spatial.process(depth_frame)
        filtered_depth = self.temporal.process(filtered_depth)
        
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
        if depth <= 0: return None
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
        return np.array(point)

    def stop(self):
        self.pipeline.stop()

class DeXHandDetector:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8):
        self.mp_hands = mp.solutions.hands
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )

    def detect_2d(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None
        
        # Only Right hand
        for i, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            if label == "Right":
                return results.multi_hand_landmarks[i], label
        
        return None, None

    def get_3d_landmarks(self, landmarks_2d, depth_frame):
        h, w = rs_cam.intrinsics.height, rs_cam.intrinsics.width
        points_3d = []
        
        for lm in landmarks_2d.landmark:
            u, v = lm.x * w, lm.y * h
            
            depth = rs_cam.get_pixel_depth(u, v, depth_frame, kernel_size=5) 
            
            if depth is None:
                return None
            
            point = rs_cam.deproject_pixel_to_point(u, v, depth)
            
            if point is None:
                return None
            
            points_3d.append(point)
        
        return np.array(points_3d)

    @staticmethod
    def estimate_wrist_frame(points_3d):
        if points_3d is None or points_3d.shape[0] < 21: return None, None, None
        
        wrist = points_3d[0]
        index_mcp = points_3d[5]
        pinky_mcp = points_3d[17]
        
        v_approach = index_mcp - wrist
        v_approach_norm = np.linalg.norm(v_approach)
        if v_approach_norm > 1e-6:
            v_approach /= v_approach_norm
        else:
            v_approach = np.array([0., 0., 1.])

        v_palm_span = pinky_mcp - index_mcp
        v_normal = np.cross(v_approach, v_palm_span)
        v_normal_norm = np.linalg.norm(v_normal)

        if v_normal_norm > 1e-6:
            v_normal /= v_normal_norm
        else:
            v_normal = np.array([0., 1., 0.])
            
        return wrist, v_approach, v_normal

    @staticmethod
    def draw_skeleton(image, landmarks_2d):
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            landmarks_2d,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )

def main():
    rs_cam = None 
    try:
        rs_cam = RealSenseCamera(fps=60)
        
        detector = DeXHandDetector()
        filter_3d = None
        
        print("\n[INFO] Starting Realsense Hand Tracking")
        print("[INFO] Press 'q' to exit.")

        while True:
            try:
                color_image, depth_frame, _ = rs_cam.get_frames()
            except RuntimeError as e:
                print(f"[ERROR] Camera stream failed: {e}")
                break 
                
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            landmarks_2d, label = detector.detect_2d(rgb_image)

            if landmarks_2d:
                points_3d = detector.get_3d_landmarks(landmarks_2d, depth_frame)
                
                if points_3d is not None:
                    current_time = time.time()
                    if filter_3d is None:
                        filter_3d = OneEuroFilter(current_time, points_3d, min_cutoff=0.1, beta=1.0, d_cutoff=1.0)
                    
                    points_3d_filtered = filter_3d(current_time, points_3d)
                    
                    wrist, v_approach, v_normal = detector.estimate_wrist_frame(points_3d_filtered)
                    
                    thumb_tip = points_3d_filtered[4]
                    index_tip = points_3d_filtered[8]
                    pinch_dist = np.linalg.norm(thumb_tip - index_tip)

                    h, w = color_image.shape[:2]
                    wrist_px = (int(landmarks_2d.landmark[0].x * w), int(landmarks_2d.landmark[0].y * h))
                    
                    detector.draw_skeleton(color_image, landmarks_2d)
                    
                    vec_len = 0.1
                    
                    # Approach (Blue)
                    p_approach = wrist + v_approach * vec_len
                    pixel_approach = rs.rs2_project_point_to_pixel(rs_cam.intrinsics, p_approach)
                    if not np.isnan(pixel_approach).any():
                        cv2.arrowedLine(color_image, wrist_px, (int(pixel_approach[0]), int(pixel_approach[1])), (255, 0, 0), 3)

                    # Normal (Green)
                    p_normal = wrist + v_normal * vec_len
                    pixel_normal = rs.rs2_project_point_to_pixel(rs_cam.intrinsics, p_normal)
                    if not np.isnan(pixel_normal).any():
                        cv2.arrowedLine(color_image, wrist_px, (int(pixel_normal[0]), int(pixel_normal[1])), (0, 255, 0), 3)

                    # Text Info
                    cv2.putText(color_image, f"Wrist Z: {wrist[2]:.3f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(color_image, f"Pinch: {pinch_dist*1000:.1f}mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(color_image, f"FPS: {rs_cam.profile.get_stream(rs.stream.color).fps}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                else:
                    cv2.putText(color_image, "Invalid Depth - Hand Tracking Paused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Realsense D455 Hand Tracking", color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        
    finally:
        if rs_cam:
            rs_cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
