import cv2
import mediapipe as mp
import numpy as np
import time
import math

# =========================================
# OneEuroFilter Implementation
# =========================================
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

# =========================================
# SingleHandDetector (Right Hand Only)
# =========================================

OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

class SingleHandDetector:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.detected_hand_type = "Right"
        self.operator2mano = OPERATOR2MANO_RIGHT
        
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style="white"):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        mp_drawing_styles = mp.solutions.drawing_styles

        if style == "default":
            mp_drawing.draw_landmarks(
                image,
                keypoint_2d,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {}
            for landmark in mp_hands.HandLandmark:
                landmark_style[landmark] = mp_drawing.DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1
                )
            connection_style = {}
            for pair in mp_hands.HAND_CONNECTIONS:
                connection_style[pair] = mp_drawing.DrawingSpec(thickness=2)

            mp_drawing.draw_landmarks(
                image,
                keypoint_2d,
                mp_hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )
        return image

    def detect(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None
        
        desired_hand_num = -1
        for i, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        
        if desired_hand_num < 0:
            return 0, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        return num_box, keypoint_3d_array, keypoint_2d

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    def get_hand_vectors(self, kps_3d):
        wrist = kps_3d[0]
        index_mcp = kps_3d[5]
        pinky_mcp = kps_3d[17]
        
        # 1. Approach Vector (Wrist -> Index MCP)
        v_approach = index_mcp - wrist
        v_approach_norm = np.linalg.norm(v_approach)
        if v_approach_norm > 1e-6:
            v_approach /= v_approach_norm
        else:
            v_approach = np.array([0., 0., 1.])

        # 2. Palm Plane Vector (Index MCP -> Pinky MCP)
        v_palm_span = pinky_mcp - index_mcp
        
        # 3. Palm Normal Vector (Cross Product)
        v_normal = np.cross(v_approach, v_palm_span)
        v_normal_norm = np.linalg.norm(v_normal)
        if v_normal_norm > 1e-6:
            v_normal /= v_normal_norm
        else:
            v_normal = np.array([0., 1., 0.])

        return wrist, v_approach, v_normal

    def get_pinch_distance(self, kps_3d):
        thumb_tip = kps_3d[4]
        index_tip = kps_3d[8]
        return np.linalg.norm(thumb_tip - index_tip)

# =========================================
# Main Execution
# =========================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = SingleHandDetector()
    filter_landmarks = None
    
    print("Starting Right Hand Vector Demo...")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        num_box, kps_3d, kps_2d = detector.detect(rgb_frame)

        if kps_3d is not None:
            current_time = time.time()
            
            if filter_landmarks is None:
                filter_landmarks = OneEuroFilter(current_time, kps_3d, min_cutoff=1.0, beta=0.0)
            
            kps_3d_filtered = filter_landmarks(current_time, kps_3d)
            
            wrist, v_approach, v_normal = detector.get_hand_vectors(kps_3d_filtered)
            pinch_dist = detector.get_pinch_distance(kps_3d_filtered)
            
            SingleHandDetector.draw_skeleton_on_image(frame, kps_2d)
            
            h, w, _ = frame.shape
            wrist_2d = np.array([kps_2d.landmark[0].x * w, kps_2d.landmark[0].y * h]).astype(int)
            idx_mcp_2d = np.array([kps_2d.landmark[5].x * w, kps_2d.landmark[5].y * h]).astype(int)
            pinky_mcp_2d = np.array([kps_2d.landmark[17].x * w, kps_2d.landmark[17].y * h]).astype(int)

            cv2.putText(frame, f"Approach: [{v_approach[0]:.2f}, {v_approach[1]:.2f}, {v_approach[2]:.2f}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Normal:   [{v_normal[0]:.2f}, {v_normal[1]:.2f}, {v_normal[2]:.2f}]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Pinch Dist: {pinch_dist:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.arrowedLine(frame, tuple(wrist_2d), tuple(idx_mcp_2d), (255, 0, 0), 3) 
            cv2.arrowedLine(frame, tuple(idx_mcp_2d), tuple(pinky_mcp_2d), (0, 255, 255), 2) 
            cv2.circle(frame, tuple(wrist_2d), 5, (0, 0, 255), -1)

        cv2.imshow("Hand Vector Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()