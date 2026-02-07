import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import time
import os
import threading
from ament_index_python.packages import get_package_share_directory

from dextel.ur3_realsense_hamer import RobustTracker, draw_ui_overlay, RealsenseCamera
from dextel.retargeting import RetargetingWrapper
from dextel.robot_interface import SimRobotInterface, RealRobotInterface
from dextel.data_recorder import DataRecorder

# Constants
STATE_HOMING = -1
STATE_WAITING = 0
STATE_CALIBRATING = 1
STATE_ACTIVE = 2

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.alpha_correction = 0.1

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev 
        
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        
        return x_hat

class DexTelNode(Node):
    def __init__(self):
        super().__init__('dextel_node')

        # Camera Serials
        self.HAND_CAM_SERIAL = '308222301160' # USB 3.2 (High Bandwidth for HaMeR)
        self.GLOBAL_CAM_SERIAL = '318122303546' # USB 2.1 (Global View)

        self.joint_filters = None 
        self.filter_min_cutoff = 0.05   
        self.filter_beta = 0.005        
        
        self.declare_parameter('use_real', False)
        self.declare_parameter('urdf_path', 'assets/ur3e_hande.urdf')
        
        self.use_real = self.get_parameter('use_real').get_parameter_value().bool_value
        param_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(pkg_dir, param_path) if not os.path.isabs(param_path) else param_path
            
        cv2.namedWindow("DexTel Control", cv2.WINDOW_NORMAL)
        
        if self.use_real:
            self.get_logger().info("MODE: REAL ROBOT")
            self.robot = RealRobotInterface(self)
        else:
            self.get_logger().info("MODE: SIMULATION")
            self.robot = SimRobotInterface(self)
        
        # [base, shoulder_lift, elbow, wrist1, wrist2, wrist3]
        self.home_joints = np.deg2rad([90, -90, -90, -90, 90, 0])
        self.robot_home_pos = None
        self.robot_home_rot = None
        
        self.get_logger().info(f"Initializing Retargeting (URDF: {urdf_path})...")
        try:
            self.retargeting = RetargetingWrapper(urdf_path, self.home_joints)
            self.retargeting_enabled = True
        except Exception as e:
            self.get_logger().error(f"Retargeting Init Failed: {e}")
            self.retargeting_enabled = False

        self.get_logger().info(f"Initializing Vision Tracker...")
        self.tracker = RobustTracker(hand_cam_serial=self.HAND_CAM_SERIAL)
        
        self.get_logger().info(f"Initializing Global Camera (Serial: {self.GLOBAL_CAM_SERIAL})...")
        try:
             self.global_cam = RealsenseCamera(serial_number=self.GLOBAL_CAM_SERIAL)
             self.global_cam.start_async()
        except Exception as e:
             self.get_logger().error(f"Global Cam Init Failed: {e}")
             self.global_cam = None

        # Determine absolute path for data collection
        data_dir = os.path.join(pkg_dir, "../data_collection")
        self.recorder = DataRecorder(save_dir=data_dir)
        
        self.recording_state = "IDLE" # IDLE, COUNTDOWN_START, RECORDING, COUNTDOWN_STOP
        self.recording_trigger_time = 0.0
        
        self.q_filtered = None
        self.alpha = 0.15 

        self.timer = self.create_timer(1.0/60.0, self.control_loop)
        self.get_logger().info("DexTel Node Ready.")

        self.state = STATE_HOMING
        self.last_homing_cmd_time = 0.0
        
        self.origin_hand_pos = None
        self.origin_hand_rot = None
        self.calib_start_time = 0.0
        self.calib_samples_pos = []
        self.calib_samples_rot = []
        self.last_hand_seen_time = 0.0
        
        self.movement_scale = 1.5 
        
        self.lock = threading.Lock()
        self.latest_state = None
        self.latest_img = None
        self.latest_global_img = None
        self.latest_hand_img = None
        
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        self.control_fps_counter = 0
        self.last_control_fps_time = time.time()
        
        self.running = True
        self.vision_thread = threading.Thread(target=self.vision_loop)
        self.vision_thread.start()

    def vision_loop(self):
        while self.running:
            # 1. Hand Tracking only
            img, state = self.tracker.process_frame()
            
            # 2. Global Cam (Async)
            global_img = None
            if self.global_cam:
                # Non-blocking call now
                g_color, _, _ = self.global_cam.get_latest_frames()
                if g_color is not None:
                    global_img = cv2.cvtColor(g_color, cv2.COLOR_BGR2RGB)
            
            # FPS Calculation
            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                 fps = self.fps_counter / (now - self.last_fps_time)
                 print(f"[PERF] Vision Loop FPS: {fps:.1f}")
                 self.fps_counter = 0
                 self.last_fps_time = now
            
            display_list = []
            
            if img is not None:
                display_list.append(img)
            
            if global_img is not None:
                g_bgr = cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR)
                if img is not None and g_bgr.shape[:2] != img.shape[:2]:
                     h, w = img.shape[:2]
                     g_bgr_resized = cv2.resize(g_bgr, (w, h))
                     display_list.append(g_bgr_resized)
                else:
                     display_list.append(g_bgr)
            
            final_img = np.hstack(display_list) if display_list else None

            with self.lock:
                self.latest_img = final_img
                self.latest_state = state
                self.latest_global_img = global_img
                self.latest_hand_img = img 
            time.sleep(0.001)

    def control_loop(self):
        self.control_fps_counter += 1
        now = time.time()
        if now - self.last_control_fps_time >= 1.0:
            fps = self.control_fps_counter / (now - self.last_control_fps_time)
            self.control_fps_counter = 0
            self.last_control_fps_time = now

        state = None
        img = None
        with self.lock:
            state = self.latest_state
            img = self.latest_img # This is now the combined image
        
        if self.robot_home_pos is None and self.retargeting_enabled:
            pos, rot = self.retargeting.compute_fk(self.home_joints)
            self.robot_home_pos = pos
            self.robot_home_rot = rot

        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            self.running = False
            self.vision_thread.join()
            if self.global_cam: self.global_cam.stop()
            self.recorder.stop_recording()
            rclpy.shutdown()
            return
        elif key & 0xFF == ord('r'):
            self.handle_reset(state)
        elif key & 0xFF == ord('c'):
            self.handle_recording_toggle()


        target_joints, ui_status, ui_color = self.process_state_logic(state)
        
        remaining = 0.0
        if self.recording_state == "COUNTDOWN_START":
            remaining = 3.0 - (time.time() - self.recording_trigger_time)
            ui_status += f" | REC in {remaining:.1f}s"
            ui_color = (255, 165, 0) # Orange
            if remaining <= 0:
                self.recorder.start_recording()
                self.recording_state = "RECORDING"
        
        elif self.recording_state == "COUNTDOWN_STOP":
            remaining = 3.0 - (time.time() - self.recording_trigger_time)
            ui_status += f" | STOP in {remaining:.1f}s"
            ui_color = (255, 165, 0) # Orange
            pass
            if remaining <= 0:
                self.recorder.stop_recording()
                self.recording_state = "IDLE"

        elif self.recording_state == "RECORDING":
            ui_status += " | REC"
            ui_color = (0, 0, 255) # Red

        gripper_val = self.get_gripper_val(state)
        
        if target_joints is not None:
            max_vel = 1
            if self.state == STATE_HOMING:
                max_vel = 0.5
                now = time.time()
                if isinstance(self.robot, SimRobotInterface):
                   self.robot.publish_full_state(target_joints, gripper_val)
                else:
                   self.robot.move_joints(target_joints, max_vel=max_vel)
                   self.robot.move_gripper(gripper_val)
                self.last_homing_cmd_time = now
            else:
                if isinstance(self.robot, SimRobotInterface):
                    self.robot.publish_full_state(target_joints, gripper_val)
                else:
                    self.robot.move_joints(target_joints, max_vel=max_vel)
                    self.robot.move_gripper(gripper_val)
                    
            # --- DATA RECORDING ---
            is_recording_active = (self.recorder.recording) or (self.recording_state == "COUNTDOWN_STOP")
            
            if is_recording_active and target_joints is not None:
                curr_q = self.home_joints
                curr_vel = np.zeros(6)
                
                if isinstance(self.robot, RealRobotInterface):
                     real_q = self.robot.get_current_joints()
                     if real_q is not None: curr_q = np.array(real_q)
                
                record_images = {}
                with self.lock:
                     # Only record High cam now
                     if self.latest_global_img is not None:
                         record_images['cam_high'] = self.latest_global_img
                         
                action = np.append(target_joints, gripper_val) 
                
                # Append gripper command as proxy/feedback for qpos
                curr_q_ext = np.append(curr_q, gripper_val) 
                curr_vel_ext = np.append(curr_vel, 0.0)
                
                self.recorder.add_frame(curr_q_ext, curr_vel_ext, action, record_images)


        if img is not None:
            display_img = img.copy()
            if state:
                try:
                    status_text = ui_status
                    # Color handled by state machine above
                    draw_ui_overlay(display_img, state, status_text, ui_color)
                except Exception: pass
            cv2.imshow("DexTel Control", display_img)

    def handle_recording_toggle(self):
        if self.recording_state == "IDLE":
            self.recording_state = "COUNTDOWN_START"
            self.recording_trigger_time = time.time()
            self.get_logger().info("Recording Countdown Started (3s)...")
        elif self.recording_state == "RECORDING":
            self.recording_state = "COUNTDOWN_STOP"
            self.recording_trigger_time = time.time()
            self.get_logger().info("Stopping Countdown Started (3s)...")
        elif self.recording_state == "COUNTDOWN_START":
             self.recording_state = "IDLE" # Cancel start
             self.get_logger().info("Recording Trigger CANCELLED.")
        elif self.recording_state == "COUNTDOWN_STOP":
             self.recording_state = "RECORDING" # Cancel stop
             self.get_logger().info("Stop Trigger CANCELLED.")

    def handle_reset(self, state):
        if state is not None:
            self.state = STATE_CALIBRATING
            self.calib_start_time = time.time()
            self.calib_samples_pos = []
            self.calib_samples_rot = []
            self.get_logger().info("Starting Calibration (2s)...")
        else:
            self.state = STATE_WAITING
            self.q_filtered = None 
            self.get_logger().info("Reset to WAITING.")

    def get_gripper_val(self, state):
        if self.use_real:
            return 1.0 if (state and state.is_pinched) else 0.0
        else:
            return -0.025 if (state and state.is_pinched) else 0.0

    def process_state_logic(self, state):
        if not self.retargeting_enabled:
            return None, "NO IK", (0, 0, 255)

        if self.state == STATE_HOMING:
            return self._handle_homing_logic()
        
        elif self.state == STATE_WAITING:
            self.q_filtered = self.home_joints
            return None, "WAITING (Press R)", (0, 165, 255)

        elif self.state == STATE_CALIBRATING:
            return self._handle_calibrating_logic(state)

        elif self.state == STATE_ACTIVE:
            return self._handle_active_logic(state)
            
        return None, "UNKNOWN", (0,0,0)

    def _handle_homing_logic(self):
        target_q = self.home_joints
        status = "ROBOT HOMING..."
        color = (255, 0, 255)
        
        if isinstance(self.robot, RealRobotInterface):
            curr = self.robot.get_current_joints()
            if curr is not None:
                np_curr = np.array(curr)
                np_home = np.array(self.home_joints)
                diffs = np_curr - np_home
                diffs_wrapped = (diffs + np.pi) % (2 * np.pi) - np.pi
                max_diff = np.max(np.abs(diffs_wrapped))
                status = f"HOMING... Error: {max_diff:.2f}"
                
                if max_diff < 0.1:
                    self.state = STATE_WAITING
                    self.get_logger().info("Robot Homing Complete. Ready.")
            else:
                status = "HOMING... (No Feedback)"
        else:
            self.state = STATE_WAITING
        
        return target_q, status, color

    def _handle_calibrating_logic(self, state):
        target_q = self.q_filtered if self.q_filtered is not None else self.home_joints
        self.q_filtered = target_q
        elapsed = time.time() - self.calib_start_time
        remaining = max(0.0, 2.0 - elapsed)
        status = f"CALIB... {remaining:.1f}s"
        color = (0, 255, 255)
        
        if state:
            self.calib_samples_pos.append(state.position)
            self.calib_samples_rot.append(state.orientation)
        
        if elapsed >= 2.0:
            if len(self.calib_samples_pos) > 0:
                pos, rot = self.retargeting.compute_fk(target_q)
                self.origin_hand_pos = np.mean(self.calib_samples_pos, axis=0)
                self.origin_hand_rot = self.calib_samples_rot[-1]
                self.robot_home_pos = pos
                self.robot_home_rot = rot
                self.retargeting.reset_state(target_q)
                self.state = STATE_ACTIVE
                self.last_hand_seen_time = time.time()
                self.get_logger().info("Calibration Done.")
            else:
                self.state = STATE_WAITING
                self.get_logger().warn("Calibration Failed.")
        
        return target_q, status, color

    def _handle_active_logic(self, state):
        status = "ACTIVE"
        color = (0, 255, 0)
        target_q = None
        
        if state:
            self.last_hand_seen_time = time.time()
            diff_pos = state.position - self.origin_hand_pos     
            
            # --- XY 180 Degree Flip ---
            # If robot moves opposite to hand, use this flip.
            diff_pos[0] *= -1.0 # Invert Forward/Backward (X)
            diff_pos[1] *= -1.0 # Invert Left/Right (Y)
            
            target_pos = self.robot_home_pos + (diff_pos * self.movement_scale)
            
            R_delta = state.orientation @ self.origin_hand_rot.T
            target_rot = R_delta @ self.robot_home_rot
            
            q_raw = self.retargeting.solve(target_pos, target_rot)
            
            if q_raw.shape[0] > 6: q_raw = q_raw[:6]
            if np.isnan(q_raw).any(): q_raw = np.zeros(6)
            
            if abs(q_raw[0] - self.home_joints[0]) > 2.0:
                self.get_logger().warn("[SAFETY] Base Flip! Holding.")
                q_raw = self.q_filtered if self.q_filtered is not None else self.home_joints
                self.retargeting.reset_state(q_raw)
            
            now = time.time()
            if self.joint_filters is None:
                 self.q_filtered = q_raw
                 self.joint_filters = [
                     OneEuroFilter(now, q_raw[i], min_cutoff=self.filter_min_cutoff, beta=self.filter_beta) 
                     for i in range(6)
                 ]
            else:
                filtered_list = []
                for i in range(6):
                    filtered_list.append(self.joint_filters[i](now, q_raw[i]))
                self.q_filtered = np.array(filtered_list)
            
            target_q = self.q_filtered
        else:
            if time.time() - self.last_hand_seen_time > 3.0:
                status = "LOST -> RE-HOMING"
                self.state = STATE_HOMING
                self.last_homing_cmd_time = 0.0
            else:
                target_q = self.q_filtered if self.q_filtered is not None else self.home_joints
                status = "Hand Lost (Wait 3s...)"
        
        return target_q, status, color

def main(args=None):
    rclpy.init(args=args)
    node = DexTelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tracker.pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
