import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
import time
import os

from dextel.ur3_realsense_hamer import RobustTracker, draw_ui_overlay
from dextel.retargeting import RetargetingWrapper
from dextel.robot_interface import SimRobotInterface, RealRobotInterface

# Calibration States
STATE_WAITING = 0
STATE_CALIBRATING = 1
STATE_ACTIVE = 2

class DexTelNode(Node):
    def __init__(self):
        super().__init__('dextel_node')
        
        # Parameters
        self.declare_parameter('urdf_path', 'assets/ur3e_hande.urdf')
        self.declare_parameter('use_real', False)
        
        self.use_real = self.get_parameter('use_real').get_parameter_value().bool_value
        param_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        
        # Paths
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(param_path):
            urdf_path = os.path.join(pkg_dir, param_path)
        else:
            urdf_path = param_path
            
        # Visuals
        cv2.namedWindow("DexTel Control", cv2.WINDOW_NORMAL)
        
        # Robot Interface
        if self.use_real:
            self.get_logger().info("MODE: REAL ROBOT")
            self.robot = RealRobotInterface(self)
        else:
            self.get_logger().info("MODE: SIMULATION")
            self.robot = SimRobotInterface(self)

        # Retargeting
        # [base, shoulder_lift, elbow, wrist1, wrist2, wrist3]
        self.home_joints = np.deg2rad([0, -90, -90, -90, 90, 0])
        self.robot_home_pos = None
        self.robot_home_rot = None
        
        self.get_logger().info(f"Initializing Retargeting (URDF: {urdf_path})...")
        try:
            self.retargeting = RetargetingWrapper(urdf_path, self.home_joints)
            self.retargeting_enabled = True
        except Exception as e:
            self.get_logger().error(f"Retargeting Init Failed: {e}")
            self.retargeting_enabled = False

        # Vision
        self.get_logger().info("Initializing Vision Tracker...")
        self.tracker = RobustTracker()
            
        self.q_filtered = None
        self.alpha = 0.1

        # Control Loop
        self.timer = self.create_timer(1.0/30.0, self.control_loop)
        self.get_logger().info("DexTel Node Ready.")

        # State
        self.state = STATE_WAITING
        self.origin_hand_pos = None
        self.origin_hand_rot = None
        self.calib_start_time = 0.0
        self.calib_samples_pos = []
        self.calib_samples_rot = []
        
        self.movement_scale = 1.5 

    def control_loop(self):
        # 1. Init Robot Home (FK)
        if self.robot_home_pos is None and self.retargeting_enabled:
            pos, rot = self.retargeting.compute_fk(self.home_joints)
            self.robot_home_pos = pos
            self.robot_home_rot = rot

        # 2. Vision Update
        img, state = self.tracker.process_frame()
        if img is None: return

        # 3. Input Handling
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            rclpy.shutdown(); return
        elif key & 0xFF == ord('r'):
            self.handle_reset(state)

        # 4. State Logic (Waiting, Calibrating, Active)
        target_joints, ui_status, ui_color = self.process_state_logic(state)

        # 5. Publish to Robot
        gripper_val = self.get_gripper_val(state)
        
        if target_joints is not None:
            if isinstance(self.robot, SimRobotInterface):
                self.robot.publish_full_state(target_joints, gripper_val)
            else:
                self.robot.move_joints(target_joints)
                self.robot.move_gripper(gripper_val)

        # 6. UI Update
        if state or img is not None:
            try:
                draw_ui_overlay(img, state, ui_status, ui_color)
            except: pass
        cv2.imshow("DexTel Control", img)

    def handle_reset(self, state):
        if state is not None:
            self.state = STATE_CALIBRATING
            self.calib_start_time = time.time()
            self.calib_samples_pos = []
            self.calib_samples_rot = []
            self.q_filtered = None 
            self.get_logger().info("Starting Calibration (2s)...")
        else:
            self.state = STATE_WAITING
            self.q_filtered = None 
            self.get_logger().info("Reset to WAITING.")

    def get_gripper_val(self, state):
        if state and state.is_pinched:
             return 0.0 # Closed
        return 0.025 # Open (Sim value, Real might need 0-255 or 0-1)

    def process_state_logic(self, state):
        target_q = None
        status = "WAITING"
        color = (100, 100, 100)

        if not self.retargeting_enabled:
            return None, "NO IK", (0, 0, 255)

        if self.state == STATE_WAITING:
            target_q = self.home_joints
            self.q_filtered = target_q
            status = "WAITING (Press R)"
            color = (0, 165, 255)

        elif self.state == STATE_CALIBRATING:
            target_q = self.home_joints
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
                    self.origin_hand_pos = np.mean(self.calib_samples_pos, axis=0)
                    self.origin_hand_rot = self.calib_samples_rot[-1] 
                    self.retargeting.reset_state(self.home_joints)
                    self.state = STATE_ACTIVE
                    self.get_logger().info("Calibration Done.")
                else:
                    self.state = STATE_WAITING
                    self.get_logger().warn("Calibration Failed.")

        elif self.state == STATE_ACTIVE:
            status = "ACTIVE"
            color = (0, 255, 0)
            
            if state:
                diff_pos = state.position - self.origin_hand_pos     
                target_pos = self.robot_home_pos + (diff_pos * self.movement_scale)
                
                # Delta Rotation
                R_delta = state.orientation @ self.origin_hand_rot.T
                target_rot = R_delta @ self.robot_home_rot
                
                q_raw = self.retargeting.solve(target_pos, target_rot)
                
                if q_raw.shape[0] > 6: q_raw = q_raw[:6]
                if np.isnan(q_raw).any(): q_raw = np.zeros(6)
                
                # Safety Check (Base Flip)
                if abs(q_raw[0] - self.home_joints[0]) > 2.0:
                    self.get_logger().warn("[SAFETY] Base Flip! Holding.")
                    q_raw = self.q_filtered if self.q_filtered is not None else self.home_joints
                    self.retargeting.reset_state(q_raw)
                
                if self.q_filtered is None: self.q_filtered = q_raw
                else: self.q_filtered = self.alpha * q_raw + (1.0 - self.alpha) * self.q_filtered
                
                target_q = self.q_filtered
            else:
                target_q = self.q_filtered if self.q_filtered is not None else self.home_joints

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
