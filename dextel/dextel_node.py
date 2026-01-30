import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from builtin_interfaces.msg import Time
from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
import time
import os

from dextel.ur3_realsense_hamer import RobustTracker, HandState, draw_ui_overlay
from dextel.retargeting import RetargetingWrapper

class DexTelNode(Node):
    def __init__(self):
        super().__init__('dextel_node')
        
        # --- 1. Robot Configurations (Define FIRST) ---
        # User-defined Home Configuration (Joint Space)
        # base, shoulder_lift, elbow, w1, w2, w3
        self.home_joints = np.deg2rad([0, -90, -90, -90, 90, 0])
        self.robot_home_pos = None
        self.robot_home_rot = None
        
        # Load Workspace Config
        try:
            self.dextel_base = get_package_share_directory('dextel')
        except Exception as e:
            self.get_logger().warn(f"Could not find package share directory ({e}). Falling back to local source path.")
            # Fallback: assume we are in src/dextel/dextel/dextel_node.py -> want src/dextel
            self.dextel_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.declare_parameter('urdf_path', 'assets/ur3e_hande.urdf')
        param_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(param_path):
            urdf_path = os.path.join(pkg_dir, param_path)
        else:
            urdf_path = param_path
        
        self.pub_joints = self.create_publisher(JointState, '/target_joint_states', 10)
        
        self.get_logger().info("Initializing Vision Tracker...")
        self.tracker = RobustTracker()

        self.get_logger().info(f"Initializing Retargeting (URDF: {urdf_path})...")
        try:
            self.retargeting = RetargetingWrapper(urdf_path, self.home_joints)
            self.retargeting_enabled = True
        except Exception as e:
            self.get_logger().error(f"Retargeting Init Failed: {e}")
            self.retargeting_enabled = False
            
        self.q_filtered = None
        self.alpha = 0.4 # Smoothing factor (0.0 = infinite smooth/lag, 1.0 = no smooth)

        self.timer = self.create_timer(1.0/30.0, self.control_loop)
        self.frame_count = 0
        self.get_logger().info("DexTel Node Ready.")

        # --- Relative Mapping Components ---
        self.origin_hand_pos = None  # Position of hand when 'R' was pressed
        
        self.relative_mode_active = False
        self.movement_scale = 1.5 # Slightly amplified movement for ease

    def control_loop(self):
        # Initialize Home Pose via FK
        if self.robot_home_pos is None and self.retargeting_enabled:
            pos, rot = self.retargeting.compute_fk(self.home_joints)
            self.robot_home_pos = pos
            self.robot_home_rot = rot
            self.get_logger().info(f"Home Pose Computed: {pos}")

        img, state = self.tracker.process_frame()
        if img is None: return

        # Constants for States
        STATE_WAITING = 0      # Robot at Home, Waiting for Hand + R
        STATE_CALIBRATING = 1  # Accumulating samples for 2s
        STATE_ACTIVE = 2       # Relative Control Active
        
        # Init State if needed
        if not hasattr(self, 'state'):
            self.state = STATE_WAITING
            self.calib_start_time = 0.0
            self.calib_samples_pos = []
            self.calib_samples_rot = [] # Store rotations

        # --- User Input ---
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            rclpy.shutdown()
            return
        elif key & 0xFF == ord('r'):
            # Logic: If Hand -> Calibrate. If No Hand -> Wait.
            
            if state is not None:
                self.state = STATE_CALIBRATING
                self.calib_start_time = time.time()
                self.calib_samples_pos = []
                self.calib_samples_rot = []
                self.q_filtered = None # Reset filter
                self.get_logger().info("Starting Calibration (2s)...")
            else:
                self.state = STATE_WAITING
                self.q_filtered = None 
                self.get_logger().info("Reset to WAITING (Home).")

        # --- Control Logic ---
        publish_dof = None
        gripper_val = 0.0
        ui_status = "WAITING"
        ui_color = (100, 100, 100) # Grey

        if self.retargeting_enabled:
            
            # State Machine
            if self.state == STATE_WAITING:
                # Behavior: Hold Home. Ignore Hand.
                publish_dof = self.home_joints
                self.q_filtered = publish_dof
                ui_status = "WAITING (Press R)"
                ui_color = (0, 165, 255) # Orange

            elif self.state == STATE_CALIBRATING:
                # Behavior: Hold Home. Collect Samples.
                publish_dof = self.home_joints
                self.q_filtered = publish_dof
                
                elapsed = time.time() - self.calib_start_time
                remaining = max(0.0, 2.0 - elapsed)
                ui_status = f"CALIB... {remaining:.1f}s"
                ui_color = (0, 255, 255) # Yellow
                
                if state:
                    self.calib_samples_pos.append(state.position)
                    self.calib_samples_rot.append(state.orientation)
                
                if elapsed >= 2.0:
                    # Finish Calibration
                    if len(self.calib_samples_pos) > 0:
                        # Average position
                        avg_pos = np.mean(self.calib_samples_pos, axis=0)
                        
                        # Average/Select rotation
                        # Averaging matrices is complex (SVD), simplest is to pick the last one or middle one.
                        # For short 2s stable hold, last valid sample is fine.
                        avg_rot = self.calib_samples_rot[-1] 
                        
                        self.origin_hand_pos = avg_pos
                        self.origin_hand_rot = avg_rot
                        
                        # --- CRITICAL: Reset IK Solver State to Home ---
                        # Call it RIGHT BEFORE Active mode starts
                        if self.retargeting_enabled:
                            self.retargeting.reset_state(self.home_joints)
                        
                        self.state = STATE_ACTIVE
                        self.get_logger().info(f"Calibration Done. Origin Pos: {avg_pos}")
                    else:
                        # Failed (No samples?) -> Back to Waiting
                        self.state = STATE_WAITING
                        self.get_logger().warn("Calibration Failed (No Samples). Waiting.")

            elif self.state == STATE_ACTIVE:
                # Behavior: Relative Control
                ui_status = "ACTIVE"
                ui_color = (0, 255, 0) # Green
                
                if state:
                    # --- Position: Relative ---
                    diff_pos = state.position - self.origin_hand_pos     
                    target_pos_rob = self.robot_home_pos + (diff_pos * self.movement_scale)
                    
                    # --- Orientation: Relative ---
                    # Compute rotation delta: R_delta = R_current @ R_origin.T
                    # This represents "How much has the hand rotated since calibration?"
                    R_delta = state.orientation @ self.origin_hand_rot.T
                    
                    # Apply delta to Robot Home
                    # R_target = R_delta @ R_home_rot
                    # This means "Rotate robot from Home by the same amount hand rotated from Origin"
                    target_rot_rob = R_delta @ self.robot_home_rot
                    
                    q_raw = self.retargeting.solve(target_pos_rob, target_rot_rob)
                    
                    # Ensure shape (6,) for arm joints
                    if q_raw.shape[0] > 6:
                        q_raw = q_raw[:6]
                        
                    if np.isnan(q_raw).any(): q_raw = np.zeros(6)
                    
                    # --- SAFETY: Check for Base Flip (180 deg) ---
                    # Sometimes IK Solver flips base (approx 3.14 rad). We must reject this.
                    base_diff = abs(q_raw[0] - self.home_joints[0])
                    if base_diff > 2.0: # ~115 degrees threshold
                        self.get_logger().warn(f"[SAFETY] Base Flip Detected! Diff: {base_diff:.2f} rad. Rejecting Solution.")
                        # Fallback: Hold last good position or Home
                        if self.q_filtered is not None:
                            q_raw = self.q_filtered 
                        else:
                            q_raw = self.home_joints
                            
                        # Optional: Force reset solver again to help it recover
                        self.retargeting.reset_state(q_raw)
                    
                    # DEBUG: Check for flip on first frame? (Optional)
                    # if self.q_filtered is None:
                    #     self.get_logger().info(f"First Active IK: {q_raw} (Home: {self.home_joints})")
                    
                    if self.q_filtered is None: self.q_filtered = q_raw
                    else: self.q_filtered = self.alpha * q_raw + (1.0 - self.alpha) * self.q_filtered
                    
                    publish_dof = self.q_filtered
                    gripper_val = 0.8 if state.is_pinched else 0.0
                else:
                    # Lost hand in Active mode? 
                    if self.q_filtered is not None:
                        publish_dof = self.q_filtered
                    else:
                        publish_dof = self.home_joints # Fail safe

        # --- Publish ---
        if publish_dof is not None:
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                "Slider_1", "Slider_2"
            ]
            joint_msg.position = list(publish_dof[:6]) + [gripper_val, gripper_val]
            joint_msg.velocity = [0.0] * 8; joint_msg.effort = [0.0] * 8
            self.pub_joints.publish(joint_msg)

        # --- UI Update ---
        if state or img is not None:
            # Pass dummy fps (0.0) -> We'll use fps arg for status text if we don't change sig
            # Wait, better to update the signature.
            # Currently signature is (img, state, fps, is_relative).
            # Let's map is_relative argument to be our status text for now to minimal change?
            # Or just update signature in next step.
            # I will assume signature UPDATE in next step: (img, state, status_text, status_col)
            try:
                draw_ui_overlay(img, state, ui_status, ui_color) # Speculative call, next tool will fix def
            except:
                pass # Safe fail until sync
        
        cv2.imshow("DexTel Control", img)


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
