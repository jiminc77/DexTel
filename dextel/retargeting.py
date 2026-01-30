import numpy as np
import os
import pinocchio as pin
from dex_retargeting.optimizer import VectorOptimizer
from dex_retargeting.robot_wrapper import RobotWrapper
from typing import Optional

class RetargetingWrapper:
    def __init__(self, urdf_path, home_joints):
        if not os.path.exists(urdf_path):
            print(f"[WARN] URDF not found at {urdf_path}. Retargeting will fail.")
            
        print(f"[INFO] Initializing Retargeting with URDF: {urdf_path}")
        
        # Initialize Robot Wrapper
        robot = RobotWrapper(urdf_path)
        
        # --- Inject Virtual Frames (tool0_z, tool0_y) if missing ---
        model = robot.model
        if model.existFrame("tool0") and not model.existFrame("tool0_z"):
            print("[INFO] Injecting virtual orientation frames (tool0_z, tool0_y)...")
            tool0_id = model.getFrameId("tool0")
            parent_placement = model.frames[tool0_id].placement
            
            # tool0_z: 0.5m along Z (Use 0.5m for definition, but 0.1m for scaling in optimizer)
            d_z = pin.SE3.Identity(); d_z.translation = np.array([0.0, 0.0, 0.5])
            model.addFrame(pin.Frame("tool0_z", model.frames[tool0_id].parent, tool0_id, parent_placement * d_z, pin.FrameType.OP_FRAME))
            
            # tool0_y: 0.5m along Y
            d_y = pin.SE3.Identity(); d_y.translation = np.array([0.0, 0.5, 0.0])
            model.addFrame(pin.Frame("tool0_y", model.frames[tool0_id].parent, tool0_id, parent_placement * d_y, pin.FrameType.OP_FRAME))
            
            robot.data = model.createData()
            
        # Define links for vector optimization
        target_origin_link_names = ["ur3e_base_link", "tool0", "tool0"]
        target_task_link_names = ["tool0", "tool0_z", "tool0_y"]
        
        # 6 Movable Joints (Explicitly ignore gripper/fixed joints)
        target_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Optimizer Setup
        # dummy_indices maps [Pos, Z, Y] vectors strictly 1-to-1
        self.optimizer = VectorOptimizer(
            robot=robot,
            target_joint_names=target_joint_names,
            target_origin_link_names=target_origin_link_names,
            target_task_link_names=target_task_link_names,
            target_link_human_indices=np.array([0, 1, 2], dtype=int),
            scaling=1.0
        )
        
        # Clamp Base Joint to prevents flips (limit to ~ +/- 90 deg)
        robot.model.lowerPositionLimit[0] = -1.6 
        robot.model.upperPositionLimit[0] = 1.6
        
        # State Management
        self.last_q = np.array(home_joints) 
        self.filter = None
        
        # Vector Scale (Must match URDF/Frame offset)
        self.vector_scale = 0.1 
        
        # Identify Fixed Joints
        self.num_fixed = robot.model.nq - len(target_joint_names)
        self.fixed_qpos = np.zeros(self.num_fixed)
        print(f"[INFO] Config: {len(target_joint_names)} Opt Joints, {self.num_fixed} Fixed Joints.")
        
    def solve(self, target_pos, target_rot):
        if np.isnan(target_pos).any() or np.isnan(target_rot).any():
             return self.last_q

        # Construct Relative Target Vectors
        # 1. Pos, 2. Z-dir * scale, 3. Y-dir * scale
        target_vecs = np.vstack([
            target_pos,
            target_rot[:, 2] * self.vector_scale,
            target_rot[:, 1] * self.vector_scale
        ])
        
        try:
            # Direct Optimizer Call (6D State)
            result_q = self.optimizer.retarget(
                ref_value=target_vecs,
                fixed_qpos=self.fixed_qpos,
                last_qpos=self.last_q
            )
            self.last_q = result_q
            return result_q
            
        except Exception as e:
            print(f"[ERR] Retargeting failed: {e}")
            return self.last_q

    def compute_fk(self, q):
        """
        Computes the Forward Kinematics for the given joint configuration.
        Returns the position (xyz) and rotation matrix (3x3) of the end-effector (tool0).
        """
        model = self.optimizer.robot.model
        data = self.optimizer.robot.data
        
        # Check Expected Size
        if q.shape[0] != model.nq:
            # print(f"[DEBUG] Padding q from {q.shape[0]} to {model.nq}. Model Joints: {model.names}")
            # Pad with zeros (assuming arm joints are first)
            q_padded = np.zeros(model.nq)
            q_padded[:min(q.shape[0], model.nq)] = q
            q = q_padded
            
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # 'tool0' is the target frame used in optimization
        if self.optimizer.robot.model.existFrame("tool0"):
            frame_id = self.optimizer.robot.model.getFrameId("tool0")
            frame = self.optimizer.robot.data.oMf[frame_id]
            return frame.translation.copy(), frame.rotation.copy()
        else:
            print("[ERR] 'tool0' frame not found in URDF!")
            return np.zeros(3), np.eye(3)

    def reset_state(self, q: np.ndarray):
        """
        Resets the internal state (warm start) of the IK solver.
        """
        # q should be 6D (optimized joints only).
        self.last_q = q
        # Optional: Reset filter if it exists (Not implemented yet)
        print(f"[INFO] Retargeting State Reset to: {q[:6]}")
