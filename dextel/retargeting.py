import numpy as np
import os
import pinocchio as pin
from dex_retargeting.optimizer import VectorOptimizer
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from typing import Optional

class RetargetingWrapper:
    def __init__(self, urdf_path, home_joints):
        if not os.path.exists(urdf_path):
            print(f"[WARN] URDF not found at {urdf_path}. Retargeting will fail.")
            
        print(f"[INFO] Initializing Retargeting with URDF: {urdf_path}")
        
        # Initialize Robot Wrapper
        robot = RobotWrapper(urdf_path)
        
        # --- DYNAMIC FRAMES INJECTION ---
        # If URDF is missing helper frames for orientation (tool0_z, tool0_y), add them now.
        model = robot.model
        if model.existFrame("tool0") and not model.existFrame("tool0_z"):
            print("[INFO] Injecting virtual orientation frames (tool0_z, tool0_y)...")
            tool0_id = model.getFrameId("tool0")
            tool0_frame = model.frames[tool0_id]
            parent_joint = tool0_frame.parent
            parent_placement = tool0_frame.placement # Transform from Joint to tool0
            
            # tool0_z: 0.5m along Z of tool0
            d_z = pin.SE3.Identity()
            d_z.translation = np.array([0.0, 0.0, 0.5])
            placement_z = parent_placement * d_z
            
            frame_z = pin.Frame("tool0_z", parent_joint, tool0_id, placement_z, pin.FrameType.OP_FRAME)
            model.addFrame(frame_z)
            
            # tool0_y: 0.5m along Y of tool0
            d_y = pin.SE3.Identity()
            d_y.translation = np.array([0.0, 0.5, 0.0])
            placement_y = parent_placement * d_y
            
            frame_y = pin.Frame("tool0_y", parent_joint, tool0_id, placement_y, pin.FrameType.OP_FRAME)
            model.addFrame(frame_y)
            
            # Re-create data to accommodate new frames
            robot.data = model.createData()
            
        
        # Define links for vector optimization
        # 1. Position: Base -> Tool0
        # 2. Orientation Z: Tool0 -> Tool0_Z
        # 3. Orientation Y: Tool0 -> Tool0_Y
        
        target_origin_link_names = ["ur3e_base_link", "tool0", "tool0"]
        target_task_link_names = ["tool0", "tool0_z", "tool0_y"]
        
        # Correctly map indices: 
        # Robot Vec 0 -> Human Vec 0 (Position)
        # Robot Vec 1 -> Human Vec 1 (Z-axis)
        # Robot Vec 2 -> Human Vec 2 (Y-axis)
        # Using 1D array to strictly map 1-to-1
        dummy_indices = np.array([0, 1, 2], dtype=int)
        
        # Explicitly define the 6 movable joints to avoid optimizing fixed joints
        target_joint_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint", 
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]

        self.optimizer = VectorOptimizer(
            robot=robot,
            target_joint_names=target_joint_names,
            target_origin_link_names=target_origin_link_names,
            target_task_link_names=target_task_link_names,
            target_link_human_indices=dummy_indices,
            scaling=1.0
        )
        
        # Add regularization to prevent twisting (redundant joints drifting)
        # Damping towards zero (or current pose in SeqRetargeting?)
        
        # --- CRITICAL: CLAMP BASE JOINT TO PREVENT FLIP ---
        # The UR3e base can rotate 360, but for teleop we want to stay "Front Facing"
        # Blocking regions > 90 deg preventing the "Back Flip" (2.31 rad) solution.
        model = robot.model
        # Index 0 is Base Pan
        model.lowerPositionLimit[0] = -1.6 # ~ -90 deg
        model.upperPositionLimit[0] = 1.6  # ~ +90 deg
        
        # Removed SeqRetargeting wrapper due to state mismanagement with Fixed Joints
        # We will manage warm-start state manually (Classic Control).
        self.last_q = np.array(home_joints) # Use provided home joints as initial seed
        self.filter = None # Initialize filter to None
        
        # Must match the offset distance of tool0_z/y in the URDF (0.1m)
        self.vector_scale = 0.1 
        
        # Identify Fixed Joints count
        # The optimizer expects us to provide values for non-optimized joints
        self.num_fixed = robot.model.nq - len(target_joint_names)
        self.fixed_qpos = np.zeros(self.num_fixed)
        print(f"[INFO] Retargeting Config: {len(target_joint_names)} Optimized Joints, {self.num_fixed} Fixed Joints.")
        
    def solve(self, target_pos, target_rot):
        # Input Validation
        if np.isnan(target_pos).any() or np.isnan(target_rot).any():
             print(f"[ERR] Retargeting Input contains NaNs!")
             return self.last_q

        # Construct Target Vectors (Relative)
        v_approach = target_rot[:, 2] # Z axis
        v_normal = target_rot[:, 1]   # Y axis
        
        target_vecs = np.vstack([
            target_pos,
            v_approach * self.vector_scale,
            v_normal * self.vector_scale
        ])
        
        try:
            # Direct Optimizer Call
            # We explicitly manage last_q (6D) to ensure correct warm start
            result_q = self.optimizer.retarget(
                ref_value=target_vecs,
                fixed_qpos=self.fixed_qpos,
                last_qpos=self.last_q
            )
            
            # Update state
            self.last_q = result_q
            
            # (Optional) Apply smoothing if filter exists. For now, raw output is safer than buggy filter.
            # Just ensure stability first.
            
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
        # compute_fk handles padding internally for Pinocchio.
             
        # Reset SeqRetargeting internal state
        if hasattr(self.retargeting, 'last_q'):
            self.retargeting.last_q = q
        
        # Perform a "Warm Up" solve to ensure optimizer internal state is synced
        # 1. Compute FK for the reset q
        pos, rot = self.compute_fk(q)
        
        # 2. Construct Target Vectors
        v_approach = rot[:, 2] # Z
        v_normal = rot[:, 1]   # Y
        
        target_vecs = np.vstack([
            pos,
            v_approach * self.vector_scale,
            v_normal * self.vector_scale
        ])
        
        # 3. Force Retarget
        try:
             # Force warm start with q
             # Note: SeqRetargeting.retarget might ignore warm_start if not explicitly supported in all versions,
             # but usually it uses self.last_q. We set self.last_q above.
             # We run it to update any other internal history.
            _ = self.retargeting.retarget(
                ref_value=target_vecs,
                fixed_qpos=self.fixed_qpos
            )
            
            # Re-enforce last_q just in case the solve moved it slightly (it shouldn't if q is exact solution)
            self.retargeting.last_q = q
        except Exception as e:
            print(f"[ERR] Reset State Warmup Failed: {e}")
