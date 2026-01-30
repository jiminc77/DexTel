import numpy as np
import os
import pinocchio as pin
from dex_retargeting.optimizer import VectorOptimizer
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from typing import Optional

class RetargetingWrapper:
    def __init__(self, urdf_path: str):
        if not os.path.exists(urdf_path):
            print(f"[WARN] URDF not found at {urdf_path}. Retargeting will fail.")
            
        print(f"[INFO] Initializing Retargeting with URDF: {urdf_path}")
        
        # Initialize Robot Wrapper
        robot = RobotWrapper(urdf_path)
        
        # Define links for vector optimization
        # 1. Position: Base -> Tool0
        # 2. Orientation Z: Tool0 -> Tool0_Z (Offset 0.1m along Z)
        # 3. Orientation Y: Tool0 -> Tool0_Y (Offset 0.1m along Y)
        
        target_origin_link_names = ["ur3e_base_link", "tool0", "tool0"]
        target_task_link_names = ["tool0", "tool0_z", "tool0_y"]
        
        # Indices are not strictly used in the logic we rely on (direct retargeting), 
        # but required by init. Passing dummy indices.
        # VectorOptimizer expects shape (2, N_vectors)
        dummy_indices = np.zeros((2, 3), dtype=int)

        self.optimizer = VectorOptimizer(
            robot=robot,
            target_joint_names=robot.dof_joint_names,
            target_origin_link_names=target_origin_link_names,
            target_task_link_names=target_task_link_names,
            target_link_human_indices=dummy_indices,
            scaling=1.0
        )
        
        self.retargeting = SeqRetargeting(
            optimizer=self.optimizer,
            has_joint_limits=True
        )
        
        self.vector_scale = 0.1 # The offset distance used in URDF for tool0_z and tool0_y
        
    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        # target_rot columns are X, Y, Z axes
        v_approach = target_rot[:, 2] # Z axis
        v_normal = target_rot[:, 1]   # Y axis
        
        # Construct target vectors matching the URDF link pairs
        # 1. Position: target_pos
        # 2. Z vector: v_approach * scale
        # 3. Y vector: v_normal * scale
        
        if np.isnan(target_pos).any() or np.isnan(target_rot).any():
             print(f"[ERR] Retargeting Input contains NaNs! Pos: {target_pos} Rot: {target_rot}")
             return np.zeros(6)

        target_vecs = np.vstack([
            target_pos,
            v_approach * self.vector_scale,
            v_normal * self.vector_scale
        ])
        
        try:
            # SeqRetargeting.retarget expects a single ref_value array
            result_q = self.retargeting.retarget(
                ref_value=target_vecs
            )
            return result_q
        except Exception as e:
            print(f"[ERR] Retargeting failed: {e}")
            return np.zeros(6)

    def compute_fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Forward Kinematics for the given joint configuration.
        Returns the position (xyz) and rotation matrix (3x3) of the end-effector (tool0).
        """
        pin.forwardKinematics(self.optimizer.robot.model, self.optimizer.robot.data, q)
        pin.updateFramePlacements(self.optimizer.robot.model, self.optimizer.robot.data)
        
        # 'tool0' is the target frame used in optimization
        frame_id = self.optimizer.robot.model.getFrameId("tool0")
        frame = self.optimizer.robot.data.oMf[frame_id]
        
        return frame.translation.copy(), frame.rotation.copy()
