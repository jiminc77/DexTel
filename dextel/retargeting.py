import numpy as np
import os
import pinocchio as pin
from dex_retargeting.optimizer import VectorOptimizer
from typing import Optional

class RetargetingWrapper:
    def __init__(self, urdf_path: str):
        if not os.path.exists(urdf_path):
            print(f"[WARN] URDF not found at {urdf_path}. Retargeting will fail.")
            
        print(f"[INFO] Initializing Retargeting with URDF: {urdf_path}")
        
        self.optimizer = VectorOptimizer(
            urdf_path=urdf_path,
            target_link_names=["tool0", "tool0", "tool0"],
            target_vector_names=["+z", "+y", "position"],
            target_joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
        )
        
    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        v_approach = target_rot[:, 2] 
        v_normal = target_rot[:, 1]
        
        try:
            result = self.optimizer.retarget(
                target_pos=target_pos,
                target_vecs=[v_approach, v_normal]
            )
            return result.q
        except Exception as e:
            print(f"[ERR] Retargeting failed: {e}")
            return np.zeros(6)
