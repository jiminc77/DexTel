import numpy as np
import os
import pinocchio as pin
from dex_retargeting.optimizer import VectorOptimizer
from dex_retargeting.robot_wrapper import RobotWrapper

class RetargetingWrapper:
    def __init__(self, urdf_path, home_joints):
        if not os.path.exists(urdf_path):
            print(f"[WARN] URDF not found at {urdf_path}. Retargeting will fail.")
            
        print(f"[INFO] Initializing Retargeting with URDF: {urdf_path}")
        
        robot = RobotWrapper(urdf_path)
        
        # Inject Virtual Frames (tool0_z, tool0_y)
        model = robot.model
        if model.existFrame("tool0") and not model.existFrame("tool0_z"):
            tool0_id = model.getFrameId("tool0")
            parent_placement = model.frames[tool0_id].placement
            
            d_z = pin.SE3.Identity()
            d_z.translation = np.array([0.0, 0.0, 0.5])
            model.addFrame(pin.Frame("tool0_z", model.frames[tool0_id].parent, tool0_id, parent_placement * d_z, pin.FrameType.OP_FRAME))
            
            d_y = pin.SE3.Identity()
            d_y.translation = np.array([0.0, 0.5, 0.0])
            model.addFrame(pin.Frame("tool0_y", model.frames[tool0_id].parent, tool0_id, parent_placement * d_y, pin.FrameType.OP_FRAME))
            
            robot.data = model.createData()
            
        self.optimizer = VectorOptimizer(
            robot=robot,
            target_joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ],
            target_origin_link_names=["ur3e_base_link", "tool0", "tool0"],
            target_task_link_names=["tool0", "tool0_z", "tool0_y"],
            target_link_human_indices=np.array([0, 1, 2], dtype=int),
            scaling=1.0
        )
        
        # Clamp Base Joint to prevent flips
        robot.model.lowerPositionLimit[0] = -1.6 
        robot.model.upperPositionLimit[0] = 1.6
        
        self.last_q = np.array(home_joints) 
        self.vector_scale = 0.1 
        
        self.num_fixed = robot.model.nq - 6
        self.fixed_qpos = np.zeros(self.num_fixed)
        
    def solve(self, target_pos, target_rot):
        if np.isnan(target_pos).any() or np.isnan(target_rot).any():
             return self.last_q

        target_vecs = np.vstack([
            target_pos,
            target_rot[:, 2] * self.vector_scale,
            target_rot[:, 1] * self.vector_scale
        ])
        
        try:
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
        model = self.optimizer.robot.model
        data = self.optimizer.robot.data
        
        if q.shape[0] != model.nq:
            q_padded = np.zeros(model.nq)
            q_padded[:min(q.shape[0], model.nq)] = q
            q = q_padded
            
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        if self.optimizer.robot.model.existFrame("tool0"):
            frame_id = self.optimizer.robot.model.getFrameId("tool0")
            frame = self.optimizer.robot.data.oMf[frame_id]
            return frame.translation.copy(), frame.rotation.copy()
        
        return np.zeros(3), np.eye(3)

    def reset_state(self, q: np.ndarray):
        self.last_q = q
