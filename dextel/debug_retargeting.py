
import numpy as np
import os
import pinocchio as pin
from dextel.retargeting import RetargetingWrapper

def main():
    urdf_path = "/Users/jimin/Code/AILAB/DexTel/dextel/assets/ur3e_hande.urdf"
    # Adjust path if running on user machine vs my simulated path
    # On user machine it is: /home/husl-ai/workspace/ros2_ws/src/dextel/dextel/assets/ur3e_hande.urdf
    # I will rely on the user running this from their workspace or I'll use relative.
    # Let's try to use the path from the logs.
    urdf_path = "dextel/assets/ur3e_hande.urdf"
    if not os.path.exists(urdf_path):
        urdf_path = "/home/husl-ai/workspace/ros2_ws/src/dextel/dextel/assets/ur3e_hande.urdf"

    print(f"Testing with URDF: {urdf_path}")
    
    wrapper = RetargetingWrapper(urdf_path)
    model = wrapper.optimizer.robot.model
    
    print(f"Frame 'tool0': {model.existFrame('tool0')}")
    print(f"Frame 'tool0_z': {model.existFrame('tool0_z')}")
    print(f"Frame 'tool0_y': {model.existFrame('tool0_y')}")
    print(f"Base Limits: [{model.lowerPositionLimit[0]}, {model.upperPositionLimit[0]}]")
    
    print(f"Base Limits: [{model.lowerPositionLimit[0]}, {model.upperPositionLimit[0]}]")
    
    print("\n[DEBUG] Optimizer Internals:")
    opt = wrapper.optimizer
    print(f"Computed Link Names: {opt.computed_link_names}")
    print(f"Task Link Names: {opt.task_link_names}")
    print(f"Target Origin Names: {opt.origin_link_names}")
    print(f"Target Link Indices: {opt.target_link_human_indices}")
    
    # Check Base Pose
    base_name = "ur3e_base_link"
    if model.existFrame(base_name):
        bid = model.getFrameId(base_name)
        base_pose = wrapper.optimizer.robot.data.oMf[bid]
        print(f"Base Link ({base_name}) Pose:\n{base_pose}")
    else:
        print(f"Base Link {base_name} NOT FOUND")

    home_joints = np.array([0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
    
    print("\n--- Manual Vector Logic Verification (HOME) ---")
    scale = wrapper.vector_scale
    print(f"Vector Scale: {scale}")
    
    # 1. Compute FK Targets from Home
    t_pos, t_rot = wrapper.compute_fk(home_joints)
    t_vec_z = t_rot[:, 2] * scale
    t_vec_y = t_rot[:, 1] * scale
    print(f"Target Pos: {t_pos}")
    print(f"Target Vec Z: {t_vec_z}")
    print(f"Target Vec Y: {t_vec_y}")
    
    # 2. Compute Robot Vectors at Home
    model = wrapper.optimizer.robot.model
    data = wrapper.optimizer.robot.data
    
    # Pad for Pinocchio
    nq = model.nq
    q_padded = np.zeros(nq)
    q_padded[:len(home_joints)] = home_joints
    
    pin.forwardKinematics(model, data, q_padded)
    pin.updateFramePlacements(model, data)
    
    def get_p(name):
        return data.oMf[model.getFrameId(name)].translation
        
    p_base = get_p("ur3e_base_link")
    p_tool0 = get_p("tool0")
    p_tool0_z = get_p("tool0_z")
    p_tool0_y = get_p("tool0_y")
    
    # Replicate VectorOptimizer Logic: current = task - origin
    # Pair 0: Base -> Tool0 ? No. Check indices.
    # Logic: indices are [0, 1, 2].
    # Task Links: tool0, tool0_z, tool0_y
    # Origin Links: ur3e_base_link, tool0, tool0
    
    rob_vec_0 = p_tool0 - p_base
    rob_vec_1 = p_tool0_z - p_tool0
    rob_vec_2 = p_tool0_y - p_tool0
    
    print(f"Robot Vec 0 (Pos): {rob_vec_0} | Err: {np.linalg.norm(rob_vec_0 - t_pos)}")
    print(f"Robot Vec 1 (Z):   {rob_vec_1} | Err: {np.linalg.norm(rob_vec_1 - t_vec_z)}")
    print(f"Robot Vec 2 (Y):   {rob_vec_2} | Err: {np.linalg.norm(rob_vec_2 - t_vec_y)}")

    # 1. Reset State
    print("\n--- Testing Reset State ---")
    wrapper.reset_state(home_joints)
    
    # 2. Get Home FK
    pos, rot = wrapper.compute_fk(home_joints)
    print(f"Home Pos: {pos}")
    print(f"Home Rot:\n{rot}")
    
    # 3. Solve for Home (Should return home_joints)
    print("\n--- Testing Solve (Home -> Home) ---")
    q_sol = wrapper.solve(pos, rot)
    print(f"Solved Joints: {q_sol}")
    
    # Slice to 6 DOF
    if q_sol.shape[0] > 6:
        q_sol = q_sol[:6]
    
    diff = np.linalg.norm(q_sol - home_joints)
    print(f"Diff Norm (Joint Space): {diff}")
    
    joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
    print("\n--- Detailed Joint Analysis ---")
    for i in range(6):
        d = q_sol[i] - home_joints[i]
        print(f"{joint_names[i]:<10}: Sol={q_sol[i]:6.3f} | Home={home_joints[i]:6.3f} | Diff={d:6.3f} ({np.rad2deg(d):6.1f} deg)")
    
    # Verify FK of solution
    sol_pos, sol_rot = wrapper.compute_fk(q_sol)
    pos_err = np.linalg.norm(sol_pos - pos)
    rot_err = np.linalg.norm(sol_rot - rot) # Frobenious norm approx
    
    print(f"\nFK Verification:")
    print(f"Pos Error: {pos_err:.6f}")
    print(f"Rot Error: {rot_err:.6f}")
    
    # Updated Check (Base clamp fixed the flip, now checking for twist)
    if abs(q_sol[0] - home_joints[0]) > 2.0:
        print("!!! BASE FLIP DETECTED !!!")
    elif diff > 0.5:
        print("!!! TWIST DETECTED (High Joint Diff) !!!")
    else:
        print("Solution OK.")

if __name__ == "__main__":
    main()
