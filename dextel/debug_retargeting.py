
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
    
    home_joints = np.array([-0.20635002, 0.13105, 0.3033, -1.8, 1.57, 0.0]) # Approx from logs (first 3 valid, rest dummy?)
    # Wait, logs said: [-0.20635002  0.13105     0.3033    ] for Home Pose Computed. That's XYZ.
    # I need home joints.
    # dextel_node.py defines: self.home_joints = np.deg2rad([0, -90, -90, -90, 90, 0]) approx?
    # Actually checking dextel_node.py lines would be better.
    # In dextel_node.py:
    # self.home_joints = np.deg2rad([0, -45, -90, -135, 90, 0]) ? NO.
    # Let's assume standard intuitive home: [0, -pi/2, -pi/2, -pi/2, pi/2, 0]
    
    home_joints = np.deg2rad([0, -90, -90, -90, 90, 0])
    
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
    print(f"Base Joint: {q_sol[0]} (Expected: {home_joints[0]})")
    
    # Verify FK of solution
    sol_pos, sol_rot = wrapper.compute_fk(q_sol)
    pos_err = np.linalg.norm(sol_pos - pos)
    rot_err = np.linalg.norm(sol_rot - rot) # Frobenious norm approx
    
    print(f"\nFK Verification:")
    print(f"Pos Error: {pos_err}")
    print(f"Rot Error: {rot_err}")
    
    if abs(q_sol[0] - home_joints[0]) > 2.0:
        print("!!! FLIP DETECTED !!!")
    else:
        print("Solution OK.")

if __name__ == "__main__":
    main()
