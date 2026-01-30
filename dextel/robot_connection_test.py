import rtde_receive
import rtde_control

ROBOT_IP = "137.49.35.26"

try:
    print(f"Connecting to robot at {ROBOT_IP}...")
    
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

    print("Connection successful!")
    actual_tcp_pose = rtde_r.getActualTCPPose()
    print(f"Current TCP position: {actual_tcp_pose}")

except Exception as e:
    print(f"Connection failed: {e}")