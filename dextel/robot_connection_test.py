import rtde_receive
import rtde_control

ROBOT_IP = "137.49.35.26"

try:
    print(f"Connecting to robot at {ROBOT_IP}...")
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    print("Connection successful!")
    print(f"Current TCP position: {rtde_r.getActualTCPPose()}")
except Exception as e:
    print(f"Connection failed: {e}")