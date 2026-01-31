import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import socket
import threading
import time

class SimpleRobotiqDriver(Node):
    def __init__(self):
        super().__init__('simple_robotiq_driver')
        
        # Parameters
        self.declare_parameter('robot_ip', '137.49.35.26')
        self.robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.port = 63352 # Dashboard/Gripper port usually, or we wrap script on 30002
        
        # NOTE: Robotiq Hand-E on UR usually uses a custom URCap port or we send script to 30002.
        # But IFRA driver uses 63352. We will assume 63352 is the "Gripper Server" enabled by URCap.
        # If that fails, we might need 30002 with Script command. 
        # For now, following IFRA approach of Socket @ 63352.
        
        self.get_logger().info(f"Connecting to Gripper at {self.robot_ip}:{self.port}...")
        self.sock = None
        self.connect()

        # Subscriber: 0.0 (Close) -> 1.0 (Open)
        self.sub = self.create_subscription(Float32, '/dextel/gripper_cmd', self.cmd_callback, 10)
        self.get_logger().info("Simple Robotiq Driver Ready. Topic: /dextel/gripper_cmd")

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.robot_ip, self.port))
            # Activate Gripper
            # Assuming IFRA protocol strings or Universal Robots Dashboard strings? 
            # Actually, port 63352 is commonly the "Dashboard" or "Gripper" socket provided by Robotiq URCap.
            # Let's try sending the activation command string if needed.
            # For Hand-E, it often needs 'ACT' command.
            # Reference: wrapper often sends "ACT" then waits.
            self.send_raw("ACT") 
            self.send_raw("GTO") # Go To
            self.get_logger().info("Connected & Activation Sent.")
        except Exception as e:
            self.get_logger().error(f"Connection Failed: {e}")
            self.sock = None

    def cmd_callback(self, msg):
        target = min(max(msg.data, 0.0), 1.0)
        # Robotiq Hand-E: 0 (Open) to 255 (Closed)? Or 0 (Closed) - 255 (Open)?
        # Usually: 0 is Open, 255 is Closed.
        # BUT DexTel Logic: 0.0 = Closed(Pinched), 1.0 = Open.
        
        # Map 0.0(Close) -> 255, 1.0(Open) -> 0
        pos_int = int((1.0 - target) * 255)
        
        # Speed: 255, Force: 150
        cmd = f"POS {pos_int} SPE 255 FOR 150"
        self.send_raw(cmd)

    def send_raw(self, text):
        if self.sock is None: 
            return
        try:
            cmd = text + "\n"
            self.sock.sendall(cmd.encode('utf-8'))
            # Recv response?
            # response = self.sock.recv(1024)
        except Exception as e:
            self.get_logger().error(f"Send Failed: {e}")
            self.sock = None
            # Reconnect?
            # self.connect()

def main(args=None):
    rclpy.init(args=args)
    node = SimpleRobotiqDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
