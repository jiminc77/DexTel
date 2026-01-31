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
            # Activate Gripper (Standard Robotiq Protocol: SET ACT 1)
            self.send_raw("SET ACT 1") 
            time.sleep(2.0) # Wait for activation cycle
            self.send_raw("SET GTO 1") # Go To
            # Set Speed and Force once
            self.send_raw("SET SPE 255")
            self.send_raw("SET FOR 150")
            self.get_logger().info("Connected & Activation Sent (SET ACT 1).")
        except Exception as e:
            self.get_logger().error(f"Connection Failed: {e}")
            self.sock = None

    def cmd_callback(self, msg):
        target = min(max(msg.data, 0.0), 1.0)
        # Map 0.0(Open) -> 0, 1.0(Close) -> 255 (Hand-E 0=Open, 255=Closed)
        pos_int = int(target * 255)
        
        # Send Position Command
        cmd = f"SET POS {pos_int}" 
        # Note: Logic handles SPE/FOR in connect(), or we can resend. keeping it minimal.
        
        self.get_logger().info(f"Gripper CMD: {target:.2f} -> '{cmd}'")
        self.send_raw(cmd)

    def send_raw(self, text):
        if self.sock is None: 
            self.get_logger().warn("Socket not connected, dropping command.")
            return
        try:
            cmd = text + "\n"
            self.sock.sendall(cmd.encode('utf-8'))
            self.get_logger().info(f"Sent: {text}")
            # Recv response?
            # response = self.sock.recv(1024)
        except Exception as e:
            self.get_logger().error(f"Send Failed: {e}")
            self.sock = None

def main(args=None):
    rclpy.init(args=args)
    node = SimpleRobotiqDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
