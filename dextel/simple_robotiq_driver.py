import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import socket
import time

class SimpleRobotiqDriver(Node):
    def __init__(self):
        super().__init__('simple_robotiq_driver')
        
        self.declare_parameter('robot_ip', '137.49.35.26')
        self.robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.port = 63352 
        
        self.get_logger().info(f"Connecting to Gripper at {self.robot_ip}:{self.port}...")
        self.sock = None
        self.connect()

        self.sub = self.create_subscription(Float32, '/dextel/gripper_cmd', self.cmd_callback, 10)
        self.get_logger().info("Simple Robotiq Driver Ready. Topic: /dextel/gripper_cmd")

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.robot_ip, self.port))
            
            # Init Sequence (Activate, set Speed/Force)
            self.send_raw("SET ACT 1") 
            time.sleep(2.0) 
            self.send_raw("SET GTO 1") 
            self.send_raw("SET SPE 255")
            self.send_raw("SET FOR 150")
            self.get_logger().info("Connected & Activation Sent (SET ACT 1).")
        except Exception as e:
            self.get_logger().error(f"Connection Failed: {e}")
            self.sock = None

    def cmd_callback(self, msg):
        target = min(max(msg.data, 0.0), 1.0)
        pos_int = int(target * 255)
        
        cmd = f"SET POS {pos_int}" 
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
