#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class DWANode(Node):
    def __init__(self):
        super().__init__("dwa_avoidance_node")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        
        self.dwa = DWA()
        self.obstacles = []
        self.current_v = 0.0
        self.current_w = 0.0
        self.goal = [2.0, 0.0]  # 目标点

    def odom_cb(self, msg):
        self.current_v = msg.twist.twist.linear.x
        self.current_w = msg.twist.twist.angular.z

    def scan_cb(self, msg):
        obs = []
        angle = msg.angle_min
        for r in msg.ranges:
            if 0.1 < r < 3.0:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                obs.append([x, y])
            angle += msg.angle_increment
        self.obstacles = obs
        self.run_dwa()

    def run_dwa(self):
        v, w = self.dwa.plan(self.goal, self.obstacles, self.current_v, self.current_w)
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = DWANode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()