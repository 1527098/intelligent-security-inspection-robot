import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
import math
import random

class NavSystemAllInOne(Node):
    def __init__(self):
        super().__init__("nav_system_all_in_one")

        # 订阅话题：里程计、雷达、地图、目标点
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)

        # 发布话题：定位结果、规划路径、速度指令
        self.amcl_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/amcl_pose", 10)
        self.path_pub = self.create_publisher(Path, "/plan", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # 机器人状态
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.goal = None
        self.particles = []          # 粒子集合
        self.map_data = None

        self.get_logger().info("Navigation system started: AMCL + Path Planning + Control")

    # 接收地图，初始化粒子
    def map_cb(self, msg):
        self.map_data = msg
        self.get_logger().info("Map loaded, initializing AMCL particles")
        self.init_particles()

    # 初始化粒子：在地图范围内随机撒点
    def init_particles(self):
        self.particles = []
        resolution = self.map_data.info.resolution
        width = self.map_data.info.width
        height = self.map_data.info.height
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        for _ in range(100):
            self.particles.append({
                "x": random.uniform(origin_x, origin_x + width * resolution),
                "y": random.uniform(origin_y, origin_y + height * resolution),
                "yaw": random.uniform(-math.pi, math.pi),
                "weight": 1.0 / 100
            })

    # 里程计回调：更新机器人位姿，并做粒子预测
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.amcl_predict()
        self.publish_amcl_pose()

    # 粒子预测：加入随机噪声模拟运动不确定性
    def amcl_predict(self):
        for p in self.particles:
            p["x"] += random.gauss(0.0, 0.05)
            p["y"] += random.gauss(0.0, 0.05)
            p["yaw"] += random.gauss(0.0, 0.03)

    # 雷达回调：根据雷达数据更新粒子权重
    def scan_cb(self, msg):
        if not self.particles or self.map_data is None:
            return
        
        total_weight = 0.0
        for p in self.particles:
            p["weight"] = self.calculate_likelihood(p, msg)
            total_weight += p["weight"]
        
        if total_weight > 0:
            for p in self.particles:
                p["weight"] /= total_weight
        
        self.amcl_resample()

    # 简化的似然计算：粒子越接近当前里程计位姿权重越高
    def calculate_likelihood(self, particle, scan_msg):
        dx = particle["x"] - self.x
        dy = particle["y"] - self.y
        distance_error = math.hypot(dx, dy)
        weight = math.exp(-distance_error * 2.0)
        return max(0.001, weight)

    # 重采样：保留权重高的粒子，复制并增加小扰动
    def amcl_resample(self):
        if not self.particles:
            return
        
        self.particles.sort(key=lambda x: -x["weight"])
        top_n = min(50, len(self.particles))
        new_particles = self.particles[:top_n]
        
        while len(new_particles) < 100:
            new_particles.append({
                "x": new_particles[-1]["x"] + random.gauss(0.0, 0.02),
                "y": new_particles[-1]["y"] + random.gauss(0.0, 0.02),
                "yaw": new_particles[-1]["yaw"] + random.gauss(0.0, 0.01),
                "weight": 1.0 / 100
            })
        
        self.particles = new_particles

    # 发布AMCL定位结果
    def publish_amcl_pose(self):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.pose.position.x = self.x
        pose_msg.pose.pose.position.y = self.y
        pose_msg.pose.pose.orientation.w = 1.0
        pose_msg.pose.covariance = [0.1] * 36
        self.amcl_pose_pub.publish(pose_msg)

    # 接收目标点
    def goal_cb(self, msg):
        self.goal = msg.pose
        self.get_logger().info(f"Goal received: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")
        self.publish_global_path()
        self.navigate()

    # 全局路径规划：起点到目标点的直线插值
    def publish_global_path(self):
        if not self.goal:
            return
        
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        num_waypoints = 30
        for i in range(num_waypoints):
            s = i / (num_waypoints - 1)
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = self.x + (self.goal.position.x - self.x) * s
            pose.pose.position.y = self.y + (self.goal.position.y - self.y) * s
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info("Global path published to /plan")

    # 导航控制：根据角度误差输出速度指令
    def navigate(self):
        if not self.goal:
            return
        
        dx = self.goal.position.x - self.x
        dy = self.goal.position.y - self.y
        distance = math.hypot(dx, dy)
        
        twist_msg = Twist()
        
        if distance > 0.3:
            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - self.yaw
            # 角度归一化到[-pi, pi]
            if angle_error > math.pi:
                angle_error -= 2.0 * math.pi
            elif angle_error < -math.pi:
                angle_error += 2.0 * math.pi
            
            twist_msg.linear.x = min(0.3, 0.2 + 0.2 * distance)
            twist_msg.angular.z = 0.8 * angle_error
            twist_msg.angular.z = max(-0.8, min(0.8, twist_msg.angular.z))
            
            self.get_logger().info(f"Navigating: distance={distance:.2f}m")
        else:
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.get_logger().info("Goal reached!")
        
        self.cmd_vel_pub.publish(twist_msg)

def main():
    rclpy.init()
    node = NavSystemAllInOne()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()