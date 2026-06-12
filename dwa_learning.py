import numpy as np

# ======================
# DWA 动态窗口法 核心算法
# ======================

class DWA:
    def __init__(self):
        # 小车物理限制（你PPT里的“速度窗口”）
        self.max_v = 0.4      # 最大线速度
        self.min_v = 0.0      # 最小线速度
        self.max_w = 1.0      # 最大角速度
        self.max_acc_v = 0.2  # 最大线加速度
        self.max_acc_w = 2.0  # 最大角加速度
        self.dt = 0.1         # 时间间隔

        # 评价函数权重（你PPT里的公式！）
        self.weight_heading = 0.3  # 朝向目标权重
        self.weight_dist = 0.5     # 避障权重
        self.weight_vel = 0.2      # 速度权重

    # 计算动态速度窗口
    def calc_dynamic_window(self, v, w):
        # 根据当前速度，算出下一帧能达到的速度范围
        v_min = max(self.min_v, v - self.max_acc_v * self.dt)
        v_max = min(self.max_v, v + self.max_acc_v * self.dt)
        w_min = max(-self.max_w, w - self.max_acc_w * self.dt)
        w_max = min(self.max_w, w + self.max_acc_w * self.dt)
        return [v_min, v_max, w_min, w_max]

    # 模拟一条轨迹（给定速度，预测小车怎么走）
    def calc_trajectory(self, v, w):
        x, y, yaw = 0, 0, 0
        trajectory = []
        for _ in range(20):  # 预测20步
            yaw += w * self.dt
            x += v * np.cos(yaw) * self.dt
            y += v * np.sin(yaw) * self.dt
            trajectory.append([x, y, yaw])
        return trajectory, v, w

    # 评价函数：给轨迹打分（你PPT的公式！）
    def calc_score(self, traj, goal, obs):
        last = traj[-1]
        
        # 1. 朝向目标得分
        heading = np.arctan2(goal[1]-last[1], goal[0]-last[0]) - last[2]
        heading_score = np.cos(heading)

        # 2. 避障得分
        dist_score = 1.0
        for ob in obs:
            dx = last[0] - ob[0]
            dy = last[1] - ob[1]
            d = np.hypot(dx, dy)
            dist_score = min(dist_score, d)
        dist_score = max(dist_score, 0.1)

        # 3. 速度得分
        vel_score = abs(traj[-1][3])

        # 总评分（PPT公式）
        total = (self.weight_heading * heading_score
               + self.weight_dist * dist_score
               + self.weight_vel * vel_score)
        return total

    # 主函数：找最优速度
    def plan(self, goal, obs, v, w):
        dw = self.calc_dynamic_window(v, w)
        best_score = -1e9
        best_v, best_w = 0, 0

        # 遍历所有可能速度（采样）
        for v in np.linspace(dw[0], dw[1], 20):
            for w in np.linspace(dw[2], dw[3], 40):
                traj, v, w = self.calc_trajectory(v, w)
                score = self.calc_score(traj, goal, obs)
                
                # 选最高分
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
        return best_v, best_w