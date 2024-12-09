import numpy as np

class ResourceEnv:
    def __init__(self):
        """初始化环境参数"""
        self.state_dim = 8  # 状态空间维度
        self.action_dim = 3  # 动作空间维度 (ΔCPU, ΔRAM, ΔDisk)
        self.action_bound = 1  # 动作范围 [-1, 1]
        self.threshold = 1.0  # 判断资源分配是否足够的阈值

        # 初始化资源和任务需求
        self.state = self._generate_state()
        self.task = self._generate_task()
        self.goal_state = self._generate_goal_state()

    def _generate_state(self):
        """生成资源特征"""
        return {
             "TOPS/W": np.random.uniform(1, 3),  # 能效，较小范围
            "GOPS": np.random.uniform(100, 200),  # 计算能力，较小范围
            "MIPS": np.random.uniform(1000, 1500),  # 指令处理能力
            "RAM": np.random.uniform(4, 8),  # 内存容量，略低
            "CPU_Idle_Rate": np.random.uniform(0.5, 1),  # CPU 空闲率较高
            "GPU_Idle_Rate": np.random.uniform(0.5, 1),  # GPU 空闲率较高
            "Throughput": np.random.uniform(10, 30),  # 吞吐量，较小
            "Remaining_Disk_Storage": np.random.uniform(50, 100),  # 剩余硬盘存储，适中
        }

    def _generate_task(self):
        """生成任务需求特征"""
        scale_factor = 1 + 0.05 * self.threshold  # 根据阈值逐步增加任务需求
        return {
            "CPU_Req": np.random.uniform(0.5, 2) * scale_factor,
            "RAM_Req": np.random.uniform(1, 4) * scale_factor,
            "Disk_Req": np.random.uniform(10, 50) * scale_factor,
        }

    def _generate_goal_state(self):
        """生成目标状态"""
        return np.zeros(self.state_dim)

    def reset(self):
        """重置环境，生成新的初始状态"""
        self.state = self._generate_state()

        # 动态调整任务需求（逐步增加）
        scale_factor = 1 + 0.05 * self.threshold  # 增加任务需求难度
        self.task = {
            "CPU_Req": np.random.uniform(0.5, 2) * scale_factor,
            "RAM_Req": np.random.uniform(1, 4) * scale_factor,
            "Disk_Req": np.random.uniform(10, 50) * scale_factor,
        }

        # 返回状态向量
        return np.array(list(self.state.values()))

    def step(self, action):
        """
        执行动作并计算下一状态、奖励和是否完成
        """
        # 限制动作范围
        action = np.clip(action, -self.action_bound, self.action_bound)

        # 更新资源分配状态
        # 更新资源分配状态，确保资源不会过低
        self.state["CPU_Idle_Rate"] = np.clip(self.state["CPU_Idle_Rate"] + action[0], 0.2, 1.0)  # 最低值不为 0
        self.state["RAM"] = np.clip(self.state["RAM"] + action[1], 0.1, 16.0)
        self.state["Remaining_Disk_Storage"] = np.clip(self.state["Remaining_Disk_Storage"] + action[2], 0.1, 200.0)

        # 计算奖励
        reward = self.compute_reward(action)

        # 生成下一个状态
        next_state = np.array(list(self.state.values()))

        # 判断是否完成
        done = self.is_done()

        return next_state, reward, done

    def compute_reward(self, action):
        """
        改进奖励函数：减少负奖励影响，增加正向奖励。
        """
        # 计算资源分配差距
        cpu_gap = abs(self.state["CPU_Idle_Rate"] - self.task["CPU_Req"])
        ram_gap = abs(self.state["RAM"] - self.task["RAM_Req"])
        disk_gap = abs(self.state["Remaining_Disk_Storage"] - self.task["Disk_Req"])
        total_gap = cpu_gap + ram_gap + disk_gap

        # 动态调整阈值
        self.threshold = max(0.5, self.threshold * 0.995)

        # 正向奖励：接近目标时给予更高奖励
        if total_gap < self.threshold:
            reward = 100.0  # 达到目标时给予高额奖励
        else:
            # 增量奖励：距离目标越近奖励越高
            distance_to_goal = np.linalg.norm(np.array(list(self.state.values())) - self.goal_state)
            incremental_reward = max(10.0 - 0.1 * distance_to_goal, 1.0)

            # 负向奖励：差距带来的惩罚
            gap_penalty = -0.02 * total_gap  # 降低惩罚影响

            reward = gap_penalty + incremental_reward

        # 限制负奖励的范围，确保正奖励的比例更高
        reward = max(reward, -1)  # 限制负奖励下限为 -1
        return reward

    def is_done(self):
        """
        判断是否完成：根据资源条件判断是否结束。
        """
        if self.threshold > 1.0:  # 初期宽松的完成条件
            return (
                    self.state["CPU_Idle_Rate"] < 0.3  # 放宽条件，允许更多的空闲资源
                    or self.state["RAM"] < 0.8 * self.task["RAM_Req"]
                    or self.state["Remaining_Disk_Storage"] < 0.8 * self.task["Disk_Req"]
            )
        else:  # 随着训练的进行，逐步收紧完成条件
            return (
                    self.state["CPU_Idle_Rate"] < self.threshold
                    or self.state["RAM"] < self.task["RAM_Req"]
                    or self.state["Remaining_Disk_Storage"] < self.task["Disk_Req"]
            )

