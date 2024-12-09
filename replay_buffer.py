import random
import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            int(done)  # 确保布尔值格式一致
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样经验"""
        import random
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

