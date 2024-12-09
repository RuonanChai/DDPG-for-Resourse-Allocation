import torch
import numpy as np
from agent import DDPG
from environment import ResourceEnv  # 假设你定义的环境文件是 environment.py
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def evaluate_policy(agent, env, episodes=10):
    """评估智能体性能"""
    agent.actor.eval()  # 切换到评估模式
    total_rewards = []  # 用于存储每个评估 Episode 的奖励
    actions = []  # 用于存储评估过程中采取的动作

    for i in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(agent.device)
            action = agent.actor(state_tensor).cpu().detach().numpy()[0]  # 动作生成
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            actions.append(action)

        total_rewards.append(episode_reward)

        print(f"Episode {i}, Reward: {episode_reward}")

    # 计算平均奖励和标准差
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Evaluation over {episodes} episodes: Average Reward: {avg_reward}, Std: {std_reward}")

    # 绘制评估结果的柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(['Average Reward'], [avg_reward], yerr=[std_reward], capsize=10, color='skyblue')
    plt.ylabel('Reward')
    plt.title('Evaluation Performance')
    plt.grid()
    plt.show()

def main():
    # 初始化超参数
    n_states = 8  # 状态空间维度
    n_actions = 3  # 动作空间维度
    action_bound = 1  # 动作范围 [-1, 1]
    n_hiddens = 128  # 隐藏层神经元个数
    actor_lr = 1e-4  # 策略网络学习率
    critic_lr = 1e-4  # 价值网络学习率
    gamma = 0.99  # 奖励折扣因子
    tau = 0.005  # 软更新参数
    sigma = 0.3  # 探索噪声标准差
    buffer_size = 100000  # 经验回放池大小
    batch_size = 64  # 批量大小
    episodes = 1000  # 训练轮数
    max_steps_per_episode = 50  # 每轮最大步数

    rewards = []  # 用于记录每个 Episode 的奖励

    # 创建环境和智能体
    env = ResourceEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = DDPG(
        n_states=n_states,
        n_hiddens=n_hiddens,
        n_actions=n_actions,
        action_bound=action_bound,
        sigma=sigma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=gamma,
        device=device,
        buffer_size=buffer_size
    )

    # 训练主循环
    for episode in range(episodes):

        # 动态调整噪声
        agent.sigma = max(0.1, agent.sigma * 0.995)  # 延长探索阶段

        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # 将经验存储到经验回放池中
            agent.memory.push(state, action, reward, next_state, done)

            # 更新模型
            agent.update(batch_size)

            # 动态调整负奖励范围
            if episode > 50 and len(rewards) >= 10 and rewards[-1] > rewards[-10]:
                reward = max(reward, -1)  # 限制负奖励的下限

            total_reward += reward  # 累加总奖励
            state = next_state

        rewards.append(total_reward)  # 记录每个 Episode 的总奖励
        print(f"Episode {episode}, Total Reward: {total_reward}")

    # 绘制奖励变化曲线
    plt.plot(range(len(rewards)), rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Over Episodes")
    plt.legend()
    plt.grid()
    plt.show()

    # 在训练完成后绘制 Actor 和 Critic 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(agent.actor_losses)), agent.actor_losses, label='Actor Loss')
    plt.plot(range(len(agent.critic_losses)), agent.critic_losses, label='Critic Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Actor and Critic Loss During Training')
    plt.legend()
    plt.grid()
    plt.show()

    # 调用评估函数
    print("Training complete!")
    evaluate_policy(agent, env, episodes=10)


if __name__ == '__main__':
    main()