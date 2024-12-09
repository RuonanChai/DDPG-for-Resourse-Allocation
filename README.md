# DDPG-for-Resourse-Allocation

This project implements the Deep Deterministic Policy Gradient (DDPG) algorithm, a model-free off-policy reinforcement learning algorithm. The agent learns to perform actions in an environment to maximize cumulative rewards. DDPG is designed to work in continuous action spaces, making it an ideal algorithm for environments such as robotics and autonomous vehicles.

## Research Design

### Overview of DDPG

Deep Deterministic Policy Gradient (DDPG) is an off-policy, model-free algorithm for reinforcement learning. It is an extension of the deterministic policy gradient algorithm, where an agent learns a deterministic policy in a continuous action space. The key components of the DDPG algorithm are:

- **Actor-Critic Architecture**: The agent uses two neural networks: the Actor, which determines the action to take given a state, and the Critic, which evaluates the action by estimating the Q-value (the expected return for that action-state pair).
- **Experience Replay**: The agent stores its experiences in a buffer and samples them randomly for training, which helps improve sample efficiency and stability.
- **Target Networks**: The algorithm uses target networks to stabilize the learning process, where the target networks slowly track the learned networks.

### Problem Setup

This project implements the DDPG algorithm in a typical reinforcement learning setup. The agent interacts with an environment defined in `environment.py` and tries to maximize the cumulative reward over time by learning a suitable policy.

The main components of the system are:
1. **Actor Network (`PolicyNet`)**: The actor network generates actions from the current state. The actions are constrained to a range defined by `action_bound`.
2. **Critic Network (`QValueNet`)**: The critic evaluates the actions taken by the actor by estimating the Q-value.
3. **Replay Buffer (`ReplayBuffer`)**: The replay buffer stores the agent's experiences (state, action, reward, next state) to enable off-policy updates.
4. **Target Networks**: The target networks are used to generate more stable learning targets. They are slowly updated to track the learned networks.

### Design of DDPG

- **Actor-Critic Networks**: The architecture uses two separate neural networks for the actor and the critic. The actor network is responsible for selecting actions based on the current state, while the critic network evaluates the actions taken by the actor. Both networks are updated using the Bellman equation and temporal difference learning.

- **Experience Replay**: The replay buffer is a circular queue that stores the agent's experiences. Instead of learning from the most recent experiences only, the agent samples randomly from the buffer to break the temporal correlations, improving stability and sample efficiency.

- **Soft Target Update**: To stabilize training, the target networks for both the actor and the critic are updated slowly, using a technique known as **soft target update**. The parameters of the target networks are updated using a weighted average of the current networks and the target networks.

- **Exploration vs. Exploitation**: The exploration strategy is implemented using noise in the action space (`sigma`). Initially, the agent explores more by taking random actions, and as training progresses, the agent increasingly exploits the learned policy to maximize rewards.

### Key Parameters

- `n_states`: The number of state variables in the environment.
- `n_actions`: The number of actions the agent can take.
- `action_bound`: The maximum absolute value of the action.
- `gamma`: Discount factor for future rewards.
- `tau`: Soft target update rate, controlling how fast the target networks are updated.
- `actor_lr` and `critic_lr`: Learning rates for the actor and critic networks.
- `buffer_size`: The size of the experience replay buffer.

### Training Process

The training process follows these steps:
1. Initialize the actor and critic networks and their target counterparts.
2. Interact with the environment and collect experiences (state, action, reward, next state).
3. Store experiences in the replay buffer.
4. Sample a batch of experiences from the buffer and perform updates on the actor and critic networks.
5. Update the target networks using soft target updates.
6. Repeat steps 2–5 for a number of episodes or until convergence.

### Evaluation

Once the training completes, the agent's performance can be evaluated by running the learned policy in the environment. The evaluation involves testing the agent over a set of episodes, recording the cumulative rewards, and comparing it to a baseline or desired performance.

## Requirements

- Python 3.x
- PyTorch
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RuonanChai/DDPG-for-Resourse-Allocation.git
   cd DDPG-for-Resourse-Allocation
   
2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use venv\\Scripts\\activate

3. Install required packages:
pip install -r requirements.txt

## Usage
To train the agent, run the following command:

python main.py
