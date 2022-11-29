# Asteroids Reinforcement Learning
## Setup
Install the requirements using `pip install -r requirements.txt`.
Train the model by running `DQN.py`.
See how the agent plays by running `agent_test.py`.

Every 10000 steps, the agent saves its network data to `models/agent.pth`. This can be loaded when running `DQN.py` and is always loaded when running `agent_test.py`.

## Tracking
We keep track of agent metrics using TensorBoard. To see the runs, simply run `tensorboard --logdir runs`.