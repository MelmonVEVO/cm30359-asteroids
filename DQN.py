import gym
from collections import deque
import random
import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQUENCY = 1000


class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)

        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action


def savenet(model: DQN, filename="agent.pth"):
    torch.save(model.state_dict(), './models/' + filename)


def loadnet(environment, filename="agent.pth"):
    net = DQN(environment)
    net.load_state_dict(torch.load('./models/' + filename))
    net.eval()
    return net


if __name__ == "__main__":
    writer = SummaryWriter()

    env = gym.make("ALE/Breakout-v5", obs_type="grayscale", full_action_space=False)

    replay_buffer = deque(maxlen=BUFFER_SIZE)

    rew_buffer = deque([0.0], maxlen=500)

    episode_reward = 0.0

    online_net = DQN(env)
    target_net = DQN(env)
    if input("load?") == 'yes':
        online_net = loadnet(env)
        target_net = loadnet(env)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

    # writer.add_graph(online_net)

    # Initialise replay buffer
    obs = env.reset()[0]
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()

        new_obs, rew, done, _, _ = env.step(action)
        if rew < 0:
            print(rew)
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
        obs = new_obs

        if done:
            obs = env.reset()[0]

    # Main training loop

    obs = env.reset()[0]

    for step in itertools.count():
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        rnd_sample = random.random()

        if rnd_sample <= epsilon:
            action = env.action_space.sample()

        else:
            action = online_net.act(obs)

        new_obs, rew, done, _, _ = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        if rew < 0:
            print(rew)
        replay_buffer.append(transition)
        obs = new_obs

        episode_reward += rew

        if done:
            obs = env.reset()[0]

            rew_buffer.append(episode_reward)
            episode_reward = 0.0


        # Start gradient step
        transitions = random.sample(replay_buffer, BATCH_SIZE)

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute targets
        target_q_values = target_net(new_obses_t)
        mac_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * mac_target_q_values

        # Compute loss
        q_values = online_net(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        if step % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(online_net.state_dict())

        # logging
        if step % 1000 == 0:
            print('')
            print('Step', step)
            print('Avg Rew past 1000 steps', np.mean(rew_buffer))
            print('Epsilon', epsilon)
            writer.add_scalar('DQN/reward',
                              episode_reward,
                              step)
            writer.add_scalar('DQN/avg_rew',
                              np.mean(rew_buffer),
                              step)
            writer.add_scalar('DQN/loss',
                              loss,
                              step)
            rew_buffer.clear()

        if step % 10000 == 0:
            savenet(online_net)

    writer.close()
