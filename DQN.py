import random
import itertools
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from wrappers import *

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 200000
MIN_REPLAY_SIZE = 2000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 150000
TARGET_UPDATE_FREQUENCY = 10000


class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        
        # features
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # fully connencted
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = x.float()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def feature_size(self):
        return self.features(torch.autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0)
        q_values = self.forward(state)
        action = q_values.max(1)[1].item()
        return action
       
       
def savenet(model, filename):
    torch.save(model.state_dict(), './models/' + filename)


def loadnet(environment, filename="agent.pth"):
    net = DQN(environment)
    net.load_state_dict(torch.load('./models/' + filename))
    net.eval()
    return net


if __name__ == "__main__":
    writer = SummaryWriter()
    
    env_id = "ALE/Breakout-v5"
    env    = make_atari(env_id)
    env    = wrap_deepmind(env)
    env    = wrap_pytorch(env)
    
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
    avg_rewards = []
    steps = []
    all_rewards = []

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
            all_rewards.append(episode_reward)
            
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
        if step % 100 == 0:
            writer.add_scalar('DQN/avg_rew',
                              np.mean(rew_buffer),
                              step)
            avg_rewards.append(np.mean(rew_buffer))
            steps.append(step)
            
        if (step % 1000 == 0) and (step > 0):
            print('')
            print('Step', step)
            print('Avg Rew', np.mean(rew_buffer))
            print('Epsilon val', epsilon)
            print('Last 5avg ', np.mean(all_rewards[-5:]))
            
        # save dqn at checkpoints
        if (step % 10000 == 0) and (step > 0): #save every 10k
            savenet(online_net, 'agent.pth')
      
        if step == 10000: #10k
            savenet(online_net, "agent1.pth")
            
        if step == 50000: #50k
            savenet(online_net, "agent2.pth")
            
        if step == 100000: #100k
            savenet(online_net, "agent3.pth")
            
        if step == 200000: #200k
            savenet(online_net, "agent4.pth")
            
        if step == 250000: #250k
            savenet(online_net, "agent5.pth")
            
        # used for plotting training results
        if (step % 50000 == 0) and (step > 190000):
            plt.plot(steps, avg_rewards)
            plt.xlabel("Steps")
            plt.ylabel("Avg reward")
            plt.show()

            plt.plot([i for i in range(1, len(all_rewards) + 1)] , all_rewards)
            plt.xlabel("Episodes")
            plt.ylabel("Episode rewards")
            plt.show()

    writer.close()
