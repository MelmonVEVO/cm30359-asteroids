from DQN import DQN, loadnet
import gym

env = gym.make("ALE/Asteroids-v5", render_mode='human')
obs = env.reset()[0]

net = loadnet()

# Doesn't train, this is just to observe the agent playing in real time
for _ in range(1000):
    action = net.act(obs)
    new_obs, rew, done, _, _ = env.step(action)

    if done:
        obs = env.reset()[0]

env.close()
