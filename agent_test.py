from DQN import DQN, loadnet
import gym

env = gym.make("ALE/Breakout-v5", obs_type="grayscale", render_mode="human", full_action_space=False)
obs = env.reset()[0]

net = loadnet(env)
action = 0
# Doesn't train, this is just to observe the agent playing in real time
for _ in range(10000):
    obs, rew, done, _, _ = env.step(action)
    action = net.act(obs)

    if done:
        obs = env.reset()[0]
