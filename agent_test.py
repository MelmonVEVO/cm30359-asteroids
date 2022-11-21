from DQN import DQN, loadnet
import gym

env = gym.make("ALE/Asteroids-v5")
obs = env.reset()[0]

net = loadnet(env)

# Doesn't train, this is just to observe the agent playing in real time
# TODO this doesn't work lmao
for _ in range(10000):
    action = net.act(obs)
    totalrew = 0

    obs, rew, done, _, _ = env.step(action)
    totalrew += rew

    if done:
        obs = env.reset()[0]
        print(totalrew)
