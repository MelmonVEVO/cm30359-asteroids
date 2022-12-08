from DQN import DQN, loadnet
import gym
from wrappers import *
import random

env_id = "ALE/Breakout-v5"
env    = make_atari(env_id, render_mode="human")
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

obs = env.reset()[0]

net = loadnet(env)
action = 0

# Doesn't train, this is just to observe the agent playing in real time
for _ in range(10000):
    obs, rew, done, _, _ = env.step(action)
    action = net.act(obs)
    if done: 
        obs = env.reset()[0]
        
