import gym
from gym.utils.play import play

play(gym.make("ALE/Breakout-v5", render_mode="rgb_array"), zoom=3)
