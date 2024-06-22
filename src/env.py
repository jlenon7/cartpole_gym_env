import gym
from src.constants import initialize_qtable, ENV_NAME

env = gym.make(ENV_NAME)
rgb_env = gym.make(ENV_NAME, render_mode='rgb_array')

action_space = env.action_space.n

initialize_qtable(action_space)
