import numpy as np
import src.constants as const

def create_bins(bins_per_obs=10):
  bins_cart_position = np.linspace(-4.8, 4.8, bins_per_obs)
  bins_cart_velocity = np.linspace(-5, 5, bins_per_obs)
  bins_pole_angle = np.linspace(-0.418, 0.418, bins_per_obs)
  bins_pole_angular_velocity = np.linspace(-5, 5, bins_per_obs)

  return np.array([
      bins_cart_position, 
      bins_cart_velocity, 
      bins_pole_angle, 
      bins_pole_angular_velocity
  ])

def discretize_obs(observations):
  binned_observations = []
  
  for i, observation in enumerate(observations):
    discretized_obs = np.digitize(observation, const.BINS[i])

    binned_observations.append(discretized_obs)

  return tuple(binned_observations)

def action_selection(epsilon, discrete_state):
    random_number = np.random.random()

    # EXPLOITATION (choose the action that maximizes Q)
    if random_number > epsilon:
        state = const.Q_TABLE[discrete_state]

        return np.argmax(state)

    # EXPLORATION (choose random action)
    return np.random.choice([
        const.ACTION.LEFT,
        const.ACTION.RIGHT
    ])

def compute_next_q_value(reward, old_q_value, next_q_value):
    return old_q_value + const.ALPHA * (reward + const.GAMMA * next_q_value - old_q_value)

def reduce_epsilon(epsilon, epoch):
    if const.BURN_IN <= epoch <= const.EPSILON_END:
      return epsilon - const.EPSILON_REDUCE
    
    return epsilon

def on_fail(terminated, points, reward):
  if terminated and points < 150:
    reward = -200

  # Angular velocity +/-
  return reward
