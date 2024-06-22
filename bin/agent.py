import numpy as np
from src.env import env
from src.helpers import path
import src.qlearning as qlearn
import matplotlib.pyplot as plt
from src.constants import EPOCHS, EPSILON, Q_TABLE

rewards = []
log_interval = 100
render_interval = 1000

points_log = []
mean_points_log = []

epoch_plot_tracker = []
total_reward_plot_tracker = []

for epoch in range(EPOCHS):
    state = env.reset()
    discretized_state = qlearn.discretize_obs(state)

    points = 0
    epoch_plot_tracker.append(epoch)

    terminated = False
    total_rewards = 0

    # Agents play game
    while not terminated:
        if epoch % render_interval == 0:  
          env.render('human')

        # ACTION
        action = qlearn.action_selection(EPSILON, discretized_state)

        # Next steps
        new_state, reward, terminated, info = env.step(action)

        reward = qlearn.on_fail(terminated, points, reward)

        new_state_discretized = qlearn.discretize_obs(new_state)

        # Old (Current) Q Value. Q(st, at)
        old_q_value = Q_TABLE[discretized_state + (action,)]

        # The next Q Value (Max Q Value for this state). Q(st+1, at+1)
        new_q_value = np.max(Q_TABLE[new_state_discretized])

        # Compute next Q Value
        next_q = qlearn.compute_next_q_value(reward, old_q_value, new_q_value)

        # Update the table
        Q_TABLE[discretized_state + (action,)] = next_q

        discretized_state = new_state_discretized
        points += 1

    # Agent finished a round of the game
    EPSILON = qlearn.reduce_epsilon(EPSILON, epoch)
    points_log.append(points)
    running_mean = round(np.mean(points_log[-30:]), 2)
    mean_points_log.append(running_mean) 

    if epoch % log_interval == 0:
        print(f'Epoch: {epoch}, Rewards: {np.sum(rewards)}, Running Mean: {running_mean}')

env.close()
plt.figure(figsize=(8, 8))
plt.scatter(epoch_plot_tracker, points_log)
plt.plot(epoch_plot_tracker, points_log)
plt.plot(epoch_plot_tracker, mean_points_log, label=f'Running Mean: {running_mean}')
plt.legend()
plt.savefig(path.plots('epoch-points-log-tracker.png'))
