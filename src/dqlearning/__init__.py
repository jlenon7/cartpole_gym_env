import random
import numpy as np
import src.constants as const
import src.dqlearning.constants as dqconst

def replay(batch_size, model, target_model):
  if (len(dqconst.REPLAY_BUFFER_DEQUE) < batch_size):
     return

  target_batch = []

  samples = random.sample(dqconst.REPLAY_BUFFER_DEQUE, batch_size)
  states, actions, rewards, new_states, dones = list(zip(*samples))

  q_values = model.predict(np.array(new_states))
  targets = target_model.predict(np.array(states))

  for i in range(batch_size):
     q_value = max(q_values[i][0])
     target = targets[i].copy()
     if dones[i]:
        target[0][actions[i]] = rewards[i]
     else:
        target[0][actions[i]] = rewards[i] + q_value * dqconst.GAMMA

     target_batch.append(target)

  model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)

def update_model_handler(epoch, update_target_model, model, target_model):
  if epoch > 0 and epoch % update_target_model == 0:
    target_model.set_weights(model.get_weights())

def action_selection(model, epsilon, observation):
    random_number = np.random.random()

    if random_number > epsilon:
        prediction = model.predict(observation)

        return np.argmax(prediction)

    return np.random.choice([
        const.ACTION.LEFT,
        const.ACTION.RIGHT
    ])
