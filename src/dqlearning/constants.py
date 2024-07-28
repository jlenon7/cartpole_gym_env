from collections import deque

# env.observation_space.shape[0]
NUM_OBSERVATIONS = 4
# env.action_space.n
NUM_ACTIONS = 2

EPOCHS = 1000

EPSILON = 1.0
EPSILON_REDUCE = 0.995

LEARNING_RATE = 0.001
GAMMA = 0.95

UPDATE_TARGET_MODEL = 10
REPLAY_BUFFER_DEQUE = deque(maxlen=20000)
