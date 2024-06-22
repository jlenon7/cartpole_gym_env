import numpy as np
from src.qlearning import create_bins

ENV_NAME = 'CartPole-v1'

NUM_BINS = 10
BINS = create_bins(NUM_BINS)

Q_TABLE = None

# Episodes
EPOCHS = 20000

# Learning rate
ALPHA = 0.8

# Discount rate
GAMMA = 0.9

EPSILON = 1.0
BURN_IN = 1
EPSILON_END = 10000
EPSILON_REDUCE = 0.0001

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

class ACTION:
    LEFT = 0
    RIGHT = 1

def initialize_qtable(action_space):
  global Q_TABLE
  Q_TABLE = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, action_space))
