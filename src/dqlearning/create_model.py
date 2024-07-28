import src.dqlearning.constants as constants
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, clone_model

def create_model():
  model = Sequential()
  model.add(Dense(16, input_shape=(1, constants.NUM_OBSERVATIONS)))
  model.add(Activation('relu'))

  model.add(Dense(32))
  model.add(Activation('relu'))

  model.add(Dense(constants.NUM_ACTIONS))
  model.add(Activation('linear'))

  return (model, clone_model(model))
