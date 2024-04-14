import gym
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.saving import load_model

nodes = 24

def agent(input_shape, n_actions):
    initializer = HeUniform()
    model = Sequential()
    model.add(Dense(nodes, input_shape=input_shape, activation='relu', kernel_initializer=initializer))
    model.add(Dense(n_actions, activation='linear', kernel_initializer=initializer))
    model.compile(loss="huber", optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model


# First, let's define and create our environment called 
env = gym.make("Pendulum-v1", render_mode="human")

# Then we reset this environment
observation = env.reset()

# Observation and action space 
observed_space = env.observation_space
action_space = env.action_space
print("The observation space:{}" .format(observed_space))
print("The action space: {}".format(action_space))

agent = agent(observed_space.shape, action_space.shape)

# agent = agent(env.observation_space.shape)

ep_length = 1000
learning_rate = 0.1
theta = np.random.rand(nodes)

#number of samples
m = 100

while ep_length > 0:
    gradient = 0

    trace = deque(maxlen=200)

    for i in range(m):
        observation = env.reset()
        while True:
            # TODO: policy gebruiken ipv random
            action = env.action_space.sample()
            # print("Action taken:", action)

            # Do this action in the environment and get
            # next_state, reward, done and info
            new_observation, reward, truncated, done, info = env.step(action)

            trace.append([observation, action, reward])

            observation = new_observation

            # print("Reward:", reward)
            
            # If the game is done (in our case we land, crashed or timeout)
            if done or truncated:
                # Reset the environment
                print("Environment is reset")
                observation = env.reset()
                break
        R = 0
        for t in range(len(trace), 0, -1):
            R = trace[t][2] + learning_rate * R
            # TODO: gradient += R * 
    
    theta = theta + (learning_rate * gradient)
    
    ep_length -= 1
