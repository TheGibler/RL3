import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

learning_rate = 0.01

# fit custom model method from tensorflow keras
class CustomSequential(keras.Sequential):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        observations, actions, rewards = data
        R = 0
        returns = np.zeros(200)
        for t in range(len(rewards) - 1, -1, -1):
            # compute state returns by looping back from the end state to the start
            R = rewards[t] + learning_rate * R
            returns[t] = R

        with tf.GradientTape() as tape:
            means = self.call(tf.convert_to_tensor(observations, dtype=tf.float32), training=True)
            # means = self.call(np.array(observations), training=True)
            means = means
            action_dist = tfp.distributions.Normal(loc=means, scale=std)  # Assuming fixed std=1
            # times 2 dist
            # action_dist = tfp.distributions.LogNormal(means, std)
            log_probs = action_dist.prob(actions)

            # TO DO: get better log_prob
            loss = -log_probs * returns
            # compute loss
            # log_prob = -0.5 * np.log(2 * np.pi * std*std) - (data[1] - action_mean) ** 2 / (2 * std * std)
            # loss[t] = - log_prob * R

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(np.array(actions), means)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def build_agent(input_shape, n_actions=1):
    model = CustomSequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_actions))
    model.compile(optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

# First, let's define and create our environment called 
env = gym.make("Pendulum-v1")  # render_mode="human"
# Then we reset this environment
observation = env.reset()
# action_shape = env.action_space.n
state_shape = env.observation_space.shape
# Observation and action space
agent = build_agent(state_shape, 1)

max_episodes = 100
learning_rate = 0.1

# set std of the action output
std = 0.4

# number of training eps
i = 0
while i < max_episodes:
    i += 1
    observations = []
    actions = []
    rewards = []
    
    observation = env.reset()
    observation = observation[0]
    while True:
        action_mean = agent.predict(np.array([observation]), verbose=0)
        action = min([2], max([-2], np.random.normal(action_mean[0], std) * 2))
        
        new_observation, reward, truncated, done, info = env.step(action)

        observations.append(observation)
        actions.append(action[0])
        rewards.append(reward)

        observation = new_observation

        if done or truncated:
            observation = env.reset()
            break
    
    print("Episode", i, "cum reward:", sum(rewards))

    # train agent on data
    agent.train_step([observations, actions, rewards])

env.close()


def evaluate(agent, render_m=None):
    tenv = gym.make("Pendulum-v1", render_mode=render_m)
    eval_returns = []
    for _ in range(3):
        observation, _ = tenv.reset()
        rewards = []
        while True:
            action_mean = agent.predict(np.array([observation]), verbose=0)
            action = min([2], max([-2], np.random.normal(action_mean[0], std) * 2))
            
            new_observation, reward, truncated, done, info = tenv.step(action)
            rewards.append(reward)
            observation = new_observation

            if done or truncated:
                observation = tenv.reset()
                break

        eval_returns.append(sum(rewards))
    print("average:", np.average(eval_returns))
    tenv.close()

evaluate(agent, render_m="human")



