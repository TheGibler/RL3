import gym

# First, let's define and create our environment called 
env = gym.make("Pendulum-v1", render_mode="human")

# Then we reset this environment
observation = env.reset()

# Observation and action space 
observed_space = env.observation_space
action_space = env.action_space
print("The observation space:{}" .format(observed_space))
print("The action space: {}".format(action_space))

while True:
    # Take a random action
    action = env.action_space.sample()
    print("Action taken:", action)

    # Do this action in the environment and get
    # next_state, reward, done and info
    observation, reward, truncated, done, info = env.step(action)

    print("Reward:", reward)
    
    # If the game is done (in our case we land, crashed or timeout)
    if done:
        # Reset the environment
        print("Environment is reset")
        observation = env.reset()