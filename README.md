# Reinforcement Learning assignment 2: Deep Q-learning agent

## Installation (Linux)

1. Create a virtual environment with

> python3 -m venv venv

2. Enter the environment with

> source venv/bin/activate

3. Install the requirements with

> pip install -r requirements.txt

4. Clone the Gym Github repository with

> git clone git@github.com:openai/gym.git

or download and extract from the [Github link](https://github.com/openai/gym)

5. Make sure to place the .py files in the root of the cloned folder

## Running the agent

The training and testing can be run with

> python3 DQN.py

Possible parameters are:

- **-r / --disable_experience_replay** -> Runs DQN without experience replay. Experience replay is on by default
- **-n / --disable_target_network** -> Runs DQN without target network. Target network is on by default
- **-p / --policy** -> Uses one of the following selection policies: epsilon-greedy (0) (Default), softmax (1), UCB_con (2) or noisy epsilon-greedy (3)
- **-e / --eptemp** -> Sets the epsilon value for (noisy) epsilon-greedy policy or the temperature for softmax/UCB_con policy. Default: 0.1
- **-l / --learning_rate** -> Sets the learning rate for the training. Default: 0.1
- **-m / --max_episodes** -> Sets the episode length for training. Default: 500
- **-s / --test_length** -> Sets the episode length for testing. Default: 100
- **-d / --discount_factor** -> Sets the discount factor for the training. Default: 0.85
- **-o / --epochs** -> Sets the number of epochs in the first layer of the model. Default: 64
- **-i {path/to/model.keras} / --import_model {path/to/model.keras}** -> Import existing model, skips training and runs tests. Argument: path/to/model.keras. Default: False. Note: it is still required to provide parameters -p, -e, -l, -m, -d, -o if they deviate from the default values
- **-t / --training_only** -> Runs DQN training only without testing. Default: false. Note: using -t with -i results in no training and testing. This argument can be useful when training is very long and testing has to be done seperately.
- **-v / --verbose** -> Displays the environment during testing. Default: off

## Experiments

For our experiments, the DQN runs have been prepared in 'experiments.py'. These can be run with

> ./experiments.py

To see which experiments were done specifically, you can read the contents of 'experiments.py' or look at the method section of the report.

## Plotting figures

After training and testing, there should be a file named 'results.csv', which contains all the test run averages from all experiments. With all experiment data in 'results.csv', the plots that were used in the report can be generated with

> python3 plot.py
