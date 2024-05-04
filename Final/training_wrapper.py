import time
import numpy as np
import gym
from reinforce import PolicyNetwork
from critic import Critic
from train_model import train_model
from Helper import LearningCurvePlot, smooth
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--baseline_subtraction", default=False, help="Runs actor-critic with baseline subtraction (running without baseline subtraction and bootstrap runs REINFORCE)", action="store_true")
parser.add_argument("-t", "--bootstrap", default=False, help="Runs actor-critic with bootstrap (running without baseline subtraction and bootstrap runs REINFORCE)", action="store_true")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Sets the policy learning rate for the training. Default: 0.001")
parser.add_argument("-k", "--critic_learning_rate", type=float, default=0.05, help="Sets critic the learning rate for the critic for the training. Default: 0.01")
parser.add_argument("-m", "--max_episodes", type=int, default=10000, help="Sets the episode length for training. Default: 10000")
parser.add_argument("-n", "--set_n", type=int, default=1, help="Sets n used for n-step target for actor-critic. Default: 1")
parser.add_argument("-o", "--n_nodes", type=int, default=64, help="Sets the number of nodes in the hidden layer of the policy model. Default: 64")
parser.add_argument("-p", "--critic_n_nodes", type=int, default=64, help="Sets the number of nodes in the hidden layer of the critic model. Default: 64")
parser.add_argument("-v", "--verbose", help="Displays the environment during testing. Default: off", action="store_true")
args = parser.parse_args()

learning_rate = args.learning_rate
max_episodes = args.max_episodes
policy_hidden_nodes = args.n_nodes
critic_learning_rate = args.critic_learning_rate
n = args.set_n
critic_hidden_nodes = args.critic_n_nodes

def call_train_actor_critic(n_repetitions, n_training_episodes, policy_learning_rate, policy_hidden_nodes,
                            eta_entropy=0.01, baseline_subtraction=False, bootstrap=False,
                            critic_learning_rate=0.01, critic_hidden_nodes=64, n=1,
                            smoothing_window=None, eval_interval=100, env_name="LunarLander-v2"):
    '''
    train_actor_critic wrapper. creates actor-critic classes, optimiser and smoothes results.
    '''

    # create environments
    # Environment setup
    env = gym.make(env_name, render_mode=None)
    eval_env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(state_size)
    print(action_size)

    returns_over_repetitions = []
    now = time.time()
    for _ in range(n_repetitions):
        policy = PolicyNetwork(state_size, action_size, policy_learning_rate, policy_hidden_nodes)
        val_func = None
        if bootstrap or baseline_subtraction:
            val_func = Critic(state_size, critic_learning_rate, critic_hidden_nodes)

        eval_rewards, eval_episodes = train_model(policy=policy,
                                                  n_training_episodes=n_training_episodes,
                                                  eval_steps=eval_interval,
                                                  env=env,
                                                  eval_env=eval_env,
                                                  val_func=val_func,
                                                  bootstrap=bootstrap,
                                                  baseline_subtraction=baseline_subtraction,
                                                  n=n,
                                                  eta_entropy=eta_entropy,
                                                  verbose=True)
        returns_over_repetitions.append(eval_rewards)

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)  # average over repetitions
    learning_curve_2 = np.mean(np.array(returns_over_repetitions), axis=0)  # second curve, returned with smooth
    if smoothing_window is not None:
        learning_curve_2 = smooth(learning_curve_2, smoothing_window)  # additional smoothing
    return learning_curve, learning_curve_2, eval_episodes


# Plot results and call wrapper
Plot = LearningCurvePlot(title='Actor Critic Lunar Lander (raw values)')
Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander')

entropies = [0.01, 0.001]
learning_rates = [0.01, 0.001]  # TODO:
for entropy in entropies:
    learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
        1, max_episodes, learning_rate, policy_hidden_nodes, eta_entropy=0.01, baseline_subtraction=args.baseline_subtraction, bootstrap=args.bootstrap,
        critic_learning_rate=critic_learning_rate, critic_hidden_nodes=critic_hidden_nodes,
        n=n, smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
    Plot.add_curve(eval_episodes, learning_curve, label=r'Entropy, $\eta$ = {}'.format(entropy))
    Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Entropy, $\eta$ = {}'.format(entropy))

Plot.save('reinforce_lunar_lander.png')
Plot2.save('reinforce_lunar_lander_smooth.png')
