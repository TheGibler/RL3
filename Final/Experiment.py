import time
import numpy as np
import gym
from reinforce import PolicyNetwork
from critic import Critic
from train_model import train_model
from Helper import LearningCurvePlot, smooth
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment", type=int, default=-1, help="Specify the experiment that must be run. Order starting from 0: [entropy, learning_rate, max_episodes, policy_hidden_nodes, critic_learning_rate, n, critic_hidden_nodes, baseline_subtraction, bootstrap]")
args = parser.parse_args()


#Run experiments with the listed hyperparameters. By default uses the first value of each list
entropies = [0.01, 0.001]
learning_rate = [0.001, 0.005, 0.01]
max_episodes = [10000]
policy_hidden_nodes = [64, 32, 128]
critic_learning_rate = [0.05, 0.01, 0.005]
n = [1, 2, 4]
critic_hidden_nodes = [64, 128, 32]
baseline_subtraction = [True, False]
bootstrap = [True, False]

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

experiment = "All experiments"

if args.experiment:
    if args.experiment == 0:
        experiment = "Entropy"
    elif args.experiment == 1:
        experiment = "Learning rate"
    elif args.experiment == 2:
        experiment = "Policy hidden nodes"
    elif args.experiment == 3:
        experiment = "Critic learning rate"
    elif args.experiment == 4:
        experiment = "n"
    elif args.experiment == 5:
        experiment = "Critic hidden nodes"
    elif args.experiment == 6:
        experiment = "Baseline subtraction"
    elif args.experiment == 7:
        experiment = "Bootstrap"

print("Experiment: ", experiment)


if args.experiment == -1 or args.experiment == 0:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander Entropies (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander Entropies')
    for entropy in entropies:
        print("Entropy: ", entropy)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], learning_rate[0], policy_hidden_nodes[0], eta_entropy=entropy, baseline_subtraction=baseline_subtraction[0], bootstrap=bootstrap[0],
            critic_learning_rate=critic_learning_rate[0], critic_hidden_nodes=critic_hidden_nodes[0],
            n=n[0], smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        #Note: replacing "Entropy" with experiment variable does not work when all experiments are done
        Plot.add_curve(eval_episodes, learning_curve, label=r'Entropy, $\eta$ = {}'.format(entropy))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Entropy, $\eta$ = {}'.format(entropy))
    Plot.save('reinforce_lunar_lander_entropy.png')
    Plot2.save('reinforce_lunar_lander_entropy_smooth.png')

    print("Entropy saved")

if args.experiment == -1 or args.experiment == 1:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander Learning Rate (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander Learning Rate')
    for lr in learning_rate:
        print("Learning rate: ", lr)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], lr, policy_hidden_nodes[0], eta_entropy=entropies[0], baseline_subtraction=baseline_subtraction[0], bootstrap=bootstrap[0],
            critic_learning_rate=critic_learning_rate[0], critic_hidden_nodes=critic_hidden_nodes[0],
            n=n[0], smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        Plot.add_curve(eval_episodes, learning_curve, label=r'Learning rate, $\alpha$ = {}'.format(lr))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Learning rate, $\alpha$ = {}'.format(lr))

    Plot.save('reinforce_lunar_lander_learningrate.png')
    Plot2.save('reinforce_lunar_lander_learningrate_smooth.png')

    print("Learning rate saved")

if args.experiment == -1 or args.experiment == 2:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander Policy hidden nodes (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander Policy hidden nodes')
    for nodes in policy_hidden_nodes:
        print("Policy hidden nodes: ", nodes)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], learning_rate[0], nodes, eta_entropy=entropies[0], baseline_subtraction=baseline_subtraction[0], bootstrap=bootstrap[0],
            critic_learning_rate=critic_learning_rate[0], critic_hidden_nodes=critic_hidden_nodes[0],
            n=n[0], smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        Plot.add_curve(eval_episodes, learning_curve, label=r'Policy hidden nodes = {}'.format(nodes))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Policy hidden nodes = {}'.format(nodes))

    Plot.save('reinforce_lunar_lander_policy_nodes.png')
    Plot2.save('reinforce_lunar_lander_policy_nodes_smooth.png')

    print("Policy hidden nodes saved")

if args.experiment == -1 or args.experiment == 3:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander Critic learning rate (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander Critic learning rate')
    for lr in critic_learning_rate:
        print("Critic learning rate: ", lr)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], learning_rate[0], policy_hidden_nodes[0], eta_entropy=entropies[0], baseline_subtraction=baseline_subtraction[0], bootstrap=bootstrap[0],
            critic_learning_rate=lr, critic_hidden_nodes=critic_hidden_nodes[0],
            n=n[0], smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        Plot.add_curve(eval_episodes, learning_curve, label=r'Critic learning rate, $\alpha_c$ = {}'.format(lr))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Critic learning rate, $\alpha_c$ = {}'.format(lr))

    Plot.save('reinforce_lunar_lander_critic_learningrate.png')
    Plot2.save('reinforce_lunar_lander_critic_learningrate_smooth.png')

    print("Critic learning rate saved")

if args.experiment == -1 or args.experiment == 4:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander n (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander n')
    for m in n:
        print("n: ", n)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], learning_rate[0], policy_hidden_nodes[0], eta_entropy=entropies[0], baseline_subtraction=baseline_subtraction[0], bootstrap=bootstrap[0],
            critic_learning_rate=critic_learning_rate[0], critic_hidden_nodes=critic_hidden_nodes[0],
            n=m, smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        Plot.add_curve(eval_episodes, learning_curve, label=r'n = {}'.format(m))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'n = {}'.format(m))

    Plot.save('reinforce_lunar_lander_n.png')
    Plot2.save('reinforce_lunar_lander_n.png')

    print("n saved")

if args.experiment == -1 or args.experiment == 5:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander Critic hidden nodes (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander Critic hidden nodes')
    for nodes in critic_hidden_nodes:
        print("Critic hidden nodes: ", nodes)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], learning_rate[0], policy_hidden_nodes[0], eta_entropy=entropies[0], baseline_subtraction=baseline_subtraction[0], bootstrap=bootstrap[0],
            critic_learning_rate=critic_learning_rate[0], critic_hidden_nodes=nodes,
            n=n[0], smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        Plot.add_curve(eval_episodes, learning_curve, label=r'Critic hidden nodes = {}'.format(nodes))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Critic hidden nodes = {}'.format(nodes))

    Plot.save('reinforce_lunar_lander_critic_hidden_nodes.png')
    Plot2.save('reinforce_lunar_lander_critic_hidden_nodes_smooth.png')

    print("Critic hidden nodes saved")

if args.experiment == -1 or args.experiment == 6:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander Baseline subtraction (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander Baseline subtraction')
    for base in baseline_subtraction:
        print("Baseline subtraction: ", base)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], learning_rate[0], policy_hidden_nodes[0], eta_entropy=entropies[0], baseline_subtraction=base, bootstrap=bootstrap[0],
            critic_learning_rate=critic_learning_rate[0], critic_hidden_nodes=critic_hidden_nodes[0],
            n=n[0], smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        Plot.add_curve(eval_episodes, learning_curve, label=r'Baseline subtraction = {}'.format(base))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Baseline subtraction = {}'.format(base))

    Plot.save('reinforce_lunar_lander_baseline.png')
    Plot2.save('reinforce_lunar_lander_baseline_smooth.png')

    print("Baseline subtraction saved")

if args.experiment == -1 or args.experiment == 6:
    Plot = LearningCurvePlot(title='Actor Critic Lunar Lander Bootstrapping (raw values)')
    Plot2 = LearningCurvePlot(title='Actor Critic Lunar Lander Bootstrapping')
    for boot in bootstrap:
        print("Bootstrap: ", boot)
        learning_curve, learning_curve_smooth, eval_episodes = call_train_actor_critic(
            1, max_episodes[0], learning_rate[0], policy_hidden_nodes[0], eta_entropy=entropies[0], baseline_subtraction=baseline_subtraction[0], bootstrap=boot,
            critic_learning_rate=critic_learning_rate[0], critic_hidden_nodes=critic_hidden_nodes[0],
            n=n[0], smoothing_window=9, eval_interval=400, env_name="LunarLander-v2")
        Plot.add_curve(eval_episodes, learning_curve, label=r'Bootstrap = {}'.format(boot))
        Plot2.add_curve(eval_episodes, learning_curve_smooth, label=r'Bootstrap = {}'.format(boot))

    Plot.save('reinforce_lunar_lander_bootstrap.png')
    Plot2.save('reinforce_lunar_lander_bootstrap_smooth.png')

    print("Bootstrap saved")
