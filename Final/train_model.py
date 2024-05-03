import numpy as np
from collections import deque
import torch
import torch.optim as optim
import gym

from evaluate import evaluate


def train_model(policy, n_training_episodes, eval_steps, env, eval_env, val_func=None,
                bootstrap=False, baseline_subtraction=False, gamma=0.99, n=1, eta_entropy=0.01, verbose=False):
    """
    Trains the model.
    takes policy and return eval_rewards and eval_episodes lists.
    Can be used for both actor and critic.
    """

    # max environment number of timesteps
    max_n_timesteps = 10_000

    eval_rewards = []
    eval_episodes = []

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        states = []
        rewards = []
        entropies = []
        state, _ = env.reset()

        budget = 0
        while True or budget < max_n_timesteps:
            states.append(state)
            action, log_prob, entropy = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            entropies.append(entropy)
            if done or truncated:
                break
            budget += 1

        returns = deque(maxlen=max_n_timesteps)
        T = len(rewards)

        val = 0
        # calculate the returns from T-1 to 0
        for t in range(T)[::-1]:
            if bootstrap and val_func is not None:
                val = 0
                if t + n >= T:
                    n2 = T - t
                    for i in range(n2):
                        val += rewards[t + i] * pow(gamma, i)
                    # since our episodes always have a terminal state, dont use critic at final state
                else:
                    for i in range(n):
                        val += rewards[t + i] * pow(gamma, i)
                    val += val_func.critic(torch.from_numpy(np.array(states[t + n])).float().unsqueeze(0)) * pow(gamma,
                                                                                                                 n)
            else:
                # calculate the returns from T-1 to 0
                val = gamma * val + rewards[t]

            # baseline subtraction
            if baseline_subtraction and val_func is not None:
                val -= val_func.critic(torch.from_numpy(np.array(states[t])).float().unsqueeze(0))
            returns.appendleft([val])



        returns = torch.tensor(returns)
        # returns is either discounted returns, Q_sa or A_sa

        # if critic, train critic
        if val_func is not None and (bootstrap or baseline_subtraction):
            val_func.train(states, returns)

        # call policy train
        policy.train(saved_log_probs, returns, entropies, eta_entropy)


        # call evaluate if eval_steps interval reached
        if i_episode % eval_steps == 0:
            mean_reward, std_reward = evaluate(policy, eval_env)
            eval_rewards.append(mean_reward)
            eval_episodes.append(i_episode)

            if verbose:
                print("Episode", i_episode, "\tAverage Score:", mean_reward)

    return eval_rewards, eval_episodes
