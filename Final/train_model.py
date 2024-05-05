import numpy as np
from collections import deque
import torch
import torch.optim as optim
import gym

from evaluate import evaluate


def train_model(policy, n_training_episodes, eval_steps, env, eval_env, val_func=None,
                bootstrap=False, baseline_subtraction=False, gamma=0.99, n=1, eta_entropy=0.01, verbose=True):
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
        done = False
        while not done or budget < max_n_timesteps:
            action, log_prob, entropy = policy.act(state)
            saved_log_probs.append(log_prob)
            entropies.append(entropy)

            next_state, reward, done, truncated, _ = env.step(action)
            states.append(state)
            rewards.append(reward)

            if done or truncated:
                break
            state = next_state
            budget += 1

        returns = deque(maxlen=max_n_timesteps)
        T = len(rewards)
        begin = 0

        # bootstrap
        if bootstrap:
            # calculate the returns from T-1 to 0
            for t in range(T - n):
                future_state = torch.from_numpy(np.array(states[t + n])).float().unsqueeze(0)
                val = val_func.critic(future_state) * pow(gamma, n)
                val = sum(gamma ** i * rewards[t + i] for i in range(n)) + val
                returns.append([val])
            begin = T - n

        val = 0
        returns2 = deque(maxlen=n)
        for t in range(T - 1, begin - 1, -1):
            val = gamma * val + rewards[t]
            returns2.appendleft([val])
        returns += returns2

        # baseline subtraction
        if baseline_subtraction:
            for i in range(T):
                returns[i][0] -= val_func.critic(torch.from_numpy(np.array(states[i])).float().unsqueeze(0))

        returns_tensor = torch.tensor(returns)
        # returns is either discounted returns, Q_sa or A_sa

        # if critic, train critic
        if val_func is not None and (bootstrap or baseline_subtraction):
            val_func.train(states, returns_tensor)

        # call policy train
        policy.train(saved_log_probs, returns_tensor, entropies, eta_entropy)

        # call evaluate if eval_steps interval reached
        if i_episode % eval_steps == 0:
            mean_reward, std_reward = evaluate(policy, eval_env)
            eval_rewards.append(mean_reward)
            eval_episodes.append(i_episode)

            if verbose:
                print("Episode", i_episode, "\tAverage Score:", mean_reward)

    return eval_rewards, eval_episodes
