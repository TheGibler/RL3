import numpy as np


def evaluate(policy, eval_env, n_eval_episodes=10, verbose=False):
	"""
	Evaluation function. Used for plotting and evaluating algorithm during and after training.
	Runs actor through n_eval_episodes and averages rewards. returns means and standard deviations.
	"""

	episode_rewards = []
	for _ in range(n_eval_episodes):
		state, _ = eval_env.reset()
		done = False
		total_rewards_ep = 0
		
		while True:
			action, _, _ = policy.act(state)

			if verbose: print(action)

			next_state, reward, done, truncated, _ = eval_env.step(action)
			total_rewards_ep += reward
			
			if done or truncated:
				break
			state = next_state
		episode_rewards.append(total_rewards_ep)
	mean_reward = np.mean(episode_rewards)
	std_reward = np.std(episode_rewards)

	return mean_reward, std_reward




