import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym


class PolicyNetwork(nn.Module):
	def __init__(self, in_dims, out_dims, hidden_size=128):
		super().__init__()  
		self.fc1 = nn.Linear(in_dims, hidden_size)
		self.fc2 = nn.Linear(hidden_size, out_dims)

		# initialize weights from a uniform distribution
		nn.init.uniform_(self.fc1.weight)
		nn.init.uniform_(self.fc2.weight)

	def forward(self, inputs):
		x = F.relu(self.fc1(inputs))
		x = F.relu(self.fc2(x))
		return F.softmax(x, dim=1)
      
	def act(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		probs = self.forward(state).cpu()
		model = Categorical(probs)
		action = model.sample()

		return action.item(), model.log_prob(action), model.entropy()
	

def evaluate(env, policy, n_eval_episodes=10, verbose=False):
	episode_rewards = []
	for _ in range(n_eval_episodes):
		state, _ = env.reset()
		done = False
		total_rewards_ep = 0
		
		while True:
			action, _, _ = policy.act(state)

			if verbose: print(action)

			next_state, reward, done, truncated, _ = env.step(action)
			total_rewards_ep += reward
			
			if done or truncated:
				break
			state = next_state
		episode_rewards.append(total_rewards_ep)
	mean_reward = np.mean(episode_rewards)
	std_reward = np.std(episode_rewards)

	return mean_reward, std_reward


def reinforce(env, policy, optimizer, n_training_episodes, gamma, eval_steps, max_timesteps=10_000, eta_entropy=0.01, verbose=False):
	eval_rewards = []
	eval_episodes = []
	
	for i_episode in range(1, n_training_episodes+1):
		saved_log_probs = []
		rewards = []
		entropies = []
		state, _ = env.reset()
		
		budget = 0
		while True or budget < max_timesteps:
			action, log_prob, entropy = policy.act(state)
			saved_log_probs.append(log_prob)
			state, reward, done, truncated, _ = env.step(action)
			rewards.append(reward)
			entropies.append(entropy)
			if done or truncated:
				break 
			budget += 1
		
		
		returns = deque(maxlen=max_timesteps) 
		T = len(rewards) 

		for t in range(T)[::-1]:
			# calculate the returns from T-1 to 0
			R_t = (returns[0] if len(returns)>0 else 0)
			returns.appendleft(gamma * R_t + rewards[t])    
			
		returns = torch.tensor(returns)
		
		# sum^{T-1}_{t=0} R_t * log \pi_\theta
		policy_loss = []

		for log_prob, R, entropy in zip(saved_log_probs, returns, entropies):
			# vanilla policy loss
			loss = -log_prob * R

			# entropy regularization
			loss -= entropy * eta_entropy

			policy_loss.append(loss)
			
		policy_loss = torch.stack(policy_loss)
		policy_loss = torch.sum(policy_loss, dim=0)

		# apply gradients
		optimizer.zero_grad()
		nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1)
		policy_loss.backward()
		optimizer.step()
		
		if i_episode % eval_steps == 0:
			mean_reward, _ = evaluate(env, policy)
			eval_rewards.append(mean_reward)
			eval_episodes.append(i_episode)

			if verbose:
				print("Episode", i_episode, "\tAverage Score:", mean_reward)
		
	return eval_rewards, eval_episodes


if __name__ == "__main__":
	# Create the environment
    env_name = "LunarLander-v2"
    env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
	
	# Set the hyperparameters
    hidden_nodes = 64
    n_training_episodes = 10_000
    max_timesteps = 1_000
    gamma = .99
    learning_rate = 0.001
    entropy = 0.001

    # Policy network
    policy = PolicyNetwork(state_size, action_size, hidden_nodes)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)	

    # Run the reinforce algorithm
    scores, eval_episodes = reinforce(env,
					policy,
                    optimizer,
                    n_training_episodes, 
                    gamma, 
                    100,
                    max_timesteps,
                    entropy,
                    verbose=True)
	
    # print(scores, eval_episodes)
	
    env.close()