import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym

from Helper import LearningCurvePlot, smooth
import time

env_name = "LunarLander-v2"

env = gym.make(env_name, render_mode="human")
eval_env = gym.make(env_name)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

n = 1
print("state size:", state_size)
print("action size:", action_size)

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
	
class Critic(nn.Module):
	def __init__(self, input_dim, learning_rate):
		super(Critic, self).__init__()

		self.critic = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)

		# self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		# self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		# self.to(self.device)

	def forward(self, state):
		value = self.critic(state)

		return value

	def squared_loss(self, states, Q_sa):
		V = self.forward(torch.tensor(states))

		# state = trace[t][0]
		# reward = trace[t][2]
		# next_state = trace[t+1][0]

		# value = agent.critic(torch.tensor(state))
		# next_value = agent.critic(torch.tensor(next_state))

		# td_target = reward + gamma * next_value

		# td_error = td_target - value

		critic_loss = torch.nn.functional.mse_loss(V, Q_sa)

		loss = pow(V - Q_sa, 2)

		# loss = torch.stack(loss)
		loss = torch.sum(loss, dim=0)

		# critic_loss = torch.stack(critic_loss)

		return loss
	
def evaluate(policy, n_eval_episodes=10, verbose=False):
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
def actor_critic(policy, val_func, optimizer, n_training_episodes, gamma, eval_steps, max_timesteps=10_000, eta_entropy=0.01, verbose=False):
	eval_rewards = []
	eval_episodes = []
	
	for i_episode in range(1, n_training_episodes+1):
		saved_log_probs = []
		states = []
		rewards = []
		entropies = []
		state, _ = env.reset()
		
		budget = 0

		#Sample trace
		while True or budget < max_timesteps:
			action, log_prob, entropy = policy.act(state)
			saved_log_probs.append(log_prob)
			state, reward, done, truncated, _ = env.step(action)
			states.append(state)
			rewards.append(reward)
			entropies.append(entropy)
			if done or truncated:
				break 
			budget += 1
		
		
		returns = deque(maxlen=max_timesteps) 
		T = len(rewards) 

		Q_sa = np.zeros(len(rewards))

		for t in range(T-1):
			# calculate the returns from T-1 to 0
			# R_t = (returns[0] if len(returns)>0 else 0)
			# returns.appendleft(gamma * R_t + rewards[t]) 
			# print(Q_sa[t])
			# print(rewards[t:t+n])
			# print(states[t+n])
			if t + n >= T:
				n2 = T - t
				Q_sa[t] = sum(rewards[t:t+n2])
			else:
				Q_sa[t] = sum(rewards[t:t+n]) + val_func.critic(torch.tensor(states[t+n]))
			
		Q_sa = torch.tensor(Q_sa)
		
		critic_loss = torch.tensor(val_func.squared_loss(states, Q_sa))
		print(critic_loss)
		print(type(critic_loss))

		# apply gradients
		optimizer.zero_grad()
		nn.utils.clip_grad_norm_(val_func.parameters(), max_norm=1)
		critic_loss.backward()
		optimizer.step()

		# sum^{T-1}_{t=0} R_t * log \pi_\theta
		policy_loss = []

		for log_prob, Q_sa, entropy in zip(saved_log_probs, Q_sa, entropies):
			# vanilla policy loss
			loss = -log_prob * Q_sa

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
			mean_reward, std_reward = evaluate(policy)
			eval_rewards.append(mean_reward)
			eval_episodes.append(i_episode)

			if verbose:
				print("Episode", i_episode, "\tAverage Score:", mean_reward)
		
	return eval_rewards, eval_episodes

# hyperparameters
hidden_nodes = 64
n_training_episodes = 10_000 #10000
max_timesteps = 10_000
gamma = .99
learning_rate = 1e-3
entropy = 0.01

def average_over_repetitions(n_repetitions, n_training_episodes, max_episode_length, learning_rate, hidden_nodes, gamma,
                            eta_entropy, smoothing_window=None, eval_interval=100):

	returns_over_repetitions = []
	now = time.time()
	for _ in range(n_repetitions): 
		policy = PolicyNetwork(state_size, action_size, hidden_nodes)
		val_func = Critic(state_size, learning_rate)
		optimizer = optim.Adam(policy.parameters(), lr=learning_rate)	
		
		eval_rewards, eval_episodes = actor_critic(policy,
							val_func,
							optimizer,
							n_training_episodes, 
							gamma, 
							eval_interval,
							max_episode_length,
							eta_entropy,
							verbose=True)

		returns_over_repetitions.append(eval_rewards)
		
	print('Running one setting takes {} minutes'.format((time.time()-now)/60))
	learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
	if smoothing_window is not None: 
		learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
	return learning_curve, eval_episodes

hidden_nodes = 64
n_training_episodes = 10_000
max_timesteps = 1_000
gamma = .99
learning_rate = 0.001

Plot = LearningCurvePlot(title = 'Actor Critic Lunar Lander')    

entropies = [0.01, 0.001]
learning_rates = [0.01, 0.001] #TODO:
for entropy in entropies:        
	learning_curve, eval_episodes = average_over_repetitions(
      	1, 
		n_training_episodes, 
		max_timesteps, 
		learning_rate,
		hidden_nodes,
		gamma,
		entropy, 
		smoothing_window=None, 
		eval_interval=100)
	Plot.add_curve(eval_episodes, learning_curve, label=r'Entropy, $\eta$ = {}'.format(entropy))  

Plot.save('reinforce_lunar_lander.png')