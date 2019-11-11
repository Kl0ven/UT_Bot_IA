import random
import math
import numpy as np


class Agent:
	def __init__(self, model, env, memory, max_eps, min_eps,
														decay, gamma, render=True):
		self._env = env
		self._model = model
		self._memory = memory
		self._render = render
		self._max_eps = max_eps
		self._min_eps = min_eps
		self._decay = decay
		self._eps = self._max_eps
		self._steps = 0
		self._gamma = gamma
		self._reward_store = []
		self._max_x_store = []
		self._eps_store = []
		self._prev_state = None
		self._tot_reward = 0
		self._max_x = -100
		self._prev_action = None
		self._prev_reward = None
		self._min_reward = -200
		self._max_reward = 200

	def update(self, state, reward, done):
		if self._render:
			self._env.render()

		action = self._choose_action(state)
		reward = self.compute_reward(state, reward)

		if state[0] > self._max_x:
			self._max_x = state[0]

		if done:
			state = None

		if self._prev_state is not None and self._prev_action is not None and self._prev_reward is not None:
			self._memory.add_sample((self._prev_state, self._prev_action, self._prev_reward, state))

		self.decay()
		self._prev_state = state
		self._prev_action = action
		self._prev_reward = reward
		self._tot_reward += reward
		return action

	def compute_reward(self, next_state, reward):
		if next_state[0] >= 0.1:
			reward += 10
		elif next_state[0] >= 0.25:
			reward += 20
		elif next_state[0] >= 0.5:
			reward += 100
		return reward

	def reset(self):
		print("Step {}, Total reward: {}, Eps: {}".format(self._steps, self._tot_reward, self._eps))
		self._reward_store.append(self._tot_reward)
		self._max_x_store.append(self._max_x)
		self._prev_state = self._env.reset()
		self._tot_reward = 0
		self._max_x = -100

	def decay(self):
		self._steps += 1
		self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.exp(-self._decay * self._steps)
		self.eps_store.append(self._eps)

	def _choose_action(self, state):
		if random.random() < self._eps:
			return random.randint(0, self._model.num_actions - 1)
		else:
			return np.argmax(self._model.predict_one(state))

	def _replay(self):
		batch = self._memory.sample(self._model.batch_size)
		if len(batch) == 0:
			return
		states = np.array([val[0] for val in batch])
		next_states = np.array([(np.zeros(self._model.num_states) if val[3] is None else val[3]) for val in batch])
		# predict Q(s,a) given the batch of states
		q_s_a = self._model.predict_batch(states)
		# predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
		q_s_a_d = self._model.predict_batch(next_states)
		# setup training arrays
		x = np.zeros((len(batch), self._model.num_states))
		y = np.zeros((len(batch), self._model.num_actions))
		for i, b in enumerate(batch):
			state, action, reward, next_state = b[0], b[1], b[2], b[3]
			# get the current q values for all actions in state
			current_q = q_s_a[i]
			# update the q value for action
			if next_state is None:
				# in this case, the game completed after action, so there is no max Q(s',a')
				# prediction possible
				current_q[action] = reward
			else:
				current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])
			x[i] = state
			y[i] = current_q
		self._model.train_batch(x, y)

	@property
	def reward_store(self):
		return self._reward_store

	@property
	def max_x(self):
		return self._max_x_store[-1]

	@property
	def reward(self):
		return self._reward_store[-1]

	@property
	def max_x_store(self):
		return self._max_x_store

	@property
	def eps_store(self):
		return self._eps_store

	@property
	def max_eps(self):
		return self._max_eps

	@max_eps.setter
	def max_eps(self, value):
		self._max_eps = value
