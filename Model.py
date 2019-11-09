import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np


class Model(tf.keras.Model):
	def __init__(self, num_actions, num_states, batch_size):
		super().__init__('mlp_policy')
		self._num_states = num_states
		self._num_actions = num_actions
		self._batch_size = batch_size
		# no tf.get_variable(), just simple Keras API
		self.hidden1 = kl.Dense(50, activation='relu', name='h1')
		self.hidden2 = kl.Dense(50, activation='relu', name='h2')
		# logits are unnormalized log probabilities
		self.logits = kl.Dense(num_actions, name='policy_logits')
		self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
		self.build((None, num_states))
		self.summary(100)

	def call(self, inputs):
		# inputs is a numpy array, convert to Tensor
		x1 = self.hidden1(inputs)
		x2 = self.hidden2(x1)
		return self.logits(x2)

	def predict_one(self, state):
		state = np.array([state])
		return self.predict(state)

	def predict_batch(self, states):
		return self.predict(states)

	def train_batch(self, x_batch, y_batch):
		self.train_on_batch(x=x_batch, y=y_batch)

	@property
	def num_actions(self):
		return self._num_actions

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def num_states(self):
		return self._num_states
