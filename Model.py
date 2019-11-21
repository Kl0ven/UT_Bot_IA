import tensorflow as tf
# import tensorflow.keras.layers as kl
# import numpy as np


class Model(tf.keras.Model):
	def __init__(self, num_actions, num_states, batch_size):
		super().__init__('mlp_policy')
		self._num_states = num_states
		self._num_actions = num_actions
		self._batch_size = batch_size
		# Setting up model
		self.model = tf.keras.models.Sequential()
		self.model.add(tf.keras.layers.Dense(50, activation='relu', name='h1'))
		self.model.add(tf.keras.layers.Dense(50, activation='relu', name='h2'))
		self.model.add(tf.keras.layers.Dense(num_actions, name='output'))
		self.model.build((None, self._num_states))
		# The loss method
		self.loss_object = tf.keras.losses.MeanSquaredError()
		# The optimize
		self.optimizer = tf.keras.optimizers.Adam()
		# This metrics is used to track the progress of the training loss during the training
		self.train_loss = tf.keras.metrics.Mean(name='train_loss')

	@tf.function
	def train_batch(self, x, y):
		with tf.GradientTape() as tape:
			# Make a prediction
			predictions = self.model(x)
			# Get the error/loss using the loss_object previously defined
			loss = self.loss_object(y, predictions)
		# Compute the gradient which respect to the loss
		gradients = tape.gradient(loss, self.model.trainable_variables)
		# Change the weights of the model
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		# The metrics are accumulate over time. You don't need to average it yourself.
		self.train_loss(loss)

	@tf.function
	def predict_one(self, x):
		return self.model(x)

	@tf.function
	def predict_batch(self, x):
		return self.model(x)

	@property
	def num_actions(self):
		return self._num_actions

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def num_states(self):
		return self._num_states
