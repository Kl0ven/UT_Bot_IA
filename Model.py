import tensorflow as tf
# import tensorflow.keras.layers as kl
# import numpy as np

num_actions = 3
# Setting up model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu', name='h1'))
model.add(tf.keras.layers.Dense(50, activation='relu', name='h2'))
model.add(tf.keras.layers.Dense(num_actions, name='output'))

# The loss method
loss_object = tf.keras.losses.MeanSquaredError()
# The optimize
optimizer = tf.keras.optimizers.Adam()
# This metrics is used to track the progress of the training loss during the training
train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def train_batch(x, y):
	global model, loss_object, optimizer, train_loss
	with tf.GradientTape() as tape:
		# Make a prediction
		predictions = model(x)
		# Get the error/loss using the loss_object previously defined
		loss = loss_object(y, predictions)
	# Compute the gradient which respect to the loss
	gradients = tape.gradient(loss, model.trainable_variables)
	# Change the weights of the model
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	# The metrics are accumulate over time. You don't need to average it yourself.
	train_loss(loss)


@tf.function
def predict_one(x):
	global model
	return model(x)


@tf.function
def predict_batch(x):
	global model
	return model(x)

# class Model(tf.keras.Model):
# 	def __init__(self, num_actions, num_states, batch_size):
# 		super().__init__('mlp_policy')
# 		self._num_states = num_states
# 		self._num_actions = num_actions
# 		self._batch_size = batch_size
# 		# no tf.get_variable(), just simple Keras API
# 		self.hidden1 = kl.Dense(50, activation='relu', name='h1')
# 		self.hidden2 = kl.Dense(50, activation='relu', name='h2')
# 		# logits are unnormalized log probabilities
# 		self.logits = kl.Dense(num_actions, name='output')
# 		self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
# 		self.build((None, num_states))
# 		self.summary(100)
#
# 	def call(self, inputs):
# 		# inputs is a numpy array, convert to Tensor
# 		x1 = self.hidden1(inputs)
# 		x2 = self.hidden2(x1)
# 		return self.logits(x2)
#
# 	def predict_one(self, state):
# 		state = np.array([state])
# 		return self.predict(state)
#
# 	def predict_batch(self, states):
# 		return self.predict(states)
#
# 	def train_batch(self, x_batch, y_batch):
# 		return self.train_on_batch(x=x_batch, y=y_batch)
#
# 	@property
# 	def num_actions(self):
# 		return self._num_actions
#
# 	@property
# 	def batch_size(self):
# 		return self._batch_size
#
# 	@property
# 	def num_states(self):
# 		return self._num_states
