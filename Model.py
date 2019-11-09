import tensorflow as tf
import tensorflow.keras.layers as kl


class ProbabilityDistribution(tf.keras.Model):
	def call(self, logits):
		# sample a random categorical action from given logits
		return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
	def __init__(self, num_actions, num_states, batch_size):
		super().__init__('mlp_policy')
		self._num_states = num_states
		self._num_actions = num_actions
		self._batch_size = batch_size
		# no tf.get_variable(), just simple Keras API
		self.hidden1 = kl.Dense(50, activation='relu')
		self.hidden2 = kl.Dense(50, activation='relu')
		self.value = kl.Dense(1, name='value')
		# logits are unnormalized log probabilities
		self.logits = kl.Dense(num_actions, name='policy_logits')
		self.dist = ProbabilityDistribution()

	def call(self, inputs):
		# inputs is a numpy array, convert to Tensor
		x = tf.convert_to_tensor(inputs)
		# separate hidden layers from the same input tensor
		hidden_logs = self.hidden1(x)
		hidden_vals = self.hidden2(x)
		return self.logits(hidden_logs), self.value(hidden_vals)

	def predict_one(self, state):
		return self.predict(state)

	def predict_batch(self, states):
		return self.predict(states)

	def train_batch(self, x_batch, y_batch):
		print(x_batch, type(x_batch), y_batch, type(y_batch))
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
