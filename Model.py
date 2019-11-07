import tensorflow as tf


class Model:
	def __init__(self, num_states, num_actions, batch_size):
		self._num_states = num_states
		self._num_actions = num_actions
		self._batch_size = batch_size
		# define the placeholders
		self._states = None
		self._actions = None
		# the output operations
		self._logits = None
		self._optimizer = None
		self._var_init = None
		# now setup the model
		self._define_model()

	def _define_model(self):
		self._states = tf.compat.v1.placeholder(shape=[None, self._num_states], dtype=tf.float32)
		self._q_s_a = tf.compat.v1.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
		# create a couple of fully connected hidden layers
		fc1 = tf.compat.v1.layers.dense(self._states, 50, activation=tf.nn.relu)
		fc2 = tf.compat.v1.layers.dense(fc1, 50, activation=tf.nn.relu)
		self._logits = tf.compat.v1.layers.dense(fc2, self._num_actions)
		loss = tf.compat.v1.losses.mean_squared_error(self._q_s_a, self._logits)
		self._optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)
		self._var_init = tf.compat.v1.global_variables_initializer()

	def predict_one(self, state, sess):
		return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

	def predict_batch(self, states, sess):
		return sess.run(self._logits, feed_dict={self._states: states})

	def train_batch(self, sess, x_batch, y_batch):
		sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

	@property
	def num_states(self):
		return self._num_states

	@property
	def num_actions(self):
		return self._num_actions

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def var_init(self):
		return self._var_init
