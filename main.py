import tensorflow as tf
from GameRunner import GameRunner
from Memory import Memory
from Model import Model
import gym
import matplotlib.pylab as plt

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50
MEMORY_SIZE = 50000


tf.compat.v1.disable_eager_execution()
env_name = 'MountainCar-v0'
env = gym.make(env_name)

num_states = env.env.observation_space.shape[0]
num_actions = env.env.action_space.n

model = Model(num_states, num_actions, BATCH_SIZE)
mem = Memory(MEMORY_SIZE)

with tf.compat.v1.Session() as sess:
	sess.run(model.var_init)
	gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA, GAMMA, False)
	num_episodes = 300
	cnt = 0
	while cnt < num_episodes:
		print('Episode {} of {}'.format(cnt + 1, num_episodes))
		gr.run()
		cnt += 1
	plt.plot(gr.reward_store)
	plt.show()
	plt.close("all")
	plt.plot(gr.max_x_store)
	plt.show()
