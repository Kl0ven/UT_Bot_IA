import tensorflow as tf
from Agent import Agent
from Memory import Memory
from Model import Model
import gym
import matplotlib.pylab as plt
from utils import save, load
import sys
import time
import numpy as np
import datetime

# For tensorboard to work you need to enable eager mode
# But with eager mode enable the learning is slower due to some bug
# https://github.com/tensorflow/tensorflow/issues/33052
# tf.compat.v1.disable_eager_execution()

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 1
MEMORY_SIZE = 50000

folder = datetime.datetime.now().strftime("%d-%m-%Y - %Hh %Mm %Ss")
writer = tf.summary.create_file_writer("results/" + folder)

env_name = 'MountainCar-v0'
env = gym.make(env_name)

num_states = env.env.observation_space.shape[0]
num_actions = env.env.action_space.n

model = Model(num_actions, num_states, BATCH_SIZE)
mem = Memory(MEMORY_SIZE)
ag = Agent(model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA, GAMMA, False)

for i, arg in enumerate(sys.argv):
	if arg == "load":
		print("Loading model {}".format(sys.argv[i + 1]))
		load(model, sys.argv[i + 1])
	elif arg == "eps":
		eps = float(sys.argv[i + 1])
		ag.max_eps = eps

num_episodes = 300
cnt = 0
action = ag._choose_action(env.reset())
starttime = time.time() * 1000
times = []
debug_times = []
with writer.as_default():
	while cnt < num_episodes:
		if cnt % 10 == 0:
			print('Episode {} of {}'.format(cnt + 1, num_episodes))
		times = []
		while True:
			next_state, reward, done, info = env.step(action)
			action = ag.update(next_state, reward, done)
			if done:
				debug_times = np.array(debug_times)
				print(debug_times.shape)
				print("Step {0}, Total reward: {1}, Eps: {2:.4f}, time: {3:.2f}ms timing {4}".format(ag._steps, ag._tot_reward, ag._eps, np.mean(times), debug_times.mean(axis=0).astype(int)))
				debug_times = []
				ag.reset()
				break
			t = ag._replay()
			if t is not None:
				debug_times.append(t)
			currenttime = time.time() * 1000
			times.append(currenttime - starttime)
			starttime = currenttime
		tf.summary.scalar("max_x", ag.max_x, step=cnt)
		tf.summary.scalar("reward", ag.reward, step=cnt)
		tf.summary.scalar("Time", np.mean(times), step=cnt)
		writer.flush()
		cnt += 1
save(model)

plt.subplot(311)
plt.plot(ag.reward_store)
plt.subplot(312)
plt.plot(ag.max_x_store)
plt.subplot(313)
plt.plot(ag.eps_store)
plt.show()
