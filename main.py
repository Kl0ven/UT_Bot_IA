import tensorflow as tf
from GameRunner import GameRunner
from Memory import Memory
from Model import Model
import gym
import matplotlib.pylab as plt


tf.compat.v1.disable_eager_execution()

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50
MEMORY_SIZE = 50000

env_name = 'MountainCar-v0'
env = gym.make(env_name)

num_states = env.env.observation_space.shape[0]
num_actions = env.env.action_space.n

model = Model(num_actions, num_states, BATCH_SIZE)
mem = Memory(MEMORY_SIZE)
gr = GameRunner(model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA, GAMMA, True)


num_episodes = 300
cnt = 0
action = gr._choose_action(env.reset())
while cnt < num_episodes:
	if cnt % 10 == 0:
		print('Episode {} of {}'.format(cnt + 1, num_episodes))
	while True:
		next_state, reward, done, info = env.step(action)
		action = gr.update(next_state, reward, done)
		if done:
			gr.reset()
			break
		gr._replay()

	cnt += 1
plt.plot(gr.reward_store)
plt.show()
plt.close("all")
plt.plot(gr.max_x_store)
plt.show()
