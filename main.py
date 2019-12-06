import tensorflow as tf
from Agent import Agent
from Memory import Memory
from Model import Model
from utils import save, load
import sys
import time
import numpy as np
import datetime
from websocket_server import WebsocketServer
import socket


MAX_EPSILON = 1
MIN_EPSILON = 0.001
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50
MEMORY_SIZE = 50000

folder = datetime.datetime.now().strftime("%d-%m-%Y - %Hh %Mm %Ss")
writer = tf.summary.create_file_writer("results/" + folder)

num_states = 6
num_actions = 5

model = Model(num_actions, num_states, BATCH_SIZE)
mem = Memory(MEMORY_SIZE)
ag = Agent(model, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA, GAMMA)

for i, arg in enumerate(sys.argv):
	if arg == "load":
		print("Loading model {}".format(sys.argv[i + 1]))
		load(model.model, sys.argv[i + 1])
	elif arg == "eps":
		eps = float(sys.argv[i + 1])
		ag.max_eps = eps


cnt = 0
action = 0
starttime = time.time() * 1000
times = []


def new_msg(client, server, message):
	global cnt, starttime
	# perp data
	next_state, done = message.split(':')
	next_state = np.array([float(d) for d in next_state.split[',']])
	done = bool(done)

	# ask update
	action = ag.update(next_state, done)

	# send action
	client.send_message(str(action))
	if done:
		print("Step {0}, Total reward: {1}, Eps: {2:.4f}, time: {3:.2f}ms".format(ag._steps, ag._tot_reward, ag.eps, np.mean(times)))
		ag.reset()
		tf.summary.scalar("max_dist", ag.max_dist, step=cnt)
		tf.summary.scalar("eps", ag.eps, step=cnt)
		tf.summary.scalar("reward", ag.reward, step=cnt)
		tf.summary.scalar("Time", np.mean(times), step=cnt)
		writer.flush()
		cnt += 1
		save(model.model)
		return
	ag._replay()
	currenttime = time.time() * 1000
	times.append(currenttime - starttime)
	starttime = currenttime


def new_client(client, server):
	print("New client")
	client.send_message(str(action))


with writer.as_default():
	print("Server's IP Address is:" + socket.gethostbyname(socket.gethostname()))
	server = WebsocketServer(13254, host='0.0.0.0')
	server.set_fn_message_received(new_msg)
	server.set_fn_new_client(new_client)
	server.run_forever()
