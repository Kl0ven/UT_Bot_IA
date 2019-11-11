import datetime
from os import path


def save(model):
	time = datetime.datetime.now().strftime("%d-%m-%Y - %Hh %Mm %Ss")
	location = './checkpoints/UT_Bot_IA'
	format = '.h5'
	model.save_weights(location + " " + time + format)
	print("Model saved ")


def load(model, location):
	print(location)
	assert(path.exists(location))
	model.load_weights(location)
	print('CKPT loaded')
