import datetime


def save(model):
	time = datetime.datetime.now().strftime("%d-%m-%Y - %Hh %Mm %Ss")
	path = './checkpoints/UT_Bot_IA'
	format = '.h5'
	model.save_weights(path + " " + time + format)
	print("Model saved ")
