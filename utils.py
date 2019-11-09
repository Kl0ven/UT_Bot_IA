
def save(saver, sess):
	save_path = saver.save(sess, "./model.ckpt")
	print("Model saved in path: %s" % save_path)
