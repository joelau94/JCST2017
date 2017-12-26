import numpy as np
import tensorflow as tf
import models
import sys
import os
import subprocess

def load_data(index_file,label_file):
	indices = []
	labels = []
	idx = open(index_file,"r")
	lbl = open(label_file,"r")
	for line in idx:
		sentence = map(int,line.split())
		indices.append(sentence)
	for line in lbl:
		sentence = map(int,line.split())
		labels.append(sentence)
	return indices, labels

def train(sess, epoch, model):
	index_file = "{}{}_index.csv".format(data_dir,epoch)
	label_file = "{}{}_label.csv".format(data_dir,epoch)

	indices, labels = load_data(index_file,label_file)
	costs = 0.0

	for i in range(len(indices)):
		cost, prediction, _ = sess.run([model.cost,model.prediction,model.train_op], feed_dict={
			model.input_data: indices[i],
			model.targets: labels[i]
			})
		costs += cost

	return costs

# main
data_dir = str(sys.argv[1])
model_dir = str(sys.argv[2])
vocab_size = int(float(sys.argv[3]))
word_emb_dim = int(float(sys.argv[4]))
hidden_dim = int(float(sys.argv[5]))
nEpochs = int(float(sys.argv[6]))
max_seq_len = int(float(sys.argv[7]))
log_file = str(sys.argv[8])

config = models.Config()
config.vocab_size = vocab_size
config.word_emb_dim = word_emb_dim
config.hidden_dim = hidden_dim
config.max_seq_len = max_seq_len

log = open(log_file,"a+")

with tf.Graph().as_default(), tf.Session() as sess:
	model = models.GEDModel(is_training=True,config=config)
	sess.run(tf.initialize_all_variables())
	
	for i in range(1, nEpochs+1):
		lr_decay = config.lr_decay ** max(i - config.lr_decay_epoch_offset, 0.0)

		sess.run(tf.assign(model.lr, config.learning_rate * lr_decay))

		train_cost = train(sess, i, model)
		model_file = "{}{}.mdl".format(model_dir,i)

		saver = tf.train.Saver()
		saver.save(sess,model_file)
		print("Epoch: %i, Training Cost: %.3f" % (i, train_cost))

		# test after each epoch
		log.write("Testing: test_data={}, vocab_size={}, word_emb_dim={}, hidden_dim={}, max_seq_len={}, model_file={}\n".format(
			data_dir+str(nEpochs-1)+"_index.csv", vocab_size, word_emb_dim, hidden_dim, max_seq_len, model_dir+str(i)+".mdl"))
		log.flush()
		# os.fsync
		subprocess.Popen(["python", "test.py",
			data_dir+str(nEpochs-1)+"_index.csv", data_dir+str(nEpochs-1)+"_label.csv",
			str(vocab_size), str(word_emb_dim), str(hidden_dim), str(max_seq_len),
			model_dir+str(i)+".mdl", log_file])