import math
import random
import numpy as np
import tensorflow as tf
import models
import sys
import os
import subprocess
import datetime

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

	random.shuffle(shuffled_order)
	for i in shuffled_order:		
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
datasets = int(float(sys.argv[6]))
sent_num = int(float(sys.argv[7]))
max_seq_len = int(float(sys.argv[8]))
log_file = str(sys.argv[9])

config = models.Config()
config.vocab_size = vocab_size
config.word_emb_dim = word_emb_dim
config.hidden_dim = hidden_dim
config.max_seq_len = max_seq_len

log = open(log_file,"a+")

datasets = 51
max_it = 1000000
rel_tol = 0.001
shuffled_order = [i for i in range(0,sent_num)]
data_order = [i for i in range(1,datasets)]

with tf.Graph().as_default(), tf.Session() as sess:
	model = models.GEDModel(is_training=True,config=config)
	sess.run(tf.initialize_all_variables())

	costs = 0.0
	new_costs = 0.0
	iteration = 0

	while (costs == 0.0 or math.fabs(costs - new_costs) >= rel_tol * costs) and iteration < max_it:
		iteration += 1
		costs = new_costs
		new_costs = 0.0

		log.write("Iteration: %i begins (%s)\n" % (iteration, str(datetime.datetime.now())))
		log.flush()

		random.shuffle(data_order)
		count = 0
		for i in data_order:
			count += 1
			lr_decay = config.lr_decay ** max(i - config.lr_decay_epoch_offset, 0.0)

			sess.run(tf.assign(model.lr, config.learning_rate * lr_decay))

			new_costs += train(sess, i, model)
			model_file = "{}{}.mdl".format(model_dir,iteration)

			saver = tf.train.Saver()
			saver.save(sess,model_file)
			log.write("Iter: %i (No.%i), Dataset: %i, Training Cost: %.3f (%s)\n" % (iteration, count, i, new_costs, str(datetime.datetime.now())))
			log.flush()

			# test after each epoch
			subprocess.Popen(["python", "test.py",
				data_dir+str(datasets-1)+"_index.csv", data_dir+str(datasets-1)+"_label.csv",
				str(vocab_size), str(word_emb_dim), str(hidden_dim), str(max_seq_len),
				model_dir+str(iteration)+".mdl", log_file])
