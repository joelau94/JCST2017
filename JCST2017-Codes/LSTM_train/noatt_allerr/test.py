import numpy as np
import tensorflow as tf
import models
import sys
import os

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

# main
index_file = str(sys.argv[1])
label_file = str(sys.argv[2])
vocab_size = int(float(sys.argv[3]))
word_emb_dim = int(float(sys.argv[4]))
hidden_dim = int(float(sys.argv[5]))
max_seq_len = int(float(sys.argv[6]))
model_file = str(sys.argv[7])
log_file = str(sys.argv[8])

config = models.Config()
config.vocab_size = vocab_size
config.word_emb_dim = word_emb_dim
config.hidden_dim = hidden_dim
config.max_seq_len = max_seq_len

log = open(log_file,"a+")

gold_std = 0	# number of errors in golden standard
recall_count = 0	# number of errors in golden standard hit by model
precision_count = 0	# number of errors reported by model correctly
err_report = 0	# number of errors reported by model

with tf.Graph().as_default(), tf.Session() as sess:
	model = models.GEDModel(is_training=False,config=config)
	tf.train.Saver().restore(sess, model_file)

	indices, labels = load_data(index_file,label_file)

	for i in range(len(indices)):
		cost, prediction, _ = sess.run([model.cost,model.prediction,model.train_op], feed_dict={
			model.input_data: indices[i],
			model.targets: labels[i]
			})
		
		# calculate f_0.5
		for j in range(len(labels[i])):
			if labels[i][j] == 0:
				gold_std += 1
				if prediction[j] <= 0.5:
					recall_count += 1
			if prediction[j] <= 0.5:
				err_report += 1
				if labels[i][j] == 0:
					precision_count += 1


recall = float(recall_count) / float(gold_std)
precision = float(precision_count) / float(err_report)
f_0_5 = float( ( 1 + 0.25 ) * recall * precision ) / float( recall + 0.25 * precision )

log.write("recall = {}\n".format(recall))
log.write("precision = {}\n".format(precision))
log.write("f_0_5 = {}\n".format(f_0_5))
log.flush()
os.fsync(log)
log.close()
