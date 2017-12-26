import numpy as np
import tensorflow as tf

def masked_softmax(scores, mask):
	"""
	scores: tensor of shape[1,s]
	mask: tensor of shape[1,s]
	"""
	exp_s = tf.exp(scores)
	exp_s = exp_s * mask
	sum_exp_s = tf.reduce_sum(exp_s,[1])
	softmax_result = exp_s / sum_exp_s
	# shape (1, s)
	return softmax_result

class Encoder(object):

	def __init__(self, hidden_dim, scope=None):
		self.hidden_dim = hidden_dim
		if scope:
			self.scope=scope
		else:
			self.scope=tf.variable_scope('Encoder')

	def __call__(self, inputs, prev_memory_state, prev_hidden_state):
		# prev_memory_state: shape (hidden_dim,)
		# prev_hidden_state: shape (hidden_dim,)
		lstm_input = tf.concat(1, [inputs, prev_hidden_state])
		# shape (1, hidden_dim + word_emb_dim)
		with tf.variable_scope("fc_activs"):
			pre_lstm_activs = tf.contrib.layers.fully_connected(inputs=lstm_input,
																num_outputs=4*self.hidden_dim,
																biases_initializer=tf.constant_initializer(0.0))
			# shape (1, 4*hidden_dim)

		i, f, o, proposed_c = tf.split(1, 4, pre_lstm_activs)
		# shape (1, hidden_dim)

		new_c = (prev_memory_state * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(proposed_c))
		new_h = tf.tanh(new_c) * tf.sigmoid(o)

		return new_c, new_h

class Classifier(object):

	def __init__(self, word_emb_dim, hidden_dim, max_seq_len, scope=None):
		self.hidden_dim = hidden_dim
		self.word_emb_dim = word_emb_dim
		self.max_seq_len = max_seq_len
		if scope:
			self.scope=scope
		else:
			self.scope=tf.variable_scope('Classifier')

	def __call__(self, decoder_input, final_state):
		"""
		decoder_input: shape (1,word_emb_dim) [input]
		final_state: shape (hidden_dim,1)
		"""

		hidden_dim = self.hidden_dim
		word_emb_dim = self.word_emb_dim
		
		# output layer
		# x dot W dot tranpose(context)
		output_w = tf.get_variable("out_w",[word_emb_dim,hidden_dim])
		output_b = tf.get_variable("out_b",[1,])
		logit = tf.add(tf.matmul(tf.matmul(decoder_input,output_w),final_state,transpose_b=True),output_b)
		# shape (1,1)

		return logit

class GEDModel(object):
	def __init__(self, is_training, config):

		self.max_seq_len = max_seq_len = config.max_seq_len
		self.hidden_dim = hidden_dim = config.hidden_dim
		self.word_emb_dim = word_emb_dim = config.word_emb_dim
		self.vocab_size = vocab_size = config.vocab_size

		self.input_data = tf.placeholder(tf.int32, shape=[None,], name="input_data")
		# shape [unknown(sentence length),]
		self.targets = tf.placeholder(tf.int32, shape=[None,], name="targets")
		# shape [unknown(sentence length),]

		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
		# word-embedding layer that convert word index to word vectors
		with tf.variable_scope("lookup"):
			embedding = tf.get_variable("embedding", [vocab_size, word_emb_dim], initializer=initializer)

		inputs = tf.nn.embedding_lookup(embedding, self.input_data)
		# shape (sent_len,word_emb_dim)

		# left-to-right-directional LSTM Encoder
		self.fw_lstm = Encoder(hidden_dim,scope="fw_lstm")
		init_state = tf.zeros([1,2*hidden_dim])

		with tf.variable_scope("fw_lstm") as scope:

			def fw_step(fw_mix_state,x):
				x = tf.reshape(x,[1,word_emb_dim])
				fw_mem_state, fw_hid_state = tf.split(1, 2, fw_mix_state)
				fw_mem_state, fw_hid_state = self.fw_lstm(x, fw_mem_state, fw_hid_state)
				fw_mix_state = tf.concat(1,[fw_mem_state,fw_hid_state])
				return fw_mix_state

			fw_mix_states = tf.scan(fw_step,inputs,initializer=init_state)
			fw_mix_states_mat = tf.reshape(fw_mix_states,[-1,2*hidden_dim])
			fw_mem_tape, fw_hid_tape = tf.split(1, 2, fw_mix_states_mat)

			reversed_hid_tape = tf.reverse(fw_hid_tape,[True,False])
			final_state = tf.slice(reversed_hid_tape,[0,0],[1,-1])
			# shape (1, hidden_dim)

		decoder_input = inputs

		# classifier without attention
		self.classifier = Classifier(word_emb_dim,hidden_dim,max_seq_len,scope="classifier")
		with tf.variable_scope("classifier") as scope:
			init_logit = tf.zeros([1,1])
			# only to get tensor of shape (1, 1)

			def decode_step(logit,x):
				x = tf.reshape(x,[1,word_emb_dim])
				logit = self.classifier(x,final_state)
				return logit

			logit_tape = tf.scan(decode_step,decoder_input,initializer=init_logit)

		logits = tf.reshape(logit_tape,[-1,])
		# shape (sent_len,)

		# predict
		self.prediction = tf.sigmoid(logits)
		# loss
		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,tf.to_float(self.targets))
		self.cost = cost = tf.reduce_mean(loss,[0])

		# back-prop
		if is_training:
			# setup learning rate variable to decay
			self.lr = tf.Variable(1.0, trainable=False)
			# define training operation and clip the gradients
			tvars = tf.trainable_variables()
			trainables = []
			for item in tvars:
				try:
					tf.gradients(cost,item)
					trainables.append(item)
				except:
					pass
			grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainables), config.max_grad_norm)
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
			self.train_op = optimizer.apply_gradients(zip(grads, trainables), name="train")
		else:
			# if this model isn't for training (i.e. testing/validation) then we don't do anything here
			self.train_op = tf.no_op()

class Config(object):
	hidden_dim = 300 # number of blocks in an LSTM cell
	word_emb_dim = 300
	vocab_size = 30000
	max_grad_norm = 5 # maximum gradient for clipping
	init_scale = 0.05 # scale between -0.1 and 0.1 for all random initialization
	learning_rate = 1.0
	lr_decay = 0.8
	lr_decay_epoch_offset = 6 # don't decay until after the Nth epoch