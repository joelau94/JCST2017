import string
import nltk
import cPickle
import math
import random
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

def build_confus(text_file,save_file):
	"""
	text_file: text file name with all tokens after eliminating oov
	save_file: pickle file name
	"""
	txt = open(text_file,"r")
	pkl = open(save_file,"w+")

	confus = { 'DT':set(['a','an','the']),
	           'IN':set(['for', 'to', 'of', 'by', 'as', 'in', 'at', 'on', 'in', 'from', 'off', 'at', 'toward', 'over', 'into', 'upon', 'with']) }
	
	nn = ("NN", "NNP", "NNPS", "NNS")
	vb = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
	
	lemmatizer = WordNetLemmatizer()

	for line in txt:

		sentence = [tok for tok in line.lower().split()]
		word_pos_map = dict(nltk.pos_tag(sentence))

		for word, pos in word_pos_map.iteritems():
			# build confusion set according to PoS tag
			if pos in nn or pos in vb:
				if pos in nn:
					lemma = lemmatizer.lemmatize(word,'n').encode('utf-8')
					lemma += "_NN"
				elif pos in vb:
					lemma = lemmatizer.lemmatize(word,'v').encode('utf-8')
					lemma += "_VB"
				if lemma in confus:
					confus[lemma].add(word)
				else:
					confus[lemma] = set()
					confus[lemma].add(word)
	# end for

	cPickle.dump(confus,pkl)
	pkl.close()

	return confus

def make_error(word, dictionary, confus):
	"""
	word: original word
	dictionary: build by genism.copora.Dictionary
	confusion_set: build by build_confus()
	"""
	
	nn = ("NN", "NNP", "NNPS", "NNS")
	vb = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")

	lemmatizer = WordNetLemmatizer()

	uniform_flag = False
	new_word = word

	pos = nltk.pos_tag([word])[0][1]
	# pos
	if word in confus['DT']:
		while new_word == word:
			new_word = random.sample(confus['DT'],1)[0]
	elif word in confus['IN']:
		while new_word == word:
			new_word = random.sample(confus['IN'],1)[0]
	# lemma
	elif pos in nn or pos in vb:
		if pos in nn:
			lemma = lemmatizer.lemmatize(word,'n').encode('utf-8')
			lemma += "_NN"
		elif pos in vb:
			lemma = lemmatizer.lemmatize(word,'v').encode('utf-8')
			lemma += "_VB"
		if lemma in confus:
			if len(confus[lemma]) > 1:
				while new_word == word:
					new_word = random.sample(confus[lemma],1)[0]
			else:
				uniform_flag = True
		else:
			uniform_flag = True
	# other words or tokens
	else:
		uniform_flag = True
	# if unable to find a substitution in confusion sets
	if uniform_flag == True:
		while new_word == word:
			sub = random.randrange(0,max(dictionary.keys()))
			new_word = dictionary.get(sub)

	return new_word
