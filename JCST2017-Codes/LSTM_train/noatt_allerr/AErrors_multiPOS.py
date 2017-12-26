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

	confus = { 'CC':set(), 'DT':set(), 'PR':set(), 'IN':set(), 'WS':set() }
	cc = ("CC")
	dt = ("DT","PDT")
	pr = ("PRP", "PRP$")
	_in = ("IN", "TO", "RP")
	ws = ("WDT", "WP", "WP$", "WRB")
	nn = ("NN", "NNP", "NNPS", "NNS")
	vb = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
	jr = ("JJ", "JJR", "JJS", "RB", "RBR", "RBS")
	
	lemmatizer = WordNetLemmatizer()
	stemmer = LancasterStemmer()

	for line in txt:

		sentence = [tok for tok in line.lower().split()]
		word_pos_map = dict(nltk.pos_tag(sentence))

		for word, pos in word_pos_map.iteritems():
			# build confusion set according to PoS tag
			if pos in cc:
				confus['CC'].add(word)
			elif pos in dt:
				confus['DT'].add(word)
			elif pos in pr:
				confus['PR'].add(word)
			elif pos in _in:
				confus['IN'].add(word)
			elif pos in ws:
				confus['WS'].add(word)
			# build confusion set according to lemma (nouns and verbs)
			elif pos in nn or pos in vb:
				if pos in nn:
					lemma = lemmatizer.lemmatize(word,'n').encode('utf-8')
				elif pos in vb:
					lemma = lemmatizer.lemmatize(word,'v').encode('utf-8')
				if lemma in confus:
					confus[lemma].add(word)
				else:
					confus[lemma] = set()
					confus[lemma].add(word)
			# build confusion set according to stem (adjectives and adverbs)
			elif pos in jr:
				word_stem = stemmer.stem(word)
				if word_stem in confus:
					confus[word_stem].add(word)
				else:
					confus[word_stem] = set()
					confus[word_stem].add(word)
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
	cc = ("CC")
	dt = ("DT","PDT")
	pr = ("PRP", "PRP$")
	_in = ("IN", "TO", "RP")
	ws = ("WDT", "WP", "WP$", "WRB")
	nn = ("NN", "NNP", "NNPS", "NNS")
	vb = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
	jr = ("JJ", "JJR", "JJS", "RB", "RBR", "RBS")

	lemmatizer = WordNetLemmatizer()
	stemmer = LancasterStemmer()

	uniform_flag = False
	new_word = word

	pos = nltk.pos_tag([word])[0][1]
	# pos
	if pos in cc:
		if len(confus['CC']) > 1:
			while new_word == word:
				new_word = random.sample(confus['CC'],1)[0]
		else:
			uniform_flag = True
	elif pos in dt:
		if len(confus['DT']) > 1:
			while new_word == word:
				new_word = random.sample(confus['DT'],1)[0]
		else:
			uniform_flag = True
	elif pos in pr:
		if len(confus['PR']) > 1:
			while new_word == word:
				new_word = random.sample(confus['PR'],1)[0]
		else:
			uniform_flag = True
	elif pos in _in:
		if len(confus['IN']) > 1:
			while new_word == word:
				new_word = random.sample(confus['IN'],1)[0]
		else:
			uniform_flag = True
	elif pos in ws:
		if len(confus['WS']) > 1:
			while new_word == word:
				new_word = random.sample(confus['WS'],1)[0]
		else:
			uniform_flag = True
	# lemma
	elif pos in nn or pos in vb:
		if pos in nn:
			lemma = lemmatizer.lemmatize(word,'n').encode('utf-8')
		elif pos in vb:
			lemma = lemmatizer.lemmatize(word,'v').encode('utf-8')
		if lemma in confus:			
			if len(confus[lemma]) > 1:
				while new_word == word:
					new_word = random.sample(confus[lemma],1)[0]
			else:
				uniform_flag = True
		else:
			uniform_flag = True
	# stem
	elif pos in jr:
		word_stem = stemmer.stem(word)
		if word_stem in confus:
			if len(confus[word_stem]) > 1:
				while new_word == word:
					new_word = random.sample(confus[word_stem],1)[0]
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
