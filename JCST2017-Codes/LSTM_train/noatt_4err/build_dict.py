import os
import sys
import random
import math
import string
import gensim
import gensim.models as gm
from gensim import corpora
from collections import defaultdict

text_file = str(sys.argv[1])
dict_file_path = str(sys.argv[2])
text_oov_file = str(sys.argv[3])
min_word_freq = int(float(sys.argv[4]))

# calculating frequency of words
frequency = defaultdict(int)
frequency['<unk>'] += 1
with open(text_file, "r") as txt:
	for line in txt:
		tok = line.lower().split()
		for word in tok:
			frequency[word] += 1
	txt.close()

# replacing non-frequent words with <unk>
tok = []
with open(text_file, "r") as txt:
	with open(text_oov_file, "w+") as oov:
		for line in txt:
			tok = line.split()
			if len(tok):
				for word in tok:
					if frequency[word.lower()] < min_word_freq:
						oov.write("<unk> ")
					else:
						oov.write(word+" ")
				oov.write("\n")
		oov.write("<unk>\n")
		oov.close()
	txt.close()

# build dictionary
sentences = []
with open(text_oov_file, "r") as oov:
	sentences = [[tok for tok in line.lower().split()] for line in oov]
	dictionary = corpora.Dictionary(sentences)
	dictionary.save_as_text(dict_file_path + str(min_word_freq) + ".i2w.dict")
	print(str(len(dictionary.values())))
	oov.close()

# reverse dictionary and get word -> index mapping
dic = open(dict_file_path + str(min_word_freq) + ".i2w.dict", "r")
rev_dict = open(dict_file_path + str(min_word_freq) + ".w2i.dict", "w+")
for line in dic:
	rev_dict.write(line.split()[1] + "\t" + line.split()[0] + "\n")
dic.close()
rev_dict.close()