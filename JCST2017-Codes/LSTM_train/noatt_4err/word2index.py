import os
import sys
import random
import string
import math
import glob
from collections import defaultdict

corpus_path = str(sys.argv[1])
dict_file = str(sys.argv[2])
maxLine = int(float(sys.argv[3]))

file_names = glob.glob(corpus_path + "*.txt")

dictionary = defaultdict(int)
word_idx_map = open(dict_file,"r")
for line in word_idx_map:
	dictionary[line.split()[0]] = int(float(line.split()[1]))
word_idx_map.close()
dictionary.default_factory = lambda: dictionary['<unk>']

for file_name in file_names:
	txt = open(file_name, "r")
	idx = open(corpus_path + file_name.split('/')[-1].split('_')[0] + "_index.csv", "w+")
	for line in txt:
		toks = line.split()
		for i in range(len(toks)):
			idx.write(str(dictionary[toks[i]])+" ")
		idx.write("\n")
	txt.close()
	idx.close()