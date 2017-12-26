import os
import sys
import random
import string
import math
from collections import defaultdict

text_file = str(sys.argv[1])
dict_file = str(sys.argv[2]) # w2i

dictionary = defaultdict(int)
word_idx_map = open(dict_file,"r")
for line in word_idx_map:
	dictionary[line.split()[0]] = int(float(line.split()[1]))
word_idx_map.close()
dictionary.default_factory = lambda: dictionary['<unk>']

txt = open(text_file, "r")
idx = open(text_file.split('/')[-1].split('.')[0] + "_index.csv", "w+")
for line in txt:
	toks = line.split()
	for i in range(len(toks)):
		idx.write(str(dictionary[toks[i]])+" ")
	idx.write("\n")
txt.close()
idx.close()
