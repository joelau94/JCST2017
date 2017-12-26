import os
import sys
import random
import string
import math
import glob
from collections import defaultdict

txt_file = str(sys.argv[1])
idx_file = str(sys.argv[2])
dict_file = str(sys.argv[3])

dictionary = defaultdict(int)
word_idx_map = open(dict_file,"r")
for line in word_idx_map:
	dictionary[line.split()[0]] = int(float(line.split()[1]))
word_idx_map.close()
dictionary.default_factory = lambda: dictionary['<unk>']

txt = open(txt_file, "r")
idx = open(idx_file, "w+")
for line in txt:
	toks = line.lower().split()
	for i in range(len(toks)):
		idx.write(str(dictionary[toks[i]])+" ")
	idx.write("\n")
txt.close()
idx.close()