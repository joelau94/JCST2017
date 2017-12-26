import sys
import os
import string

idx_file = str(sys.argv[1])
lbl_file = str(sys.argv[2])
start_id = int(float(sys.argv[3])) - 1
maxline = 10000

ids = open(idx_file, 'r').readlines()
lbls = open(lbl_file, 'r').readlines()

idx_spl = None
lbl_spl = None

for i in xrange(len(ids)):
	if i % maxline == 0:
		start_id += 1
		idx_spl = open(str(start_id)+'_index.csv','w+')
		lbl_spl = open(str(start_id)+'_label.csv','w+')
	idx_spl.write(ids[i])
	lbl_spl.write(lbls[i])
