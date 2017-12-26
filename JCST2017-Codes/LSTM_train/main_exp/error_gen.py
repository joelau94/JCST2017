import os
import sys
import random
import string
import math
from gensim import corpora
import AErrors_multiPOS

text_file = str(sys.argv[1])
dict_file = str(sys.argv[2])
maxLine = int(float(sys.argv[3]))
save_path = str(sys.argv[4])

tok = []
dictionary = corpora.Dictionary.load_from_text(dict_file)

save_file = "{}{}.pkl".format(save_path,text_file.split('/')[-1].split('.')[0])
confus = AErrors_multiPOS.build_confus(text_file, save_file)
print("Confusion Set built.")

lineCount = 0
fileCount = 0
with open(text_file, "r") as corr:
	# iterating over lines
	for line in corr:
		lineCount += 1
		# lineCount % maxLine == 1 indicates the start of a new file
		if lineCount%maxLine == 1:
			fileCount += 1
			print("  Train text " + str(fileCount) + " ... ")
			err = open(save_path + str(fileCount) + "_source.txt", "w+")
			lbl = open(save_path + str(fileCount) + "_label.csv", "w+")
		tok = line.split()
		if len(tok):
			# replace a randomly chosen word in a sentence
			error_pos = random.randrange(0,len(tok),1)
			# with a generated erroneous word
			tok[error_pos] = AErrors_multiPOS.make_error(tok[error_pos],dictionary,confus)
			for word in tok:
				err.write(word+" ")
			err.write("\n")
			# then record the error position with labels
			for i in range(len(tok)):
				if i == error_pos :
					lbl.write("0 ")
				else:
					lbl.write("1 ")
			lbl.write("\n")
		# lineCount % maxLine == 0 indicates reaching the maximum line count of a single file
		if lineCount%maxLine == 0:
			err.close()
			lbl.close()
	# for the last file, if it does not reach the maximum line count, close it
	if lineCount%maxLine != 0:
		err.close()
		lbl.close()

shared = open(save_path + "shared", "w+")
shared.write(str(fileCount))
shared.close()