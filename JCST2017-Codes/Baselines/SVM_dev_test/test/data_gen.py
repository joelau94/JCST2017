import string
import sys
import kenlm

ngram_model_file = str(sys.argv[1])
src_file = str(sys.argv[2])
lbl_file = str(sys.argv[3])
test_inp_file = str(sys.argv[4])

ngram_model = kenlm.LanguageModel(ngram_model_file)
src = open(src_file,"r")
lbl = open(lbl_file,"r")

inp = open(test_inp_file,"w+")

for line in lbl:
	labels = line.split()
	toks = src.readline().lower().split()
	for i in range(0,len(toks)):
		if labels[i] == "1":
			inp.write("+1 ")
		else:
			inp.write("-1 ")
		inp.write("1:" + str(ngram_model.score( toks[i] )) + " ")
		if i - 2 >= 0:
			inp.write("2:" + str(ngram_model.score( toks[i-2] )) + " ")
		if i - 1 >= 0:
			inp.write("3:" + str(ngram_model.score( toks[i-1] )) + " ")
		if i + 1 < len(toks):
			inp.write("4:" + str(ngram_model.score( toks[i+1] )) + " ")
		if i + 2 < len(toks):
			inp.write("5:" + str(ngram_model.score( toks[i+2] )) + " ")
		if i - 1 >= 0:
			inp.write("6:" + str(ngram_model.score( toks[i-1] + " " + toks[i] )) + " ")
		if i + 1 < len(toks):
			inp.write("7:" + str(ngram_model.score( toks[i] + " " + toks[i+1] )) + " ")
		if i - 2 >= 0:
			inp.write("8:" + str(ngram_model.score( toks[i-2] + " " + toks[i-1] + " " + toks[i] )) + " ")
		if i + 2 < len(toks):
			inp.write("9:" + str(ngram_model.score( toks[i] + " " + toks[i+1] + " " + toks[i+2] )) + " ")
		if i - 1 >= 0 and i + 1 < len(toks):
			inp.write("10:" + str(ngram_model.score( toks[i-1] + " " + toks[i] + " " + toks[i+1] )) + " ")
		inp.write("\n")
