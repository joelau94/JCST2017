import sys
import os
import string
import subprocess

min_word_freq = int(float(sys.argv[1]))
maxLine = 10000
max_seq_len = 50

word_emb_sizes = [100, 300]
rnn_sizes = [100, 300]

data_dir = "./data/"
dict_dir = "./dict/"
text_file = data_dir + "acl_5_50.txt"

# main

text_oov_file = data_dir + str(min_word_freq) + "_acl.oov.txt"
save_path = data_dir + str(maxLine) + "_" + str(min_word_freq) + "_data/"

os.system("mkdir " + save_path)

build_dict_cmd = "python build_dict.py {} {} {} {}".format(text_file, dict_dir, text_oov_file, str(min_word_freq))
os.system(build_dict_cmd)

dict_file = dict_dir + str(min_word_freq) + ".i2w.dict"
reverse_dict_file = dict_dir + str(min_word_freq) + ".w2i.dict"

error_gen_cmd = "python error_gen.py {} {} {} {}".format(text_oov_file, dict_file, str(maxLine), save_path)
os.system(error_gen_cmd)

word2index_cmd = "python word2index.py {} {} {}".format(save_path, reverse_dict_file, str(maxLine))
os.system(word2index_cmd)

dic = open(dict_file,"r")
vocab_size = 0
for line in dic:
	vocab_size += 1
dic.close()

shared = open(save_path + "shared","r")
nEpochs = shared.readline().strip()
shared.close()

# for word_emb_size in word_emb_sizes:
# 	for rnn_size in rnn_sizes:
# 		subprocess.Popen(["python","sub_worker.py",str(vocab_size),str(word_emb_size),str(rnn_size),save_path,nEpochs])
#subprocess.Popen(["python","sub_worker.py",str(vocab_size),"100","100",save_path,nEpochs,str(max_seq_len)])
subprocess.Popen(["python","sub_worker.py",str(vocab_size),"150","150",save_path,nEpochs,str(max_seq_len)])
#subprocess.Popen(["python","sub_worker.py",str(vocab_size),"500","500",save_path,nEpochs,str(max_seq_len)])
