import os
import sys
import string
import glob

test_set = str(sys.argv[1])
log_file = str(sys.argv[2])

# min_word_freq, vocab_size, dict_file
# vocab_config = (10,52276,"./dict/10.w2i.dict")
# vocab_config = (25,30182,"./dict/25.w2i.dict")
# vocab_config = (3,38561,"./dict/3.w2i.dict")
vocab_config = (2,166578,"./dict/2.w2i.dict")
word_emb_dim = hidden_dim = 150 # 100, 300, 500

max_seq_len = 50

log = open(log_file,"a+")

if test_set == "aesw":
	text_file = "./data/aesw/aesw_test_err.test.src.txt"
	label_file = "./data/aesw/aesw_test_err.test.lbl.txt"
elif test_set == "ccl":
	text_file = "./data/ccl/ccl_src_50.test.src.txt"
	label_file = "./data/ccl/ccl_label_50.test.lbl.txt"
elif test_set == "conll":
	text_file = "./data/conll/conll.test.src.txt"
	label_file = "./data/conll/conll.test.lbl.txt"

index_file = text_file.split('/')[-1].split('.')[0] + "_index.csv"

vocab_size = vocab_config[1]
dict_file = vocab_config[2]

# os.system("python word2index.py {} {}".format(text_file,dict_file))

for i in range(1,37):
	model_file = "./model/{}.mdl".format(i)
	test_cmd = "python test.py {} {} {} {} {} {} {} {}".format(index_file,label_file,vocab_size,word_emb_dim,hidden_dim,max_seq_len,model_file,log_file)
	log.write(test_cmd)
	log.flush()
	os.fsync(log)
	os.system(test_cmd)
