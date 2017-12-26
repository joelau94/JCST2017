import string
import sys
import os

vocab_size = str(sys.argv[1])
word_emb_dim = str(sys.argv[2])
hidden_dim = str(sys.argv[3])
save_path = str(sys.argv[4])
nEpochs = int(float(sys.argv[5]))
max_seq_len = int(float(sys.argv[6]))

log_file = "{}_{}_{}.log".format(vocab_size,word_emb_dim,hidden_dim)

log = open(log_file,"a+")

model_dir = "./model/{}_{}_{}_model/".format(vocab_size, word_emb_dim, hidden_dim)
os.system("mkdir "+model_dir)

train_cmd = "python train.py {} {} {} {} {} {} {} {}".format(save_path,model_dir,vocab_size,word_emb_dim,hidden_dim,nEpochs,max_seq_len,log_file)
log.write("Training: " + train_cmd + "\n")
log.flush()
# os.fsync()
os.system(train_cmd)