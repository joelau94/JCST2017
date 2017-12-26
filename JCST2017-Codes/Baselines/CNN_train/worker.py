import os
import sys
import random
import math
import string

def build_model(window_size, vector_size, min_word_freq, save_path, src_file, log_file):
	cmd = "python build_model.py"
	cmd += " " + str(window_size)
	cmd += " " + str(vector_size)
	cmd += " " + str(min_word_freq)
	cmd += " " + save_path
	cmd += " " + src_file
	cmd += " " + log_file
	os.system(cmd)

def src_lbl(maxLine, err_file, lbl_file, save_path, sharedFile, logFile):
	cmd = "python txt2src+lbl.py"
	cmd += " " + str(maxLine)
	cmd += " " + err_file
	cmd += " " + lbl_file
	cmd += " " + save_path
	cmd += " " + sharedFile
	cmd += " " + logFile
	os.system(cmd)

def train(modelPath, dataPath, dictionary, langMdl, logFile, vectorSize, windowSize, batchSize, nEpochs, LR=1e-3, WD=0, momentum=0, LRD=1e-7):	
	cmd = "~/torch/install/bin/th train.lua"
	#cmd = "th train.lua"
	cmd += " -modelPath " + modelPath
	cmd += " -dataPath " + dataPath
	cmd += " -dict " + dictionary
	cmd += " -langMdl " + langMdl
	cmd += " -logFile " + logFile
	cmd += " -vectorSize " + str(vectorSize)
	cmd += " -windowSize " + str(windowSize)
	cmd += " -batchSize " + str(batchSize)
	cmd += " -nEpochs " + str(nEpochs)
	cmd += " -learningRate " + str(LR)
	cmd += " -weightDecay " + str(WD)
	cmd += " -momentum " + str(momentum)
	cmd += " -learningRateDecay " + str(LRD)
	os.system(cmd)

def test(modelPath, dataPath, dictionary, langMdl, logFile, dataNo, modelNo, vectorSize, windowSize):
	cmd = "~/torch/install/bin/th test.lua"
	#cmd = "th test.lua"
	cmd += " -modelPath " + modelPath
	cmd += " -dataPath " + dataPath
	cmd += " -dict " + dictionary
	cmd += " -langMdl " + langMdl
	cmd += " -logFile " + logFile
	cmd += " -dataNo " + str(dataNo)
	cmd += " -modelNo " + str(modelNo)
	cmd += " -vectorSize " + str(vectorSize)
	cmd += " -windowSize " + str(windowSize)
	os.system(cmd)


# Hyperparameters
vectorSize = int(float(sys.argv[1]))
windowSize = int(float(sys.argv[2]))
min_word_freq = int(float(sys.argv[3]))
batchSizes = [5]
maxLine = 1000
epochStep = 20 # 126k lines == 100 MiB of training text

# Global Configurations
corpus_path = "./corpus/"
src_file = corpus_path + str(min_word_freq) + "_acl.oov.txt"
err_file = corpus_path + "acl_min_" + str(min_word_freq) + ".src.txt"
lbl_file = corpus_path + "acl_min_" + str(min_word_freq) + ".lbl.txt"
#corpus stores txt with <unk> (without oov), without error
#file_name_match = "acl"
#lm_train_text = "./nyt1.txt"
LM_path = "LM/"
dataDir = "data/"
# tmp_data_path = dataDir + specific
logFile = str(vectorSize) + "_" + str(windowSize) + "_" + str(min_word_freq) + "_" + "log.txt"

# main
dataPath = "data/data" + "_" + str(vectorSize) + "_" + str(windowSize) + "_" + str(min_word_freq)+"/"
os.system("mkdir "+dataPath)

build_model(windowSize, vectorSize, min_word_freq, LM_path, src_file, logFile)
dict_file = LM_path + str(vectorSize)+"_"+str(windowSize)+"_"+str(min_word_freq)+"_dict"
model_file = LM_path + str(vectorSize)+"_"+str(windowSize)+"_"+str(min_word_freq)+"_mdl"
sharedFile = dataPath + "shared"

# partition source and label file by max_line
src_lbl(maxLine, err_file, lbl_file, dataPath, sharedFile, logFile)

with open(sharedFile, "r") as shared:
	nEpochs = int(float(shared.read()))
	shared.close()
for batchSize in batchSizes:
	nnModel_path = "nnModel/" + str(vectorSize) + "_" + str(windowSize) + "_" + str(min_word_freq) + "_" + str(batchSize) + "_mdl/"
	train(nnModel_path, dataPath, dict_file, model_file, logFile, vectorSize, windowSize, batchSize, nEpochs-1)
	for modelNo in range(epochStep, nEpochs-2, epochStep):
		test(nnModel_path, dataPath, dict_file, model_file, logFile, nEpochs-1, modelNo, vectorSize, windowSize)
	test(nnModel_path, dataPath, dict_file, model_file, logFile, nEpochs-1, nEpochs-2, vectorSize, windowSize)
#os.system("rm -r " + "data/data" + "_" + str(vectorSize) + "_" + str(windowSize) + "_" + str(min_word_freq))
