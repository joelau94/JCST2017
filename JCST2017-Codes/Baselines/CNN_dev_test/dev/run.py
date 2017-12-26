import sys
import os

def test(modelPath, dataFile, labelFile, dictionary, langMdl, logFile, modelNo, vectorSize, windowSize):
	cmd = "~/torch/install/bin/th test_general_dev.lua"
	#cmd = "th test.lua"
	cmd += " -modelPath " + modelPath
	cmd += " -dataFile " + dataFile
	cmd += " -labelFile " + labelFile
	cmd += " -dict " + dictionary
	cmd += " -langMdl " + langMdl
	cmd += " -logFile " + logFile
	cmd += " -modelNo " + str(modelNo)
	cmd += " -vectorSize " + str(vectorSize)
	cmd += " -windowSize " + str(windowSize)
	os.system(cmd)

test_set = str(sys.argv[1])
logFile = str(sys.argv[2])

dataFile = test_set + ".csv"

if test_set == "aesw":
	src_file = "./data/aesw/aesw_test_err.dev.src.txt"
	label_file = "./data/aesw/aesw_test_err_cnn.dev.lbl.txt"
elif test_set == "ccl":
	src_file = "./data/ccl/ccl_src_50.dev.src.txt"
	label_file = "./data/ccl/ccl_label_50_cnn.dev.lbl.txt"
elif test_set == "conll":
	src_file = "./data/conll/conll.dev.src.txt"
	label_file = "./data/conll/conll_cnn.dev.lbl.txt"

size_config = (50,3,2)
#size_config = (75,5,25)

vectorSize = size_config[0]
windowSize = size_config[1]
min_word_freq = size_config[2]

modelNos = range(20,2965,20)

dict_file = "LM/" + str(vectorSize)+"_"+str(windowSize)+"_"+str(min_word_freq)+"_dict"
model_file = "LM/" + str(vectorSize)+"_"+str(windowSize)+"_"+str(min_word_freq)+"_mdl"
batchSize = 5
nnModel_path = "nnModel/" + str(vectorSize) + "_" + str(windowSize) + "_" + str(min_word_freq) + "_" + str(batchSize) + "_mdl/"

data_gen_cmd = "python src2vec_general_dev.py {} {} {} {} {} {} {}".format(vectorSize,windowSize,dict_file,model_file,src_file,dataFile,logFile)
os.system(data_gen_cmd)

for modelNo in modelNos:
	test(nnModel_path, dataFile, label_file, dict_file, model_file, logFile, modelNo, vectorSize, windowSize)
