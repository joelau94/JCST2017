import os
import sys

train_size = str(sys.argv[1])
test_size = str(sys.argv[2])

svm_home = "/global-mt/liuzhuoran/liblinear-2.1/"

train_input = "train.in." + train_size + ".txt"
test_input = "test.in." + train_size + "." + test_size + ".txt"
test_ouput = "test.out." + train_size + "." + test_size + ".txt"
eval_file = "eval." + train_size + "." + test_size + ".txt"
model_name = train_size + ".model"

os.system("python data_gen.py acl.klm acl_min_2.src.txt acl_min_2.lbl.txt " + train_input + " " + test_input + " " + train_size + " " + test_size)

os.system(svm_home + "train -s 2 " + train_input + " " + model_name)
os.system(svm_home + "predict " + test_input + " " + model_name + " " + test_ouput)
os.system("python score.py " + test_input + " " + test_ouput + " 0.5 > " + eval_file)
