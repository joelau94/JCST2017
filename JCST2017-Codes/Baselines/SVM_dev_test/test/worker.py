import os
import sys

src_file = str(sys.argv[1])
lbl_file = str(sys.argv[2])
model_name = str(sys.argv[3])

test_input = src_file.split('/')[-1].split('.')[0] + ".in.txt"
test_output = test_input.split('.')[0] + model_name.split('/')[-1].split('.')[0] + ".out.txt"
eval_file = "eval." + test_output.split('.')[0] + ".log"
svm_home = "/global-mt/liuzhuoran/liblinear-2.1/"

# generate data
os.system("python data_gen.py acl.klm {} {} {} ".format(src_file,lbl_file,test_input))

# classify(test)
os.system(svm_home + "predict " + test_input + " " + model_name + " " + test_output)

# calculate F_0.5
os.system("python score.py " + test_input + " " + test_output + " 0.5 > " + eval_file)
