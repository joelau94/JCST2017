import os
import sys
import string
import subprocess
import glob

test_set = str(sys.argv[1])

models = glob.glob("models/*.model")

if test_set == "aesw":
	text_file = "./data/aesw/aesw_test_err.test.src.txt"
	label_file = "./data/aesw/aesw_test_err.test.lbl.txt"
elif test_set == "ccl":
	text_file = "./data/ccl/ccl_src_50.test.src.txt"
	label_file = "./data/ccl/ccl_label_50.test.lbl.txt"
elif test_set == "conll":
	text_file = "./data/conll/conll.test.src.txt"
	label_file = "./data/conll/conll.test.lbl.txt"

# main
for model in models:
	subprocess.Popen(["python","worker.py",text_file,label_file,model])
