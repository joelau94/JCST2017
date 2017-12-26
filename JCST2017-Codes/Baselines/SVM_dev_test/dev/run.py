import os
import sys
import string
import subprocess
import glob

test_set = str(sys.argv[1])

models = glob.glob("models/*.model")

if test_set == "aesw":
	text_file = "./data/aesw/aesw_test_err.dev.src.txt"
	label_file = "./data/aesw/aesw_test_err.dev.lbl.txt"
elif test_set == "ccl":
	text_file = "./data/ccl/ccl_src_50.dev.src.txt"
	label_file = "./data/ccl/ccl_label_50.dev.lbl.txt"
elif test_set == "conll":
	text_file = "./data/conll/conll.dev.src.txt"
	label_file = "./data/conll/conll.dev.lbl.txt"

# main
for model in models:
	subprocess.Popen(["python","worker.py",text_file,label_file,model])
