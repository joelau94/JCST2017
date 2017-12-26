import os
import sys
import random
import math
import string
import subprocess

# Hyperparameters
vectorSizes = [50]
windowSizes = [3]
minWordFreqs = [2]

"""
To run this script, there should be
'<min_word_freq>_acl.oov.txt', 'acl_min_<min_word_freq>.src.txt', 'acl_min_<min_word_freq>.lbl.txt'
in corpus/ directory
"""

# main
for vectorSize in vectorSizes:
	for windowSize in windowSizes:
		for min_word_freq in minWordFreqs:
			subprocess.Popen(["python","worker.py",str(vectorSize),str(windowSize),str(min_word_freq)])
