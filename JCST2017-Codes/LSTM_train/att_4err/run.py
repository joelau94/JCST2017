import sys
import os
import string
import subprocess

min_word_freqs = [2]

for min_word_freq in min_word_freqs:
	subprocess.Popen(["python","worker.py",str(min_word_freq)])
