import sys
import string

test_input_file = str(sys.argv[1])
predict_file = str(sys.argv[2])
f_measure = float(sys.argv[3])

test_in = open(test_input_file,"r")
pred = open(predict_file,"r")

gold_std = 0
recall_count = 0
err_report = 0
precision_count = 0

for line in test_in:
	label = line.split()[0]
	predict = pred.readline().strip()
	if label == "-1":
		gold_std += 1
		if predict[0] == "-":
			recall_count += 1
	if predict[0] == "-":
		err_report += 1
		if label == "-1":
			precision_count += 1

recall = 0
precision = 0
f_score = 0

try:
	recall = float(recall_count) / float(gold_std)
	precision = float(precision_count) / float(err_report)
	f_score = float( ( 1 + f_measure * f_measure ) * recall * precision ) / float( recall + f_measure * f_measure * precision )
except ZeroDivisionError:
	"Zero Division Error!"

display = "recall=" + str(recall) + "\nprecision=" + str(precision) + "\nf_" + str(f_measure) + "=" + str(f_score) + "\n"
print( display )
