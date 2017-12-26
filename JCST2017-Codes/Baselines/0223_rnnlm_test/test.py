import string
import sys
import os
import cPickle as pkl

testset = str(sys.argv[1])

def make_partial_sentences(input_file, output_file, unk_cmp_file, pointer_file):
	sentences = [ [ word for word in sent.strip().lower().split() ] for sent in open(input_file,'r') ]
	parts = [ [ ' '.join(sent[0:i+1]) for i in xrange(len(sent)) ] for sent in sentences ]
	unk_parts = [ [ ' '.join(sent[0:i]+['<rand>']) for i in xrange(len(sent)) ] for sent in sentences ]
	out = []
	unk = []
	pointers = []
	cnt = 0
	for i in xrange(len(parts)):
		pointers.append(cnt)
		for j in xrange(len(parts[i])):
			out.append( '{} {}'.format(cnt, parts[i][j]) )
			unk.append( '{} {}'.format(cnt, unk_parts[i][j]) )
			cnt += 1
	pointers.append(cnt)
	open(output_file,'w+').write( '\n'.join(out) + '\n' )
	open(unk_cmp_file,'w+').write( '\n'.join(unk) + '\n' )
	pkl.dump(pointers, open(pointer_file,'wb'))

def score_sentence(input_file, output_file):
	cmd = './rnnlm -rnnlm model -test {} -nbest -debug 0 > {}'.format(input_file, output_file)
	os.system(cmd)

def predict(src_score_file, unk_score_file, pointer_file, output_file):
	src_scores = map(float, open(src_score_file, 'r').readlines())
	unk_scores = map(float, open(unk_score_file, 'r').readlines())
	pointers = pkl.load(open(pointer_file, 'rb'))
	prediction = [ '1' if src_scores[i]>unk_scores[i] else '0' for i in xrange(len(src_scores)) ]
	out = '\n'.join([ ' '.join(prediction[pointers[i]:pointers[i+1]]) for i in xrange(len(pointers)-1) ])
	open(output_file,'w+').write(out)

def evaluate(res_file, truth_file, log_file):
	gold_std = 0	# number of errors in golden standard
	recall_count = 0	# number of errors in golden standard hit by model
	precision_count = 0	# number of errors reported by model correctly
	err_report = 0	# number of errors reported by model

	res = [ map(int, line.strip().split()) for line in open(res_file,'r').readlines() ]
	truth = [ map(int, line.strip().split()) for line in open(truth_file,'r').readlines() ]

	for i in xrange(len(truth)):
		for j in range(len(truth[i])):
			if truth[i][j] == 0:
				gold_std += 1
				if res[i][j] == 0:
					recall_count += 1
			if res[i][j] == 0:
				err_report += 1
				if truth[i][j] == 0:
					precision_count += 1

	recall = float(recall_count) / float(gold_std)
	precision = float(precision_count) / float(err_report)
	f_0_5 = float( ( 1 + 0.25 ) * recall * precision ) / float( recall + 0.25 * precision )
	open(log_file,'a+').write("recall = {}\nprecision = {}\nf_0_5 = {}\n".format(recall, precision, f_0_5))


# main
make_partial_sentences(testset+'.in',testset+'.out',testset+'.unk',testset+'.pkl')
score_sentence(testset+'.out',testset+'.src.score')
score_sentence(testset+'.unk',testset+'.unk.score')
predict(testset+'.src.score',testset+'.unk.score',testset+'.pkl',testset+'.res')
evaluate(testset+'.res',testset+'.truth',testset+'.log')