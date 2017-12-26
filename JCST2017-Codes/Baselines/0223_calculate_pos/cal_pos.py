import sys
import string
import nltk

testset = str(sys.argv[1])

CONJ = ("CC") # conjunctions
DET = ("DT","PDT") # determiners
PRON = ("PRP", "PRP$") # pronouns
PREP = ("IN", "TO", "RP") # prepositions
WH = ("WDT", "WP", "WP$", "WRB") # wh- questions
NOUN = ("NN", "NNP", "NNPS", "NNS")
VERB = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
ADJ = ("JJ", "JJR", "JJS")
ADV = ("RB", "RBR", "RBS")

cnt = {'CONJ':0,
		'DET':0,
		'PRON':0,
		'PREP':0,
		'WH':0,
		'NOUN':0,
		'VERB':0,
		'ADJ':0,
		'ADV':0,
		'other':0,
		'total':0}

def pct_pos_tag(txt_file, lbl_file, log_file):
	tags = [ [ tag for (word, tag) in nltk.pos_tag(line.split()) ] for line in open(txt_file,'r').readlines() ]
	lbls = [ map(int, line.split()) for line in open(lbl_file,'r').readlines() ]
	for i in xrange(len(lbls)):
		for j in xrange(len(lbls[i])):
			if lbls[i][j] == 0:
				cnt['total'] += 1
				if tags[i][j] in CONJ:
					cnt['CONJ'] += 1
				elif tags[i][j] in DET:
					cnt['DET'] += 1
				elif tags[i][j] in PRON:
					cnt['PRON'] += 1
				elif tags[i][j] in PREP:
					cnt['PREP'] += 1
				elif tags[i][j] in WH:
					cnt['WH'] += 1
				elif tags[i][j] in NOUN:
					cnt['NOUN'] += 1
				elif tags[i][j] in VERB:
					cnt['VERB'] += 1
				elif tags[i][j] in ADJ:
					cnt['ADJ'] += 1
				elif tags[i][j] in ADV:
					cnt['ADV'] += 1
				else:
					cnt['other'] += 1
	# end for
	log = open(log_file,'a+')
	for key, value in cnt.iteritems():
		if not key == 'total':
			log.write('{}: {}%\n'.format( key, float(value*100)/float(cnt['total']) ))

# main
pct_pos_tag(testset+'.in',testset+'.truth',testset+'.log')