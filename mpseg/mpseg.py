# -*- coding: utf-8 -*-
import sys
import os.path
import math
import nltk
from nltk.corpus import PlaintextCorpusReader

# sys.argv.append('./gold/pku_training_words.utf8')
# sys.argv.append('./training/pku_training.utf8')
# sys.argv.append('./testing/pku_test.utf8')

assert len(sys.argv) == 4

with open(sys.argv[1], 'rt', encoding='utf8') as f:
    training_words = [w.strip() for w in f.readlines()]

training = PlaintextCorpusReader( *os.path.split(sys.argv[2]) )
training_words += list(training.words())
#training_words = list(training.words())
N = len(training_words)
V = len( set(training_words) )
fdist = nltk.FreqDist(training_words)
fdist = dict([(w, math.log((c+1.0)/(N+V))) for w, c in fdist.items()])
defprob = math.log(1.0/(N+V))

with open(sys.argv[3], 'rt', encoding='utf8') as f:
    test = f.readlines()

def get_DAG(sentence):
    DAG = {}
    T = len(sentence)
    for x in range(T):
        ys = []
        for y in range(x+1, T+1):
            if sentence[x:y] in fdist:
                ys.append(y)
        if not ys:
            ys.append(x+1)
        DAG[x] = ys
    return DAG

def dfs(DAG, sentence):
    segments = []
    T = len(sentence)
    def _dfs(words, x):
        for y in DAG[x]:
            if y < T:
                new = words.copy()
                new.append(sentence[x:y])
                _dfs(new, y)
            else:
                new = words.copy()
                new.append(sentence[x:y])
                segments.append( new )
    _dfs([], 0)
    bestseg = max([(sum(fdist.get(w, defprob) for w in seg), seg) for seg in segments])
    return bestseg[1]
    
def dp(DAG, sentence):
    T = len(sentence)
    prob = {T:(0.0,)}
    for x in range(T-1, -1, -1):
        prob[x] = max([(fdist.get(sentence[x:y], defprob) + prob[y][0], y) for y in DAG[x]])
    x = 0
    bestseg = []
    while x < T:
        y = prob[x][1]
        bestseg.append( sentence[x:y] )
        x = y
    return bestseg

for sent in test:
    DAG = get_DAG(sent)
    #seg1 = dfs(DAG, sent)
    seg2 = dp(DAG, sent)
    #print('  '.join(seg1), sep='', end='')
    print('  '.join(seg2), sep='', end='')
    #break