# -*- coding: utf-8 -*-
import sys, re, math
import numpy as np

#sys.argv.append('./gold/pku_training_words.utf8')
#sys.argv.append('./training/pku_training.utf8')
#sys.argv.append('./testing/pku_test.utf8')

assert len(sys.argv) == 4

training_words_filename = sys.argv[1]
training_filename = sys.argv[2]
test_filename = sys.argv[3]

with open(training_words_filename, 'rt', encoding='utf8') as f:
    training_words = f.readlines()

with open(training_filename, 'rt', encoding='utf8') as f:
    training = f.readlines()
    
with open(test_filename, 'rt', encoding='utf8') as f:
    test = f.readlines()

# training += training_words
# word tag by char
hidden_state = ['B','M','E','S']
A, B, P = {}, {}, {}
_N = 0
_O = {}
for line in training:
    #print( line )
    prev_a = None
    for w, word in enumerate(re.split(r'\s{2}', line)):
        I = len(word)
        _N += I
        for i, c in enumerate(word):
            _O[c] = _O.get(c, 0) + 1
            if I == 1:
                a = 'S'
            else:
                if i == 0:
                    a = 'B'
                elif i == I-1:
                    a = 'E'
                else:
                    a = 'M'
            # print(w, i, c, a)
            if prev_a is None: # calculate Initial state Number
                if a not in P: P[a] = 0
                P[a] += 1
            else: # calculate State transition Number
                if prev_a not in A: A[prev_a] = {}
                if a not in A[prev_a]: A[prev_a][a] = 0
                A[prev_a][a] += 1
            prev_a = a
            # calculate Observation Number
            if a not in B: B[a] = {}
            if c not in B[a]: B[a][c] = 0
            B[a][c] += 1
_B = B.copy()            
# calculate probability
for k, v in A.items():
    total = sum(v.values())
    A[k] = dict([(x, math.log(y / total)) for x, y in v.items()])
for k, v in B.items():
    # plus 1 smooth
    total = sum(v.values())
    V = len(v.values())
    B[k] = dict([(x, math.log((y+1.0) / (total+V))) for x, y in v.items()])
    # plus 1 smooth
    B[k]['<UNK>'] = math.log(1.0 / (total+V))
minlog = math.log( sys.float_info.min )
total = sum(P.values())
for k, v in P.items():
    P[k] = math.log( v / total )

A2,B2 = {}, {}
for line in training:
    temptags = []
    tempsent = []
    for w, word in enumerate(re.split(r'\s{2}', line)):
        I = len(word)
        for i, c in enumerate(word):
            if I == 1:
                a = 'S'
            else:
                if i == 0:
                    a = 'B'
                elif i == I-1:
                    a = 'E'
                else:
                    a = 'M'
            temptags.append(a)
            tempsent.append(c)
            if len(temptags) >= 3:
                if temptags[-3] not in A2: A2[temptags[-3]] = {}
                if temptags[-2] not in A2[temptags[-3]]: A2[temptags[-3]][temptags[-2]] = {}
                if temptags[-1] not in A2[temptags[-3]][temptags[-2]]: A2[temptags[-3]][temptags[-2]][temptags[-1]] = 0
                A2[temptags[-3]][temptags[-2]][temptags[-1]] += 1
            if len(temptags) >= 2:
                if temptags[-2] not in B2: B2[temptags[-2]] = {}
                if temptags[-1] not in B2[temptags[-2]]: B2[temptags[-2]][temptags[-1]] = {}
                if tempsent[-1] not in B2[temptags[-2]][temptags[-1]]: B2[temptags[-2]][temptags[-1]][tempsent[-1]] = 0
                B2[temptags[-2]][temptags[-1]][tempsent[-1]] += 1
    #print( temptags, tempsent )
    #break

# calculate A2 log probabilitis
for i in A2:
    for j in A2[i]:
        total = sum([A2[i][j][k] for k in A2[i][j]])
        for k in A2[i][j]:
            A2[i][j][k] = math.log( A2[i][j][k] / total )
_A = np.array( [0,0,0] )
for i in B2:
    for j in B2[i]:
        total = sum([B2[i][j][k] for k in B2[i][j]])
        V = len( B2[i][j] )
        for k in B2[i][j]:
            # TnT smooth
            a3 = ((B2[i][j][k] - 1.) / (total - 1.)) if total > 1 else 0
            a2 = (_B[j][k] - 1.) / ( sum([_B[j][n] for n in _B[j]]) - 1. )
            a1 = (_O[k] - 1.) / (_N - 1.)
            _A[np.argmax([a1,a2,a3])] += B2[i][j][k]
            B2[i][j][k] = math.log( ( B2[i][j][k] + 1 ) / (total + V) )
        B2[i][j]['<UNK>'] = math.log( 1.0 / (total + V) )

_A = _A / _A.sum()
for o in _O:
    _O[o] = math.log( (_O[o] + 1.0) / (_N + len(_O)) )
_O['<UNK>'] = math.log( 1.0 / (_N + len(_O)) )
def viterbi2(observation):
    def _A2(i,j,k):
        key = ''.join([i,j,k])
        if key.find('BB') >=0 : return minlog
        if key.find('MB') >=0 : return minlog
        if key.find('SM') >=0 : return minlog
        if key.find('EM') >=0 : return minlog
        if key.find('EE') >=0 : return minlog
        if key.find('SE') >=0 : return minlog
        if key.find('BS') >=0 : return minlog
        if key.find('MS') >=0 : return minlog
        try:
            return A2[i][j][k]
        except Exception as e:
            print( i, j, k)
            raise e
            
    def _B2(i,j,o):
        #return B[j].get(o, B[j]['<UNK>'])
        key = ''.join([i,j])
        if key == 'BB': return minlog
        if key == 'MB': return minlog
        if key == 'SM': return minlog
        if key == 'EM': return minlog
        if key == 'EE': return minlog
        if key == 'SE': return minlog
        if key == 'BS': return minlog
        if key == 'MS': return minlog
        #if o not in B2[i][j]:
        #    return B[j].get(o, B[j]['<UNK>'])
        #return B2[i][j].get(o, B2[i][j]['<UNK>'])
        return _A[0] * _O.get(o, _O['<UNK>']) + _A[1] * B[j].get(o, B[j]['<UNK>']) + _A[2] * B2[i][j].get(o, B2[i][j]['<UNK>'])
    
    state = ['B','M','E','S']
    T = len(observation)
    delta = [None] * (T + 1)
    psi = [None] * (T + 2)
    for i in state:
        if delta[1] is None: delta[1] = {}
        if i not in delta[1]: delta[1][i] = {}
        for j in state:
            delta[1][i][j] = P.get(i, minlog) + B[i].get(observation[0], B[i]['<UNK>']) + A[i].get(j, minlog) + _B2(i, j, observation[1])
    for t in range(2, T):
        Ot = observation[t]
        if delta[t] is None: delta[t] = {}
        if psi[t] is None: psi[t] = {}
        for j in state:
            if j not in delta[t]: delta[t][j] = {}
            if j not in psi[t]: psi[t][j] = {}
            for k in state:
                delta[t][j][k], psi[t][j][k] = max( [ (delta[t-1][i][j] + _A2(i,j,k) + _B2(j,k,Ot), i) for i in state] )
    delta[T], (psi[T], psi[T+1]) = max( [ (delta[T-1][i][j], (i, j)) for i in state for j in state] )
    q = [None] * (T+2)
    q[T+1] = psi[T+1]
    q[T] = psi[T]
    for t in range(T-1, 1, -1):
        q[t] = psi[t][q[t+1]][q[t+2]]
    return q[2:]

for sent in test:
    if len(sent) < 2:
        print(sent, sep='', end='')
        continue
    sequence = viterbi2( list(sent) )
    segment = []
    for char, tag in zip(sent, sequence):
        if tag == 'B':
            segment.append(char)
        elif tag == 'M':
            segment[-1] += char
        elif tag == 'E':
            segment[-1] += char
        elif tag == 'S':
            segment.append(char)
        else:
            raise Exception()
    print('  '.join(segment), sep='', end='')
    #break
    
