# -*- coding: utf-8 -*-
import sys, re, math

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
A, B, P = {}, {}, {}
for line in training:
    #print( line )
    prev_a = None
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

def viterbi(observation):
    state = ['B','M','E','S']
    T = len(observation)
    delta = [None] * (T + 1)
    psi = [None] * (T + 1)
    delta[0] = dict([(i, P.get(i, minlog) + B[i].get(observation[0], B[i]['<UNK>'])) for i in state])
    psi[0] = dict( [ (i, None) for i in state ] )
    for t in range(1, len(observation)):
        Ot = observation[t]
        delta[t] = dict([(j, max([delta[t-1][i] + A[i].get(j, minlog) + B[j].get(Ot, B[j]['<UNK>']) for i in state])) for j in state])
        psi[t] = dict([(j, max([(delta[t-1][i] + A[i].get(j, minlog), i) for i in state])[1]) for j in state ])
    delta[T] = max( [ delta[T-1][i] for i in state ] )
    psi[T] = max( [(delta[T-1][i], i) for i in state  ] )[1]
    q = [None] * (T+1)
    q[T] = psi[T]
    for t in range(T-1, -1, -1):
        q[t] = psi[t][q[t+1]]
    return q[1:]

for sent in test:
    sequence = viterbi( list(sent) )
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
    