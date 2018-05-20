# -*- coding: utf-8 -*-
import sys, re, os, pickle, math
import numpy as np
import numba as nb

#sys.argv.append('./memm_model.pkl')
#sys.argv.append('./testing/pku_test.utf8')

assert len(sys.argv) == 3

model_file = sys.argv[1]
test_filename = sys.argv[2]

with open(test_filename, 'rt', encoding='utf8') as f:
    test = f.readlines()

with open(model_file, 'rb') as f:
    matrix, features = pickle.load(f)

templates = {}
feature2weight = {}
labels = set()
for (template, value, label), proba in features:
    key = str(template)
    if key not in templates:
        templates[key] = template
    key = str((template, value, label))
    if key not in feature2weight:
        feature2weight[key] = proba
    labels.add(label)
templates = templates.values()

cache = {}
def memm_log_proba(x, y):
    key = str((x,y))
    if key in cache: return cache[key]
    sequence, index = dict(enumerate(x[0])), x[1]
    values = [(template, [sequence.get(index + pos, None) for pos in template]) for template in templates]
    ally = dict([ (_label, math.exp( sum([feature2weight.get(str((value[0],value[1], _label)), 0.0) for value in values]))) for _label in labels])
    log_proba = math.log( ally[y] / sum(ally.values()) )
    cache[key] = log_proba
    return log_proba
        
def model_log_proba(label, prev_label, word):
    proba = memm_log_proba(word, label) + math.log(max(sys.float_info.min, matrix[prev_label][label], sys.float_info.min))
    return proba

def viterbi(observation):
    T = len(observation)
    delta = [None] * (T + 1)
    psi = [None] * (T + 1)
    delta[0] = dict([(i, model_log_proba(i, None, (observation, 0))) for i in labels])
    psi[0] = dict( [ (i, None) for i in labels ] )
    for t in range(1, len(observation)):
        Ot = observation,t
        delta[t] = dict([(j, max([delta[t-1][i] + model_log_proba(j, i, Ot) for i in labels])) for j in labels])
        psi[t] = dict([(j, max([(delta[t-1][i] + model_log_proba(j, i, Ot), i) for i in labels])[1]) for j in labels ])
    delta[T] = max( [ delta[T-1][i] for i in labels ] )
    psi[T] = max( [(delta[T-1][i], i) for i in labels  ] )[1]
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
    