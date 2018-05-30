# -*- coding: utf-8 -*-
import sys, pickle, math
import numpy as np
import numba as nb

#sys.argv.append('./pku_crf_model.pkl')
#sys.argv.append('../icwb2-data/testing/pku_test.utf8')

assert len(sys.argv) == 3

with open(sys.argv[2], 'rt', encoding='utf8') as f:
    test = f.readlines()

with open(sys.argv[1], 'rb') as f:
    yset, features, weights = pickle.load(f)

features1 = {}
features2 = {}
templates = {}
labels = list(yset.keys())
for feature, weight in zip(features, weights):
  if len(feature)==2:
    features1[feature] = weight
  else:
    template, _, _ = feature
    features2[feature] = weight
    if template not in templates: templates[template] = template
templates = list(templates.values())
del feature, weight, template, weights, yset, features

cache = {}
def model_weight(prev_label, label, word):
    cachekey = str((prev_label, label, word))
    if cachekey in cache: return cache[cachekey]
    sequence, index = dict(enumerate(word[0])), word[1]
    f1key = prev_label, label
    weight = features1.get(f1key, 0)
    f2keys = [(template, tuple([sequence.get(index + pos, '\0') for pos in template]), label) for template in templates]
    weight += np.sum( [features2.get(key, 0) for key in f2keys] )
    cache[cachekey] = weight
    return weight
  
def viterbi(observation):
    T = len(observation)
    delta = [None] * (T + 1)
    psi = [None] * (T + 1)
    delta[0] = dict([(label, model_weight('^', label, (observation, 0))) for label in labels])
    psi[0] = dict( [ (label, None) for label in labels ] )
    for t in range(1, len(observation)):
        Ot = observation,t
        delta[t] = dict([(label, max([delta[t-1][prev_label] + model_weight(prev_label, label, Ot)               for prev_label in labels]))    for label in labels])
        psi[t] = dict([(label, max([(delta[t-1][prev_label] + model_weight(prev_label, label, Ot), prev_label) for prev_label in labels])[1]) for label in labels ])
    Ot = observation,T
    delta[T] = max( [ delta[T-1][label]+model_weight(label, '$', Ot) for label in labels ] )
    psi[T] = max( [(delta[T-1][label]+model_weight(label, '$', Ot), label) for label in labels  ] )[1]
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
      if segment:
        segment[-1] += char
      else:
        segment =  [char]
    elif tag == 'E':
      if segment:
        segment[-1] += char
      else:
        segment =  [char]
    elif tag == 'S':
      segment.append(char)
    else:
      raise Exception()
  print('  '.join(segment), sep='', end='')
  #break
    