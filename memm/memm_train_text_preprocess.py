# -*- coding: utf-8 -*-
import re, sys
import numpy as np
import numba as nb

#sys.argv.append('./training/pku_training.utf8')

assert len(sys.argv) == 2

def readfileBylines(filename):
    with open(filename, 'rt', encoding='utf8') as f:
        lines = f.readlines()
    return lines

lines = readfileBylines(sys.argv[1])

for line in lines:
    sent,label = line.replace(r'  ',''), []
    for w, word in enumerate(re.split(r'\s{2}', line)):
        I = len(word)
        for i, c in enumerate(word):
            if I == 1: a = 'S'
            else:
                if i == 0: a = 'B'
                elif i == I-1: a = 'E'
                else: a = 'M'
            label.append(a)
    sent, label = np.asarray(list(sent))[:-1], np.asarray(label)[:-1]
    for s,l in zip(sent,label):
        print('%s\t%s' % (s, l))
    print('')
