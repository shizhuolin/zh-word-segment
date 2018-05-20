# -*- coding: utf-8 -*-
import time, math, pickle, sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def readfileBylines(filename):
    with open(filename, 'rt', encoding='utf8') as f:
        lines = f.readlines()
    return lines

def getTextSequences(lines):
    sequences, result = [[[],[]]], []
    for line in lines:
        if len(line.split('\t')) == 2:
            word,label =  line.split('\t')
            sequences[-1][0].append(word)
            sequences[-1][1].append(label.strip())
        else:
            sequences.append([[],[]])
    for i in range(len(sequences)):
        if len(sequences[i][0]) == len(sequences[i][1]) > 0:
            result.append( dict(enumerate(zip(*sequences[i]))) )
    return result

def statistics_transition_matrix(labelTypes, sequences):
    matrix = {}
    for sequence in sequences:
        prev_label = None
        for i in range(len(sequence)):
            if prev_label not in matrix:
                matrix[prev_label] = {}
            if sequence[i][1] not in matrix[prev_label]:
                matrix[prev_label][sequence[i][1]] = 0
            matrix[prev_label][sequence[i][1]] += 1
            prev_label = sequence[i][1]
    for row in [None]+labelTypes:
        total = sum(matrix[row].values())
        for col in labelTypes:
            matrix[row][col] = matrix[row].get(col,0.0) / total
    return matrix

def generate_features(sequences):
    # define template for input sequence
    templates = [[-2],[-1],[0],[1],[2],
                 [-2,-1],[-1,0],[0,1],[1,2]]
    X, XY, features, total = {},{},{},0
    for sequence in sequences:
        for i in range(len(sequence)):
            for template in templates:
                value = [ sequence.get(i+pos, (None,None))[0] for pos in template]
                x  = (template, value)
                xy = (template, value, sequence[i][1])
                feature = (template, value, sequence[i][1])                
                key_x = str(x)
                key_xy = str(xy)
                key_f = str(feature)                
                if key_x not in X: X[key_x] = [x, 0]
                if key_xy not in XY: XY[key_xy] = [xy, 0]
                if key_f not in features: features[key_f] = feature
                X[key_x][1] += 1
                XY[key_xy][1] += 1
                total += 1
    features = list(features.values())
    weights = np.zeros((len(features)))
    featureHashTable = {}
    for i, feature in enumerate(features):
        featureHashTable[str(feature)] = (feature, i)
    X = dict([(k, (x, c/total) ) for k,(x,c) in X.items()])
    XY = dict([(k, (xy, c/total) ) for k,(xy,c) in XY.items()])
    return templates, features, featureHashTable, weights, X, XY

#@nb.jit
def model_probability(xy, featureHashTable, weights, labelTypes):
    allyx = {}
    for _y in labelTypes:
        key_f = str((xy[0],xy[1],_y))
        allyx[_y] = math.exp(weights[featureHashTable[ key_f ][1]]) if key_f in featureHashTable else 1.0
    return allyx[xy[2]] / sum(allyx.values())

def opt_fun(weights, features, featureHashTable, labelTypes, data_x, data_xy):
    return -1.0 * sum([p * math.log( model_probability(xy, featureHashTable, weights, labelTypes) ) for xy,p in data_xy.values()])

def grads_fun(weights, features, featureHashTable, labelTypes, data_x, data_xy):
    return np.asarray( [data_x[str((features[i][0], features[i][1]))][1] * model_probability(features[i],featureHashTable, weights, labelTypes) - data_xy[str(features[i])][1] for i in range( len(features) )] )

#sys.path.append('./memm_train_text.txt')
#sys.path.append('./memm_model.pkl')

assert len(sys.argv) == 3

_starTime = time.time()
filename_ = sys.argv[1]
defineLabelTypes_ = ['B','M','E','S']
sequences_ = getTextSequences(readfileBylines(filename_))
ransition_matrix_ = statistics_transition_matrix(defineLabelTypes_, sequences_)
templates_, features_, featureHashTable_, weights_, data_X_, data_XY_ = generate_features(sequences_)

#opt_fun(weights_, features_, featureHashTable_, defineLabelTypes_, data_X_, data_XY_)
#grads_fun(weights_, features_, featureHashTable_, defineLabelTypes_, data_X_, data_XY_)

print( 'templates:', len(templates_))
print( 'features:', len(features_))

iter_n = 0
def callback(xk):
    global iter_n
    v = opt_fun(xk, features_, featureHashTable_, defineLabelTypes_, data_X_, data_XY_)
    iter_n += 1
    print('iter:', iter_n, 'f:', v)
    
x,_,_ = fmin_l_bfgs_b(func=opt_fun, x0=weights_, fprime=grads_fun,
                       args=(features_, featureHashTable_, defineLabelTypes_, data_X_, data_XY_),
                       callback=callback, disp=1)

features = [(features_[i], x[i]) for i in range(x.shape[0])]
with open(sys.argv[2], 'wb') as f:
    pickle.dump((ransition_matrix_, features), f)

_endTime = time.time()
print( 'execute time: %fs' % (_endTime - _starTime) )
