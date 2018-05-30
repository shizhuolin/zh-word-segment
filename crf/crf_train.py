# -*- coding: utf-8 -*-
import time, sys, pickle
import numpy as np
import numba as nb
import scipy.misc as misc
from scipy.optimize import fmin_l_bfgs_b
import crfext

def getTextSequences(filename):
  sequences, result = [['','']], []
  with open(filename, 'rt', encoding='utf8') as f:
    for line in f.readlines():
      if len(line.split('\t')) == 2:
        word, label =  line.split('\t')
        label = label.strip()
        sequences[-1][0] += word
        sequences[-1][1] += label
      else:
        sequences.append(['',''])
  for sequence in sequences:
    if len(sequence[0]) == len(sequence[1]) > 0:
      result.append( (sequence[0], sequence[1]) )
  return result

def statistics_distribution(sequences):
    # statistics sample probability
    data_x, data_xy = {}, {}
    for sequence in sequences:
        if sequence[0] in data_x:
            data_x[sequence[0]] += 1.0
        else:
            data_x[sequence[0]] = 1.0
        key_xy = sequence
        if key_xy in data_xy:
            data_xy[key_xy][1] += 1.0
        else:
            data_xy[key_xy] = [sequence, 1.0]
    data_x = [ (x, c / len(data_x)) for x, c in data_x.items()]
    data_xy = [ ((x,y), c / len(data_xy)) for (x,y),c in data_xy.values()]
    return data_x, data_xy

# scan template on sequcens and genreate features
def genrate_features(sequences, mincout=1):
    templates = [[-2],[-1],[0],[1],[2], [-2,-1],[-1,0],[0,1],[1,2]]
    yset, features1, features2 = set(), {}, {}
    for sequence in sequences:
        prev_label = '^'
        x = dict(enumerate(sequence[0]))
        y = dict(enumerate(sequence[1]))
        for i in range(len(sequence[0])+1):
            label = y.get(i, '$')
            t = prev_label, label
            if prev_label not in ('^', '$'): yset.add(prev_label)
            if label not in ('^', '$'): yset.add(label)
            if t not in features1: features1[t] = [t,0]
            features1[t][1] += 1
            for template in templates:
                value = [ x.get(i+pos, '\0') for pos in template]
                s = (tuple(template), tuple(value), label )
                if s not in features2: features2[s] = [s,0]
                features2[s][1] += 1
            prev_label = label
    features = list(features1.values()) + list(features2.values())
    features = [feature[0] for feature in features if feature[1] >= mincout]
    yset = dict([(y,i) for i,y in enumerate(yset)])
    #weights = np.random.uniform( -0.1, +0.1, len(features) )
    weights = np.zeros( len(features) )
    return yset, features, weights

def fkyyxi(feature, prev_label, label, x, i):
    value=0.
    if len(feature)==2:
        value=float( feature[0]==prev_label and feature[1]==label )
    elif len(feature)==3 and feature[2]==label:
        eqc = np.sum( [(x[i+pos] if 0<=(i+pos)<len(x) else '\0') == word for pos, word in zip(feature[0], feature[1])] )
        value=float(eqc == len(feature[0]))
    return value
    
def feature_kxy(feature, x, y):
  assert len(x) == len(y)
  n = len(x)
  prev_label = '^'
  value = 0
  for i in range(n+1):
    label = y[i] if i<n else '$'
    value += fkyyxi(feature, prev_label, label, x, i)
    prev_label = label
  return value

def stat_all_xy_features_values(features, data_xy):
  return crfext.stat_all_xy_features_values(features, data_xy)
  values = np.zeros(len(features))
  for k in range(values.shape[0]):
    for (x,y),p in data_xy:
      values[k] += p * feature_kxy(features[k], x, y)
  return values

def yyxi(features, weights, prev_label, label, x, i):
  val = 0
  for k in range(weights.shape[0]):
    if fkyyxi(features[k], prev_label, label, x, i) > 0.5:
      val += weights[k]
  return val

def xmatrixi(yset, features, weights, x, i):
  m = np.empty((len(yset), len(yset)))
  for y0, yi in yset.items():
    for y1, yj in yset.items():
      if (i==0):
        m[yi, yj] = yyxi(features, weights, '^', y1, x, i) if yi == 0 else 0
      elif (i == len(x)):
        m[yi, yj] = yyxi(features, weights, y0, '$', x, i) if yj == 0 else 0
      else:
        m[yi, yj] = yyxi(features, weights, y0, y1, x, i)
  return m

def xmatrices(yset, features, weights, x):
  result = []
  for i in range( len(x)+1 ):
    result.append( xmatrixi(yset, features, weights, x, i) )
  return result

def xalphas(ms):
    mslen = len(ms)
    alpha = np.empty((mslen, ms[0].shape[0]))
    alpha[0,:] = ms[0][0,:]
    for i in range(1,mslen-1):
        alpha[i,:] = [ misc.logsumexp(alpha[i-1,:] + ms[i][:,y1]) for y1 in range(ms[i].shape[1]) ]
    alpha[-1,:] = alpha[-2,:] + ms[-1][:,0]
    return alpha

def xbetas(ms):
    mslen = len(ms)
    beta = np.zeros((mslen, ms[0].shape[0]))
    beta[-1,:] = ms[-1][:,0]
    for i in range(mslen-2, 0, -1):
        beta[i,:] = [ misc.logsumexp(ms[i][y0,:] + beta[i+1,:]) for y0 in range(ms[i].shape[0]) ]
    beta[0,:] = ms[0][0,:] + beta[1,:]
    return beta

def xyproba(yset, ms, logz, y):
    v = ms[0][0, yset[y[0]]]
    v += np.sum( [ms[i][yset[y[i-1]], yset[y[i]]] for i in range(1, len(y))] )
    v += ms[len(y)][yset[y[-1]], 0]
    return np.exp( v - logz )

def xyiproba(yset, ms, alpha, beta, s, i, logz):
    assert 0<=i<len(ms)-1
    return np.exp( alpha[i][yset[s]] + beta[i+1][yset[s]] - logz )

def xyiiproba(yset, ms, alpha, beta, s0, s1, i, logz):
    assert 0<=i<len(ms)
    p1 = -logz
    if i==0:
        p1 += ms[i][0, yset[s1]] + beta[i+1][yset[s1]]
    elif i==len(ms)-1:
        p1 += alpha[i-1][yset[s0]] + ms[i][yset[s0], 0]
    else:
        p1 += alpha[i-1][yset[s0]] + ms[i][yset[s0], yset[s1]] + beta[i+1][yset[s1]]
    return np.exp( p1 )

def model_values(weights, yset, features, data_x):
  return crfext.model_values(list(weights), yset, features, data_x)
  val = 0.
  for x, p in data_x:
    val += p * misc.logsumexp( xalphas( xmatrices(yset, features, weights, x ) )[-1])
  return val

def kx(yset, feature, x, ms, alpha, beta, logz):
  v = 0.
  for s1 in yset:
    if fkyyxi(feature, '^', s1, x, 0) > 0.5:
      v += xyiiproba(yset, ms, alpha, beta, '^', s1, 0, logz)
  for i in range(1, len(x)):
    for s0 in yset:
      for s1 in yset:
        if fkyyxi(feature, s0, s1, x, i) > 0.5:
          v += xyiiproba(yset, ms, alpha, beta, s0, s1, i, logz)
  for s0 in yset:
    if fkyyxi(feature, s0, '$', x, len(x)) > 0.5:
      v += xyiiproba(yset, ms, alpha, beta, s0, '$', len(x), logz)
  return v

def create_kxcache(yset, features, data_x):
  kxtable = []
  for k in range(len(features)):
    feature = features[k]
    kx = []
    for xnum, (x, _) in enumerate(data_x):
      xs0s1i = []
      for s1 in yset:
        if fkyyxi(feature, '^', s1, x, 0) > 0.5:
          xs0s1i.append( ('^', s1, 0) )
      for i in range(1, len(x)):
        for s0 in yset:
          for s1 in yset:
            if fkyyxi(feature, s0, s1, x, i) > 0.5:
              xs0s1i.append( (s0, s1, i) )
      for s0 in yset:
        if fkyyxi(feature, s0, '$', x, len(x)) > 0.5:
          xs0s1i.append( (s0, '$', len(x)) )
      if xs0s1i:
        kx.append( (xnum, xs0s1i) )
    kxtable.append( kx )
  return kxtable

kxtablecache = None
def model_grads1(yset, features, weights, data_x):
  return np.asarray( crfext.model_grads1(yset, features, list(weights), data_x) )
  K = len(weights)
  grads = np.zeros( K )
  malphabetaz = []
  for x, p in data_x:
    ms = xmatrices(yset, features, weights, x)
    alpha = xalphas(ms)
    beta = xbetas(ms)
    logz1 = misc.logsumexp( alpha[-1] )
    logz2 = misc.logsumexp( beta[0] )
    assert np.allclose(logz1, logz2)
    malphabetaz.append( (x, p, ms, alpha, beta, logz1) )
  global kxtablecache
  if not kxtablecache: kxtablecache = create_kxcache(yset, features, data_x)
  for k in range(K):
    for xnum, xs0s1i in kxtablecache[k]:
      _, proba, ms, alpha, beta, logz = malphabetaz[xnum]
      grads[k] += proba * np.sum( [ xyiiproba(yset, ms, alpha, beta, s0, s1, i, logz) for s0, s1, i in xs0s1i ] )
  return grads

def model_grads(yset, features, weights, data_x):
  return np.asarray( crfext.model_grads(yset, features, list(weights), data_x) )
  K = len(weights)
  grads = np.zeros( K )
  malphabetaz = []
  for x, p in data_x:
    ms = xmatrices(yset, features, weights, x)
    alpha = xalphas(ms)
    beta = xbetas(ms)
    logz1 = misc.logsumexp( alpha[-1] )
    logz2 = misc.logsumexp( beta[0] )
    assert np.allclose(logz1, logz2)
    malphabetaz.append( (x, p, ms, alpha, beta, logz1) )
  for k in range(K):
    for x, p, ms, alpha, beta, logz in malphabetaz:
      grads[k] += p * kx(yset, features[k], x, ms, alpha, beta, logz)
  return grads

def optfun(weights, yset, features, all_xy_features_values, data_x, data_xy, sigma):
    _starTime = time.time()
    print( 'optfun:', end='', flush=True)
    val1 = model_values(weights, yset, features, data_x)
    val2 = np.dot( weights, all_xy_features_values )
    l2 = 1. / (2 * np.square(sigma)) * np.dot(weights, weights)
    neglikehold = val1 - val2# + l2
    print('nlh=', neglikehold, end='', flush=True)
    print( ' time:%fs ' % (time.time() - _starTime), end='', flush=True)
    return neglikehold
  
def gradsfun(weights, yset, features, all_xy_features_values, data_x, data_xy, sigma):
    _starTime = time.time()
    print( 'grads:', end='', flush=True)
    grads = model_grads1(yset, features, weights,  data_x)
    grads = grads - all_xy_features_values
    grads = grads# + weights / np.square(sigma)
    print('norm=', np.linalg.norm(grads) , end=' ', flush=True)
    print(' %fs' % (time.time() - _starTime), flush=True)
    return grads

#sys.argv.append('./pku_training_crf.utf8')
#sys.argv.append('./pku_crf_model.pkl')
assert len(sys.argv) == 3

_starTime_ = time.time()
sigma_ = 10.
featuremincout_ = 3

_starTime = time.time()
print( 'load data ... ', end='', flush=True)
sequences_ = getTextSequences(sys.argv[1])
print( 'sequences total:%d time:%fs' % (len(sequences_),time.time() - _starTime), flush=True)

_starTime = time.time()
print( 'stat data distribution ... ', end='', flush=True )
data_x_, data_xy_ = statistics_distribution(sequences_)
print( 'data_x:%d  data_xy:%d  time:%fs' % (len(data_x_),len(data_xy_),time.time() - _starTime), flush=True)

_starTime = time.time()
print( 'genrate features ... ', end='', flush=True )
yset_, features_, weights_ = genrate_features(sequences_, featuremincout_)
print( 'features total:%d time:%fs' % (len(features_), time.time() - _starTime), flush=True)

_starTime = time.time()
print( 'calculate all xy feature values on all data xy ... ', end='', flush=True )
all_xy_features_values_ = np.asarray( stat_all_xy_features_values(features_, data_xy_) )
print( 'time: %fs' % (time.time() - _starTime), flush=True)

def gradcheck(weights, k):
  #fvalue = optfun(weights_, yset_, features_, all_xy_features_values_, data_x_, data_xy_, sigma_)
  grads = gradsfun(weights, yset_, features_, all_xy_features_values_, data_x_, data_xy_, sigma_)
  epsilon=1e-6
  weights[k] -= epsilon
  fvalue1 = optfun(weights, yset_, features_, all_xy_features_values_, data_x_, data_xy_, sigma_)
  weights[k] += 2*epsilon
  fvalue2 = optfun(weights, yset_, features_, all_xy_features_values_, data_x_, data_xy_, sigma_)
  numericgrad = (fvalue2 - fvalue1) / (2*epsilon)
  print('\n', k, numericgrad, grads[k])

# [ gradcheck(weights_, k) for k in range( min(weights_.shape[0], 3) ) ]

def callback(xk):
  with open(sys.argv[2], 'wb') as f:
    pickle.dump( (yset_, features_, xk) , f)

xf,_,_ = fmin_l_bfgs_b(func=optfun, x0=weights_, fprime=gradsfun, args=(yset_, features_, all_xy_features_values_, data_x_, data_xy_, sigma_), callback=callback, disp=0)
print( ' save features and weights' , flush=True )
callback(xf)

print( 'execute time: %fs' % (time.time() - _starTime_) )
