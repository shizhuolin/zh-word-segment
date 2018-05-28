# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import tensorflow as tf
from model import *

assert len(sys.argv) == 6

modelsavepath = sys.argv[1]
filename = sys.argv[2]
hidden_size=int(sys.argv[3])
batch_size=int(sys.argv[4])
word2vec=sys.argv[5]

_starTime_ = time.time()

_starTime = time.time()
print( 'load data ... ', end='', flush=True)
sequences_ = getTextSequences(filename)
print( 'sequences total:%d time:%fs' % (len(sequences_),time.time() - _starTime), flush=True)

_starTime = time.time()
print( 'build dataset ... ', end='', flush=True)
char2vector_, char_dictionary_, char_reverse_dictionary_, label_dictionary_, label_reverse_dictionary_, label_transition_proba_, data_ = build_dataset(word2vec, sequences_)
print( 'chartable size:%d time:%fs' % (len(char_dictionary_),time.time() - _starTime), flush=True)



config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

model = BI_LSTM_CRF(sess=sess,
                    dtype=tf.float32,
                    char2vector=char2vector_,
                    transition_tmatrix=label_transition_proba_,
                    hidden_size=hidden_size,
                    ntags=len(label_dictionary_))

print(' train and save model ')
sess.run( tf.global_variables_initializer() )
model.train(modelsavepath, data_, batch_size=batch_size)

print( 'execute time: %fs' % (time.time() - _starTime_) )
