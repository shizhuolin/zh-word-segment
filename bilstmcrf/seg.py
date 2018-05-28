# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
from model import *


assert len(sys.argv) == 6

word2vec=sys.argv[1]
modelsavepath = sys.argv[2]
filename = sys.argv[3]
hidden_size=int(sys.argv[4])
test_file=sys.argv[5]


sequences_ = getTextSequences(filename)
char2vector_, char_dictionary_, char_reverse_dictionary_, label_dictionary_, label_reverse_dictionary_, label_transition_proba_, data_ = build_dataset(word2vec, sequences_)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

model = BI_LSTM_CRF(sess=sess,
                    dtype=tf.float32,
                    char2vector=char2vector_,
                    transition_tmatrix=label_transition_proba_,
                    hidden_size=hidden_size,
                    ntags=len(label_dictionary_))

ckpt = tf.train.get_checkpoint_state( modelsavepath ) 
model.restore(  ckpt.model_checkpoint_path )

def viterbi(sent):
  #char_dictionary_, label_reverse_dictionary_
  end_char=''
  if sent[-1]=='\n':
    end_char = '\n'
    sent = sent[:-1]
  sentence = [[char_dictionary_.get(w, 0) for w in sent ]]
  labels = model.test( sentence, label_transition_proba_ )
  labels = [ label_reverse_dictionary_[i] for i in labels ]
  return sent, labels, end_char

with open( test_file , 'rt', encoding='utf8') as f:
  test = f.readlines()
for sent in test:
  seq, labels, end = viterbi(sent)
  segment = []
  for char, tag in zip(seq, labels):
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
  print('  '.join(segment), sep='', end=end)
  #break