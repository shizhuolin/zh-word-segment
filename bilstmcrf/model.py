# -*- coding: utf-8 -*-
import os
import collections
import gensim
import numpy as np
import tensorflow as tf

# const int INFO = 0;            // base_logging::INFO;
# const int WARNING = 1;         // base_logging::WARNING;
# const int ERROR = 2;           // base_logging::ERROR;
# const int FATAL = 3;           // base_logging::FATAL;
# const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# tf.logging.DEBUG = 10
# tf.logging.INFO = 20
# tf.logging.WARN = 30
# tf.logging.ERROR = 40
# tf.logging.FATAL = 50
tf.logging.set_verbosity(tf.logging.WARN)

np.random.seed(seed=123)

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

def build_dataset(word2vecfname, sequences):
  model = gensim.models.KeyedVectors.load_word2vec_format(word2vecfname, binary=False)
  char_vector = np.zeros([len(model.index2word)+1, model.vector_size])
  char_dictionary = dict(UNK=0)
  for w in model.index2word:
    char_vector[len(char_dictionary), :] = model.wv[w]
    char_dictionary[w] = len(char_dictionary)
    
  label_translation_count = collections.defaultdict(lambda: 0)
  label_dictionary = dict()  
  for sequence in sequences:
    for i, (character, label) in enumerate(zip(*sequence)):
      if i>0: label_translation_count[(sequence[1][i-1],sequence[1][i])] += 1
      if label not in label_dictionary:
        label_dictionary[label] = len(label_dictionary)
        
  data = []
  for sequence in sequences:
    charrow, labelrow = [], []
    for character, label in zip(*sequence):
      if character in char_dictionary:
        char_index = char_dictionary[character]
      else:
        char_index = 0 # dictionary['UNK']
      label_index = label_dictionary[label]
      charrow.append( char_index )
      labelrow.append( label_index )
    data.append( [charrow, labelrow] )
    
  label_transition_proba = np.zeros([len(label_dictionary), len(label_dictionary)])
  for prev_label, prev_index in label_dictionary.items():
    for label, index in label_dictionary.items():
      label_transition_proba[prev_index, index] = label_translation_count[(prev_label, label)]  
  label_transition_proba = np.divide(label_transition_proba, np.sum(label_transition_proba) )
    
  char_reverse_dictionary = dict(zip(char_dictionary.values(), char_dictionary.keys()))
  label_reverse_dictionary = dict(zip(label_dictionary.values(), label_dictionary.keys()))
  return char_vector, char_dictionary, char_reverse_dictionary,label_dictionary,label_reverse_dictionary,label_transition_proba,data

class BI_LSTM_CRF:
  
  def __init__(self, sess, dtype, char2vector, transition_tmatrix, hidden_size, ntags):
    
    self.sess = sess
    self.global_step = tf.Variable(0, trainable=False)
    self.sentence_inputs = tf.placeholder(tf.int32, shape=[None, None], name="words")
    self.sentence_labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
    self.sentence_lengths = tf.placeholder(tf.int32, shape=[None,])
    self.embeddings = tf.Variable(char2vector, dtype=dtype, name="embeddings", trainable=False)
    self.transition_tmatrix = tf.Variable(transition_tmatrix, dtype=dtype, name="transition_matrix", trainable=False)
    
    self.char_embeddings = tf.nn.embedding_lookup(self.embeddings, self.sentence_inputs)
    
    def cell():
      return tf.contrib.rnn.GRUCell(hidden_size)
      #return tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)
    
    cell_fw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(1)], state_is_tuple=True)
    cell_bw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(1)], state_is_tuple=True)
    
    (output_fw, output_bw), ( _, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,
                                                                      self.char_embeddings,
                                                                      sequence_length=self.sentence_lengths,
                                                                      dtype=dtype)
    
    self.W = tf.get_variable("W", shape=[2*hidden_size, ntags], dtype=dtype, initializer=tf.zeros_initializer())
    self.b = tf.get_variable("b", shape=[ntags], dtype=dtype, initializer=tf.zeros_initializer())
    
    context_rep = tf.concat([output_fw, output_bw], axis=-1)
    ntime_steps = tf.shape(context_rep)[1]
    context_rep_flat = tf.reshape(context_rep, [-1, 2*hidden_size])
    pred = tf.matmul(context_rep_flat, self.W) + self.b
    self.scores = tf.reshape(pred, [-1, ntime_steps, ntags])
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, 
                                                                          self.sentence_labels,
                                                                          self.sentence_lengths,
                                                                          self.transition_tmatrix)
    self.loss = tf.reduce_mean(-log_likelihood)
    optimizer = tf.train.AdamOptimizer()
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm( tf.gradients(self.loss, tvars), 5 )
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    tf.summary.scalar('loss', self.loss)
    self.merged = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter( './tmp/', sess.graph)
    self.saver = tf.train.Saver( tf.global_variables() )
    
  def save(self, savepath):
    self.saver.save(self.sess, savepath, global_step=self.global_step)
    
  def restore(self, restorepath):
    self.saver.restore( self.sess, restorepath )
    
  def step(self, i, data):
    sentences_x, sentences_y, sentences_w, sentence_lengths = data
    
    loss, summary, _ = self.sess.run( [ self.loss, self.merged, self.train_op ], feed_dict={self.sentence_inputs: sentences_x,
                                                                                            self.sentence_labels: sentences_y,
                                                                                            self.sentence_lengths: sentence_lengths} )
    self.writer.add_summary(summary, i)
    return loss
  
  def valid(self, data):
    sentences_x, sentences_y, sentences_w, sentence_lengths = data
    loss = self.sess.run( self.loss, feed_dict={self.sentence_inputs: sentences_x,
                                                self.sentence_labels: sentences_y,
                                                self.sentence_lengths: sentence_lengths} )
    return loss
  
  def test(self, sentence, label_transition_proba):
    scores = self.sess.run(self.scores,feed_dict={self.sentence_inputs: sentence,
                                                  self.sentence_lengths: [len(sentence[0])]} )
    viterbi, _ = tf.contrib.crf.viterbi_decode(np.squeeze( scores, [0] ), label_transition_proba)
    return viterbi
  
  def train(self, modelsavepath, data, batch_size=256):
    valid_ids = np.random.randint( low=0, high=len( data ),  size=np.ceil(len( data ) / 5) )
    train_data, valid_data = [], []
    for i in range( len(data) ):
      if i in valid_ids:
        valid_data.append( data[i] )
      else:
        train_data.append( data[i] )
    last_loss=np.inf
    historiesloss = []
    while True:
      loss = self.step(i, self.get_batch( train_data, batch_size) )
      if i % 3==0:
        loss = np.mean( [ self.valid( self.get_batch(valid_data[s:s+batch_size]) ) for s in range(0, len(valid_data), batch_size) ]  )
        historiesloss.append( loss )
        print(i,  loss )
        if len(historiesloss) >= 30:
          firstscore = np.mean( historiesloss[-30:-15] )
          secondscore = np.mean( historiesloss[-15:] )
          if loss < last_loss:
            last_loss = loss
            #print( 'save', last_loss )
            self.save(modelsavepath)
          if firstscore < secondscore:
            print( "train complate" )
            break
          
  def get_batch(self, data, size=np.inf):
    # get a random batch of data by specified batch_size
    sentences = []
    max_sentence_length = 0
    sentence_lengths = []
    size = min(size, len(data))
    for i in np.random.randint(low=0, high=len(data), size=size):
      sentences.append( data[i] )
      sentence_lengths.append(len(data[i][0]))
    max_sentence_length = np.max(sentence_lengths)
    sentences_x, sentences_y, sentences_w = [], [], []
    for words, labels in sentences:
      sentences_x.append( words + [0] * (max_sentence_length-len(words)) )
      sentences_y.append( labels + [0] * (max_sentence_length-len(labels)) )
      sentences_w.append( [1]*len(labels) + [0]*(max_sentence_length-len(labels)) )
    return sentences_x, sentences_y, sentences_w, sentence_lengths
