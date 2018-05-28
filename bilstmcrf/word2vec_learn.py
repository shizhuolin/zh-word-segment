# -*- coding: utf-8 -*-
import os, re, sys
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        
    def __iter__(self):
      for fname in os.listdir(self.dirname):
        if fname.endswith('.utf8'):
          for line in open(os.path.join(self.dirname, fname)):
            if fname == 'as_training.utf8' or fname == 'cityu_training.utf8':
              s = re.sub(r'\s{1}','',line)
            else:
              s = re.sub(r'\s{1,2}','',line)
            yield list(s)

assert len(sys.argv) == 5
sentences = MySentences(sys.argv[1])
model = gensim.models.Word2Vec(sentences, size=int(sys.argv[4]), min_count=int(sys.argv[3]), workers=8)
model.wv.save_word2vec_format(sys.argv[2],binary=False)