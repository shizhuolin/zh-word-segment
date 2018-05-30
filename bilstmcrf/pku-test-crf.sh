#!/bin/sh
vocab_corpus='../icwb2-data/training/'
wordvecfilename='./word2vec.txt'
trainfile="./pku_training_bilstmcrf.utf8"
modelcheckpointpath='./checkpoint/'
testfile="pku_test_liblstmcrf_segmentation.utf8"

vocab_min_count=3
word2vec_dim=512
hidden_size=256
batch_size=256

rm -fr $wordvecfilename
rm -fr $trainfile
rm -fr $testfile
rm -fr $modelcheckpointpath

if test ! -e $wordvecfilename
then
  python3 ./word2vec_learn.py $vocab_corpus $wordvecfilename $vocab_min_count $word2vec_dim
fi

if test ! -e $trainfile
then
  python3 ../crf/crf_train_text_preprocess.py ../icwb2-data/training/pku_training.utf8 > $trainfile
fi

if test ! -e $modelcheckpointpath
then
  mkdir $modelcheckpointpath
  rm -fr ./tmp
  python3 ./learning.py $modelcheckpointpath $trainfile $hidden_size $batch_size $wordvecfilename
fi

if test ! -e $testfile
then
  python3 ./seg.py $wordvecfilename $modelcheckpointpath $trainfile $hidden_size ../icwb2-data/testing/pku_test.utf8 > $testfile
fi

perl ../icwb2-data/scripts/score ../icwb2-data/gold/pku_training_words.utf8 ../icwb2-data/gold/pku_test_gold.utf8 $testfile
