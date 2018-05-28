#!/bin/sh
vocab_corpus='../icwb2-data/training/'
wordvecfilename='./word2vec.txt'
vocab_min_count=5
word2vec_dim=64
trainfile="./pku_training_bilstmcrf.utf8"
modelcheckpointpath='./checkpoint/'
hidden_size=192
batch_size=128
testfile="pku_test_liblstmcrf_segmentation.utf8"

rm -fr $wordvecfilename $trainfile $testfile
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
