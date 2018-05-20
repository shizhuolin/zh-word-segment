#!/bin/sh

trainfile="./pku_training_crf.utf8"
modelfile="./pku_crf_model.pkl"
testfile="pku_test_crf_segmentation.utf8"

if test ! -e $trainfile
then
  python3 ./crf_train_text_preprocess.py ../icwb2-data/training/pku_training.utf8 > $trainfile
fi

if test ! -e $modelfile
then
  python3 ./crf_train.py $trainfile $modelfile
fi

if test ! -e $testfile
then
  python3 ./crf_seg.py $modelfile ../icwb2-data/testing/pku_test.utf8 > $testfile
fi

perl ../icwb2-data/scripts/score ../icwb2-data/gold/pku_training_words.utf8 ../icwb2-data/gold/pku_test_gold.utf8 $testfile
