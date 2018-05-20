#!/bin/bash

python3 ./hmmseg.py ../icwb2-data/gold/pku_training_words.utf8 ../icwb2-data/training/pku_training.utf8 ../icwb2-data/testing/pku_test.utf8 > pku_test_hmm_segmentation.utf8

perl ../icwb2-data/scripts/score ../icwb2-data/gold/pku_training_words.utf8 ../icwb2-data/gold/pku_test_gold.utf8 pku_test_hmm_segmentation.utf8
