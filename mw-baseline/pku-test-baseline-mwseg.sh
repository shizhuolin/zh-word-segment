#!/bin/bash

perl ../icwb2-data/scripts/mwseg.pl ../icwb2-data/gold/pku_training_words.utf8 ../icwb2-data/testing/pku_test.utf8 > pku_test_mw_segmentation.utf8
perl ../icwb2-data/scripts/score ../icwb2-data/gold/pku_training_words.utf8 ../icwb2-data/gold/pku_test_gold.utf8 pku_test_mw_segmentation.utf8
