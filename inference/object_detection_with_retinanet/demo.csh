#!/bin/csh
python ./inference_odwr_snn.py --dat ./data --mdl ../../training/classification/cats_dogs_giraffes_elephants_lions_classification.h5 --mov ./mov/zoo.mp4 --sfn 1400 --ofs 100
python ./inference_odwr_snn.py --dat ./data --mdl ../../training/classification/cats_dogs_giraffes_elephants_lions_classification.h5 --mov ./mov/zoo.mp4 --sfn 3500 --ofs 100
python ./inference_odwr_snn.py --dat ./data --mdl ../../training/classification/cats_dogs_giraffes_elephants_lions_classification.h5 --mov ./mov/zoo.mp4 --sfn 10000 --ofs 100
