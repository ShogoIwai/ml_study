#!/bin/csh

python ../../inference/image_classification_from_scratch/inference_icfs_snn.py --mdl cats_dogs_giraffes_elephants_lions_classification.h5 --img ./cat.jpg
python ./pars_nobunet.py

