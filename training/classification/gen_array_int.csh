#!/bin/csh

rm -fr ./image ./layers ./weights

# python ../../inference/image_classification_from_scratch/inference_icfs_snn.py --mdl ./cats_dogs_giraffes_elephants_lions_classification.tflite --img ./dog.jpg --int
# python ../../inference/image_classification_from_scratch/inference_icfs_snn.py --mdl ./cats_dogs_giraffes_elephants_lions_classification.tflite --img ./cat.jpg --int
python ./pars_nobunet.py  --mdl ./cats_dogs_giraffes_elephants_lions_classification.tflite --int
