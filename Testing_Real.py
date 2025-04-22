#/deac/csc/classes/csc373/passta23/model_distillation_backup/distilled_model
from transformers import ViTForImageClassification
import pandas as pd

#get the feature extractor
from transformers import ViTImageProcessor
from PIL import Image
import numpy as np

import os
import sys

import time


print("Testing the model on real images")
#get the images from /deac/csc/classes/csc373/passta23/model_distillation_backup/imagenet-sample-images-master

images_directory = '/deac/csc/classes/csc373/passta23/model_distillation_backup/imagenet-sample-images-master'

#get the image meta data to get the labels

#/deac/csc/classes/csc373/passta23/model_distillation_backup/imagenet-sample-images-master/gallery.md

image_meta_data = pd.read_csv('/deac/csc/classes/csc373/passta23/model_distillation_backup/imagenet-sample-images-master/gallery.md', sep='|', header=None)

print(image_meta_data.head())