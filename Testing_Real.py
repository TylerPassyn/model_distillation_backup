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

#get the images from /deac/csc/classes/csc373/passta23/model_distillation_backup/imagenet-sample-images-master

images_directory = '/deac/csc/classes/csc373/passta23/model_distillation_backup/imagenet-sample-images-master'


for file in os.listdir(images_directory):
    if file.endswith('.JPEG') or file.endswith('.jpg'):
        
        