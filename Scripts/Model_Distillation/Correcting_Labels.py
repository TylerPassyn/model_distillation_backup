#This file converts labels from the models
#into the correct labels
# we loop through each model in the directory
#and make the correct id2label mapping

import numpy as np
import pandas as pd
import torch
import os
from transformers import ViTForImageClassification

