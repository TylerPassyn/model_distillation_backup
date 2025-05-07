import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from ast import literal_eval 
from transformers import ViTForImageClassification
from PIL import Image
import time



#load in image /deac/csc/classes/csc373/passta23/model_distillation_backup/Data/Demo_Examples/Ostrich.jpg
image_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/Demo_Examples/Ostrich.jpg"


#convert image to tensor 

#/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Initial_Testing/distilled_model_updated
distilled_model_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/distilled_student_Hidden_Layers_1_epochs_25_hidden_layers_1_temperature_4.0" 
#load distilled model
model  = ViTForImageClassification.from_pretrained(distilled_model_path)
model.eval()

#load in the image
image = Image.open(image_path)
#convert image to tensor
from transformers import ViTImageProcessor
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
inputs = image_processor(images=image, return_tensors="pt")
#convert inputs to tensor
inputs = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
#/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/class_labels.csv
labels = pd.read_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/class_labels.csv")
#convert the labels to a list

#run the model
with torch.no_grad():
    outputs = model(pixel_values=inputs)  # Forward pass through the model the features
    logits = outputs.logits  # Get the predicted class indices
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    print(f"Predicted label: {predicted_label}")
    #get the predicted label based upon the predicted class index using the labels list
    labels = labels.iloc[predicted_class_idx]
    print(f"Predicted label: {labels}")



