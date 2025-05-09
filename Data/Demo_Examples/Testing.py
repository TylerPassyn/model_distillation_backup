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
from torch.utils.data import DataLoader
import tqdm

#load in a small subset of the image net dataset, the one with the 1000 classes

from datasets import load_dataset

ds = load_dataset("zh-plus/tiny-imagenet")


#load in image /deac/csc/classes/csc373/passta23/model_distillation_backup/Data/Demo_Examples/Ostrich.jpg
image_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/Demo_Examples/Tiger_Shark.jpg"


#convert image to tensor 

#/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Initial_Testing/distilled_model_updated
distilled_model_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Systematic_Testing/distilled_student_Temperature_3_epochs_25_hidden_layers_8_temperature_4.0" 
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

actual_model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224")

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



#now run the actual model
with torch.no_grad():
    outputs = actual_model(pixel_values=inputs)  # Forward pass through the model the features
    logits = outputs.logits  # Get the predicted class indices
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = actual_model.config.id2label[predicted_class_idx]
    print(f"Predicted label: {predicted_label}")

#now use the data set to test accuracy of both the models
#ds = load_dataset("zh-plus/tiny-imagenet") 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
actual_model.to(device)

# confirm splits available
print("Available splits:", ds.keys())

# the val split is named "val"
split_name = "valid"

def preprocess_batch(batch):
    # 1) Convert all to RGB
    imgs = [img.convert("RGB") for img in batch["image"]]
    # 2) Use those RGB images for processing
    pixel_vals = image_processor(images=imgs, return_tensors="pt")["pixel_values"]
    return {"pixel_values": pixel_vals, "labels": torch.tensor(batch["label"], dtype=torch.long)}

# map over the 'val' split
ds_val = ds[split_name].map(preprocess_batch, batched=True, batch_size=32)
ds_val.set_format(type="torch", columns=["pixel_values", "labels"])

val_loader = DataLoader(ds_val, batch_size=32, shuffle=False)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=inputs)
            preds = outputs.logits.argmax(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

acc_distilled = evaluate(model, val_loader)
acc_full     = evaluate(actual_model, val_loader)

print(f"Distilled model accuracy on Tiny-ImageNet val: {acc_distilled*100:.2f}%")
print(f"Full ViT-Large accuracy on Tiny-ImageNet val: {acc_full*100:.2f}%")
