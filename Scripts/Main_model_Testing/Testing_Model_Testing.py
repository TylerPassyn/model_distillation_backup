import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from ast import literal_eval 
from transformers import ViTForImageClassification



#load logits and labels
#logits: /deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/logits_output.csv
#labels: /deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/extracted_labels.csv
logits_test = pd.read_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/logits_output.csv")
print(f"Column names in logits_test: {logits_test.columns}")

logits_test = logits_test[['Logits']] # Select only the 'Logits' column

logits_test['Logits'] = logits_test['Logits'].apply(literal_eval)  # Convert string to list
logits_test = np.array(logits_test['Logits'].tolist())  # Convert to a NumPy array
labels_test = pd.read_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/extracted_labels_with_ids.csv")

print(f"Column names in labels_test: {labels_test.columns}")
#get the id column from the labels_test
labels_test = labels_test[['ID']] # Select only the 'ID' column
# Convert the 'ID' column to a NumPy array
labels_test = np.array(labels_test)  # Convert to a NumPy array

print(f"Logits test header: {logits_test[0]}")
print(f"Logits test shape: {logits_test.shape}")
print(f"Labels test header: {labels_test[0]}")
print(f"Labels test shape: {labels_test.shape}")






#/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Initial_Testing/distilled_model_updated
distilled_model_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Initial_Testing/distilled_model_updated" 
save_directory = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Initial_Testing"
#load distilled model
model  = ViTForImageClassification.from_pretrained(distilled_model_path)
model.eval()

#/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/all_features.npy
features = np.load("/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/all_features.npy")

# Convert features to PyTorch tensor
features = torch.tensor(features, dtype=torch.float32)

with torch.no_grad():
        logits_test = torch.tensor(logits_test, dtype=torch.float32)
        labels_test = torch.tensor(labels_test, dtype=torch.long)
        outputs = model(features)  # Forward pass through the model the features
        predicted = torch.argmax(outputs.logits, dim=1)  # Get the predicted class indices
        accuracy = accuracy_score(labels_test, predicted)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    # Save the distilled model

