

from datasets import load_dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import ViTForImageClassification

# Load the dataset
ds = load_dataset("imagenet-1k")

# Use a random split of the dataset, no more than 1000 images
ds = ds['train'].shuffle(seed=42).select(range(1000))

# Preprocess the images
def preprocess_image(example):
    image = example['image'].convert("RGB").resize((224, 224))  # Resize to 224x224 for ViT
    example['pixel_values'] = np.array(image) / 255.0  # Normalize pixel values
    return example

ds = ds.map(preprocess_image)

# Define the model (replace with your actual model)
model = ViTForImageClassification.from_pretrained("/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Systematic_Testing/distilled_student_Temperature_3_epochs_25_hidden_layers_8_temperature_4.0")
model.eval()

# Evaluate the model
def to_torch_tensor(example):
    example['pixel_values'] = torch.tensor(example['pixel_values']).permute(2, 0, 1).float()  # HWC to CHW
    example['label'] = torch.tensor(example['label'])
    return example

ds = ds.map(to_torch_tensor)

# Evaluate the model
all_preds = []
all_labels = []

with torch.no_grad():
    for example in ds:
        inputs = example['pixel_values'].unsqueeze(0)  # Add batch dimension
        labels = example['label'].unsqueeze(0)
        outputs = model(inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy on the subset of ImageNet: {accuracy * 100:.2f}%")


