from PIL import Image
from transformers import ViTFeatureExtractor
from transformers import ViTConfig, ViTForImageClassification
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor

from Scripts.Model_Distillation.Image_Distiller import Distill_Model_VIT_to_VIT as Distill_VIT

from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import os
import sys



def train_and_test_model(logits_train, extracted_features, class_labels, logits_test, labels_test, num_classes, model_save_directory, num_hidden_layers=8, hidden_size=512, attention_heads=12, temperature=4.0, num_epochs=5, learning_rate=0.001):

    # Create a new instance of the model
    # num_classes, num_hidden_layers = 8, temperature = 4.0, num_epoches = 5, hidden_size=768, attention_heads=12, learning_rate = 0.01, class_names = None
    distiller = Distill_VIT(
        num_hidden_layers=num_hidden_layers,
        num_classes=num_classes,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        temperature=temperature,
        num_epoches=num_epochs,
        class_names=class_labels,
        learning_rate=learning_rate
    )

    distilled_model = distiller.distill(logits_train, extracted_features)

    #evaluate the model
    distilled_model.eval()
    with torch.no_grad():
        logits_test = torch.tensor(logits_test, dtype=torch.float32)
        labels_test = torch.tensor(labels_test, dtype=torch.long)
        outputs = distilled_model(logits_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(labels_test, predicted)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    # Save the distilled model
    distiller.save_model(model_save_directory)
    return distilled_model, accuracy
    

    
def testing_num_hidden_layers(logits_train, extracted_features, class_labels, labels_train, logits_test, labels_test, num_classes, model_save_directory, results_file_directory):
    # hidden layer sizes
    hidden_layer_sizes = [2, 4, 6, 8, 10, 12]
    accuracies = []
    model_features = []
    for size in hidden_layer_sizes:
        model, accuracy = train_and_test_model(logits_train, extracted_features, class_labels, labels_train, logits_test, labels_test, num_classes, model_save_directory, num_hidden_layers=size)
        #save each of the features of the model to a csv file
        #num_hidden_layers, hidden_size, attention_heads=12, temperature=4.0, num_epochs=5, learning_rate=0.001
        #get each of these features from the model
        num_hidden_layers = model.num_hidden_layers
        hidden_size = model.hidden_size
        attention_heads = model.attention_heads
        temperature = model.temperature
        num_epochs = model.num_epochs
        learning_rate = model.learning_rate
        #add features to a list
        model_features.append([num_hidden_layers, hidden_size, attention_heads, temperature, num_epochs, learning_rate])
        accuracies.append(accuracy)

    #save all results to a csv 
    #stack the accuracies and model features
    accuracies = np.array(accuracies)
    model_features = np.array(model_features)
    all_results = np.hstack((model_features, accuracies.reshape(-1, 1)))
    #save the accuracies and model features to a csv file
    with open(results_file_directory + '/model_features_hidden_layer_tests.csv', 'w') as f:
        f.write('num_hidden_layers, hidden_size, attention_heads, temperature, num_epochs, learning_rate, accuracy\n')
        for i in range(len(all_results)):
            f.write(f"{model_features[i][0]}, {model_features[i][1]}, {model_features[i][2]}, {model_features[i][3]}, {model_features[i][4]}, {model_features[i][5]}, {accuracies[i][6]}\n")
        
    



def main():
    # image_directory, labels_file, results_file_directory, model_save_directory
    if len(sys.argv) != 4:
        print("Usage: python Main_Model_Accuracy_Testing.py <image_directory> <labels_file> <logits_file> <testing_results_directory> <model_save_directory>") 
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"Error: The directory {sys.argv[1]} does not exist.")
        sys.exit(1)
    if not os.path.isdir(sys.argv[1]):
        print(f"Error: The path {sys.argv[1]} is not a directory.")
        sys.exit(1)
    if not os.path.exists(sys.argv[2]):
        print(f"Error: The file {sys.argv[2]} does not exist.")
        sys.exit(1)
    if not os.path.isfile(sys.argv[2]):
        print(f"Error: The path {sys.argv[2]} is not a file.")
        sys.exit(1)
    if not os.path.exists(sys.argv[3]):
        print(f"Error: The directory {sys.argv[3]} does not exist.")
        sys.exit(1)
    if not os.path.isdir(sys.argv[3]):
            print(f"Error: The path {sys.argv[3]} is not a directory.")
            sys.exit(1)

    image_directory = sys.argv[1]
    labels_file = sys.argv[2]
    results_file_directory = sys.argv[3]

    # Load the data
    logits_train = np.load(os.path.join(image_directory, 'logits_train.npy'))
    extracted_features = np.load(os.path.join(image_directory, 'extracted_features.npy'))
    class_labels = np.load(os.path.join(image_directory, 'class_labels.npy'))
    logits_test = np.load(os.path.join(image_directory, 'logits_test.npy'))
    labels_test = np.load(os.path.join(image_directory, 'labels_test.npy'))

    num_classes = len(class_labels)

    # Train and test the model
    model_save_directory = os.path.join(results_file_directory, 'distilled_model')
    
    testing_num_hidden_layers(logits_train, extracted_features, class_labels, logits_train, logits_test, labels_test, num_classes, model_save_directory, results_file_directory)
    

if __name__ == "__main__":
    main()

