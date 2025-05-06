from PIL import Image
from transformers import ViTFeatureExtractor
from transformers import ViTConfig, ViTForImageClassification
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor

from Image_Distiller import Distill_Model_VIT_to_VIT as Distill_VIT

from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import os
import sys



def train_and_test_model(logits_train, logits_test, features_train, features_test, class_labels, labels_test, num_classes, model_save_directory, num_hidden_layers=8, hidden_size=512, attention_heads=12, temperature=4.0, num_epochs=5, learning_rate=0.001):
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

    #convert logits_train to PyTorch tensor
    logits_train = torch.tensor(logits_train, dtype=torch.float32)
    
    
    distilled_model = distiller.distill(logits_train, extracted_features)

    #evaluate the model
    distilled_model.eval()

    #get

    # Convert logits_test and labels_test to PyTorch tensors
    logits_test = torch.tensor(logits_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    with torch.no_grad():
        logits_test = torch.tensor(logits_test, dtype=torch.float32)
        labels_test = torch.tensor(labels_test, dtype=torch.long)
        outputs = distilled_model(features)  # Forward pass through the model the features
        predicted = torch.argmax(outputs.logits, dim=1)  # Get the predicted class indices
        accuracy = accuracy_score(labels_test, predicted)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    # Save the distilled model
    distiller.save_model(model_save_directory)
    return distilled_model, accuracy
    

    
def testing_num_hidden_layers(logits_train, extracted_features, class_labels, labels_train, logits_test, labels_test, num_classes, model_save_directory, results_file_directory):
    # hidden layer sizes
    print("Testing different hidden layer sizes")
    hidden_layer_sizes = [2, 4, 6, 8, 10, 12]
    accuracies = []
    model_features = []
    for size in hidden_layer_sizes:
        print(f"Testing hidden layer size: {size}")
        model, accuracy = train_and_test_model(
            logits_train = logits_train, 
            extracted_features = extracted_features, 
            class_labels = class_labels, 
            logits_test = logits_test,
            labels_test = labels_test, 
            num_classes = num_classes,
            model_save_directory = model_save_directory, 
            num_hidden_layers=size)
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
    # arguments:  labels_file, logits_file, extracted_features_file, testing_results_directory, model_save_directory
    if len(sys.argv) != 6:
        print("Usage: python Main_Model_Accuracy_Testing.py <labels_file> <logits_file> <extracted_features_file> <testing_results_directory> <model_save_directory>") 
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"Error: The file {sys.argv[1]} does not exist.")
        sys.exit(1)
    if not os.path.isfile(sys.argv[1]):
        print(f"Error: The path {sys.argv[1]} is not a file.")
        sys.exit(1)
    if not os.path.exists(sys.argv[2]):
        print(f"Error: The file {sys.argv[2]} does not exist.")
        sys.exit(1)
    if not os.path.isfile(sys.argv[2]):
        print(f"Error: The path {sys.argv[2]} is not a file.")
        sys.exit(1)
    if not os.path.exists(sys.argv[3]):
        print(f"Error: The file {sys.argv[3]} does not exist.")
        sys.exit(1)
    if not os.path.isfile(sys.argv[3]):
        print(f"Error: The path {sys.argv[3]} is not a file.")
        sys.exit(1)
    if not os.path.exists(sys.argv[4]):
        print(f"Error: The directory {sys.argv[4]} does not exist.")
        sys.exit(1)
    if not os.path.isdir(sys.argv[4]):
        print(f"Error: The path {sys.argv[4]} is not a directory.")
        sys.exit(1)
    if not os.path.exists(sys.argv[5]):
        print(f"Error: The directory {sys.argv[5]} does not exist.")
        sys.exit(1)
    if not os.path.isdir(sys.argv[5]):
        print(f"Error: The path {sys.argv[5]} is not a directory.")
        sys.exit(1)
    

    
    #load the data 
    labels_file = sys.argv[1]
    logits_file = sys.argv[2]
    extracted_features_file = sys.argv[3]
    testing_results_directory = sys.argv[4]
    model_save_directory = sys.argv[5]

    #load the logits, where the first 800 are the training logits and the last 200 are the testing logits
    logits = pd.read_csv(logits_file)
    #convert to numpy array
    logits = logits.to_numpy()
    logits_train = logits[:800]
    logits_test = logits[800:]
    #load the extracted features, but in this case we only need the first 800 for training
    extracted_features = np.load(extracted_features_file)
    extracted_features_train = extracted_features[:800]
    #load the labels
    labels = pd.read_csv(labels_file)
    labels_train = labels[:800]
    labels_test = labels[800:]

    #load the class labels
    class_labels = pd.read_csv(labels_file)
    class_labels = class_labels['Class Label'].unique()
    num_classes = len(class_labels)

    #run the tests
    testing_num_hidden_layers(logits_train, extracted_features_train, class_labels, labels_train, logits_test, labels_test, num_classes, model_save_directory, testing_results_directory)


if __name__ == "__main__":
    main()

