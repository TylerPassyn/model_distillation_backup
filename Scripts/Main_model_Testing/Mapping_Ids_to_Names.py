import numpy as np
import pandas as pd
import os
import sys


#get the first file that has the id and the label name 
#/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/class_labels.csv
class_labels = pd.read_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/class_labels.csv")

#get the second file which is just the true labels, but who lack the actual id numbers
#/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/extracted_labels.csv
extracted_labels = pd.read_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/extracted_labels.csv")

ids = []
for row in extracted_labels.iterrows():
    # Get the label name from the extracted labels
    label_name = row[1]['Class Label']
    
    #find the substring that maches that label in the class_labels
    matching_row = class_labels[class_labels['Class Label'].str.contains(label_name, na=False)]
    # Get the id number from the matching row
    id = matching_row['Class ID'].values[0]
    #replace the label name with the full label name from class_labels
    #but make sure to convert the whole label name to a string, so it does nto get confused in the commas
    extracted_labels.at[row[0], 'Class Label'] = str(matching_row['Class Label'].values[0])
    # Append the id number to the list
    ids.append(id)

# Add the id numbers to the extracted labels
extracted_labels['ID'] = ids
# Save the new dataframe with the id numbers
extracted_labels.to_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/extracted_labels_with_ids.csv", index=False)
