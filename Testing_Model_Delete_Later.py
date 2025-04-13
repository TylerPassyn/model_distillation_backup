from transformers import ViTForImageClassification
import pandas as pd
# Load the model
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')

# Get the id2label mapping
id2label = model.config.id2label

# Print the list of classes
print("Classes the model can classify:")
for class_id, class_name in id2label.items():
    print(f"Class ID {class_id}: {class_name}")

#/deac/csc/classes/csc373/passta23/model_distillation_backup/distilled_model
model_2 = ViTForImageClassification.from_pretrained('/deac/csc/classes/csc373/passta23/model_distillation_backup/distilled_model')
# Get the id2label mapping
id2label_2 = model_2.config.id2label
# Print the list of classes
print("Classes the distilled model can classify:")
for class_id, class_name in id2label_2.items():
    print(f"Class ID {class_id}: {class_name}")

#set the labels to new labels from /deac/csc/classes/csc373/passta23/model_distillation_backup/Results/Main_Model_Outputs/class_labels.csv
class_labels = pd.read_csv('/deac/csc/classes/csc373/passta23/model_distillation_backup/Results/Main_Model_Outputs/class_labels.csv')
# Create a mapping from old labels to new labels
label_mapping = {int(row['Class ID']): row['Class Label'] for index, row in class_labels.iterrows()}
# Update the id2label mapping in the distilled model
for class_id, class_name in id2label_2.items():
    if int(class_id) in label_mapping:
        id2label_2[class_id] = label_mapping[int(class_id)]
    else:
        print(f"Class ID {class_id} not found in the new labels.")
# Print the updated list of classes
print("Updated classes the distilled model can classify:")
for class_id, class_name in id2label_2.items():
    print(f"Class ID {class_id}: {class_name}")
# Save the updated model with the new labels
model_2.save_pretrained('/deac/csc/classes/csc373/passta23/model_distillation_backup/distilled_model_updated')


    