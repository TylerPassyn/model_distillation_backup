''''
@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''

#https://huggingface.co/google/vit-large-patch16-224 


from PIL import Image
import numpy as np
import pandas as pd
import requests
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
import time
import os
import sys 
from tqdm import tqdm


#load the model
def test_main_model(image_directory, labels_file, results_file_directory):
    print("Loading model.....")
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224")
    labels = pd.read_csv(labels_file)
    #convert the csv file to a list
    labels = labels.values.tolist()
    last_200_labels = labels[-200:]

    model_size = model.num_parameters()
    model_size_mb = model_size / (1024 * 1024)
    num_attention_heads = model.config.num_attention_heads
    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    image_file_names = []
    num_files_traversed = 0
    index = 0
    total_correct = 0
    total_time = 0

    for file in os.listdir(image_directory):
          if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".JPEG"):
               num_files_traversed += 1
               if num_files_traversed <= 800:
                    continue
               else: 
                    image_path = os.path.join(image_directory, file)
                    image_file_names.append(file)
                    image = Image.open(image_path)
                    image = image.convert("RGB")
                    start_time = time.time()
                    inputs = feature_extractor(images=image, return_tensors="pt")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    predicted_label = model.config.id2label[predicted_class_idx]
                    end_time = time.time()
                    inference_time = end_time - start_time
                    total_time += inference_time
                    actual_label = last_200_labels[index]
                    #make sure that atcual label is not in [] 
                    index += 1
                    # check if any of the predicted labels are in the actual label
                    if any(substring in predicted_label for substring in actual_label):
                         print(f"Correctly classified {file} as {predicted_label}, actual label is {actual_label}")
                         total_correct += 1
                    else:
                         print(f"Incorrectly classified {file} as {predicted_label}, actual label is {actual_label}")
     
     #save the results
    accuracy = total_correct/len(last_200_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Results saved to {results_file_directory}/model_accuracy.txt")
    with open(results_file_directory + '/model_accuracy.txt', 'w') as f:
          f.write(f"Model size (in parameters): {model_size}\n")
          f.write(f"Model size (in MB): {model_size_mb}\n")
          f.write(f"Number of attention heads: {num_attention_heads}\n")
          f.write(f"Number of hidden layers: {num_hidden_layers}\n")
          f.write(f"Hidden size: {hidden_size}\n")
          f.write(f"Total inference time for all images: {total_time} seconds\n")
          f.write(f"Average inference time per image: {total_time / len(last_200_labels)} seconds\n")
          f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
          f.write(f"Results were tested on {len(last_200_labels)} images\n")
          f.write(f"The files traversed were the last 200 files in the directory\n")
          f.write(f"The names are as follows:\n")
          for name in image_file_names:
               f.write(f"{name}\n")

def main():
     # image_directory, labels_file, results_file_directory
      if len(sys.argv) != 4:
           print("Usage: python Main_Model_Accuracy_Testing.py <image_directory> <labels_file> <results_file_directory>")
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
      test_main_model(image_directory, labels_file, results_file_directory)
if __name__ == "__main__":
      main()
          
          
          
     
     
            

            
    


