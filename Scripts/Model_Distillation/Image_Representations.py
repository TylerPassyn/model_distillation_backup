import os
from PIL import Image
import torch
from transformers import ViTModel, ViTFeatureExtractor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Configuration
IMAGE_DIR = "/mnt/data/images"  # Path to your images folder
paths = []
weevil_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02177972_weevil.JPEG"
fly_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02190166_fly.JPEG"
bee_path ="/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02206856_bee.JPEG"
ant_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02219486_ant.JPEG"
grasshopper_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02226429_grasshopper.JPEG"
cricket_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02229544_cricket.JPEG"
walking_stick_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02231487_walking_stick.JPEG"
cockroach_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02233338_cockroach.JPEG"
mantis_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02236044_mantis.JPEG"
cicada_path = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/n02256656_cicada.JPEG"
MODEL_NAME = "google/vit-large-patch16-224"  # Replace with your distilled ViT model path/name
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_MODEL_PATH = "/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Systematic_Testing/distilled_student_Temperature_3_epochs_25_hidden_layers_8_temperature_4.0"
# Add paths to the list
paths.append(weevil_path)
paths.append(fly_path)
paths.append(bee_path)
paths.append(ant_path)
paths.append(grasshopper_path)
paths.append(cricket_path)
paths.append(walking_stick_path)
paths.append(cockroach_path)
paths.append(mantis_path)
paths.append(cicada_path)

#create a dictionary of the paths
paths_dict = {
    "weevil": weevil_path,
    "fly": fly_path,
    "bee": bee_path,
    "ant": ant_path,
    "grasshopper": grasshopper_path,
    "cricket": cricket_path,
    "walking_stick": walking_stick_path,
    "cockroach": cockroach_path,
    "mantis": mantis_path,
    "cicada": cicada_path
}
# Load model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Load images and extract representations
embeddings = []
labels = []
for filename in paths:
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(filename).convert("RGB")
        inputs = feature_extractor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token representation
            cls_embedding = outputs.pooler_output.squeeze().cpu().numpy()
            embeddings.append(cls_embedding)  # Or derive a label/category from filename

embeddings = np.vstack(embeddings)

# Dimensionality reduction (choose one)
# PCA to 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Alternatively, use t-SNE
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
# embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
# Color by derived label/category; here we just use indices
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='tab10', s=100)
for name in paths_dict.keys():
    txt = name
    #now plot the name of the image
    plt.annotate(txt, (embeddings_2d[paths.index(paths_dict[name]), 0], embeddings_2d[paths.index(paths_dict[name]), 1]), fontsize=12, )
plt.title("2D Visualization of ViT Representations")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig("representation_plot.png")
plt.show()


plt.plot([0,1,2,3], [10,20,10,5])
plt.title("Test Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("test_plot.png")
plt.show()