import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
from PIL import Image


class DistillModelVIT:
    def __init__(
        self,
        num_classes: int,
        num_hidden_layers: int = 8,
        hidden_size: int = 768,
        attention_heads: int = 12,
        temperature: float = 4.0,
        learning_rate: float = 1e-4,
        num_epochs: int = 5,
        batch_size: int = 32,
        device: torch.device = None,
        image_processor_name: str = "google/vit-base-patch16-224-in21k",
    ):
        # Hyperparameters
        self.num_classes       = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size       = hidden_size
        self.attention_heads   = attention_heads
        self.temperature       = temperature
        self.learning_rate     = learning_rate
        self.num_epochs        = num_epochs
        self.batch_size        = batch_size
        self.device            = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image processor for prediction
        self.processor = ViTImageProcessor.from_pretrained(image_processor_name)

        # Build the student model
        cfg = ViTConfig(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.attention_heads,
            num_labels=self.num_classes,
        )
        self.student = ViTForImageClassification(cfg).to(self.device)

    def _kl_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        t_probs = nn.functional.softmax(teacher_logits / T, dim=-1)
        s_logp  = nn.functional.log_softmax(student_logits / T, dim=-1)
        return nn.functional.kl_div(s_logp, t_probs, reduction="batchmean") * (T * T)

    def distill(
        self,
        features: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> ViTForImageClassification:
        """
        Pure-KL distillation loop.

        Args:
          features:        Tensor of shape [N, seq_len, hidden_size]
                           containing your pre-extracted patch embeddings.
          teacher_logits:  Tensor of shape [N, num_classes] with teacher outputs.
        Returns:
          The distilled student model.
        """
        # 1) DataLoader
        ds     = TensorDataset(features, teacher_logits)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # 2) Optimizer
        optimizer = optim.Adam(self.student.parameters(), lr=self.learning_rate)

        # 3) Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.student.train()
            running_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.num_epochs}", unit="batch")

            for i, (feat_batch, tlogits_batch) in enumerate(pbar):
                feat_batch     = feat_batch.to(self.device)      # [B, seq_len, H]
                tlogits_batch  = tlogits_batch.to(self.device)   # [B, C]

                # Forward pass on pre-extracted embeddings
                out   = self.student(feat_batch)  # [B, C, H, W]
                slogs  = out.logits                            

                # Pure KL loss
                loss = self._kl_loss(tlogits_batch, slogs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(avg_loss=running_loss / (i + 1))

            avg = running_loss / len(loader)
            print(f"→ Epoch {epoch} done — Avg KL Loss: {avg:.4f}")

        print("✅ Distillation finished.")
        return self.student

    def save_model(self, path: str):
        """Save the student model to path."""
        self.student.save_pretrained(path)
        print(f"Student model saved to {path}")

    def load_model(self, path: str) -> ViTForImageClassification:
        """Load a pretrained student model from path."""
        self.student = ViTForImageClassification.from_pretrained(path).to(self.device)
        print(f"Student model loaded from {path}")
        return self.student

    def predict(self, image_path: str) -> int:
        """
        Run inference on a single image.

        Returns the predicted class index.
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        self.student.eval()
        with torch.no_grad():
            outs = self.student(**inputs)
        return int(outs.logits.argmax(dim=-1))

def main():
    import numpy as np
    import pandas as pd
    from ast import literal_eval 
    import os 
    import sys
    from sklearn.metrics import accuracy_score
    from ast import literal_eval 
    from transformers import ViTForImageClassification

    if len(sys.argv) != 5:
        print("Usage: python Better_Distillation.py <save_model_directory> <results_save_directory> <type_of_distillation_experiment> <disillation_experiment_number>")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f"Error: The directory {sys.argv[1]} does not exist.")
        sys.exit(1)
    if not os.path.isdir(sys.argv[1]):
        print(f"Error: The path {sys.argv[1]} is not a directory.")
        sys.exit(1)
    if not os.path.exists(sys.argv[2]):
        print(f"Error: The directory {sys.argv[2]} does not exist.")
        sys.exit(1)
    if not os.path.isdir(sys.argv[2]):
        print(f"Error: The path {sys.argv[2]} is not a directory.")
        sys.exit(1)
    

    save_directory_model = sys.argv[1]
    save_directory = sys.argv[2]
    type_of_distillation_experiment = sys.argv[3]
    distillation_experiment_num = sys.argv[4]

    #try to cast the distillation experiment number to an int
    try:
        distillation_experiment_num = int(distillation_experiment_num)
    except ValueError:
        print("Error: The distillation experiment number must be an integer.")
        sys.exit(1)

    if type_of_distillation_experiment not in ["1", "2", "3"]:
        print("Error: The type of distillation experiment must be 1, 2, or 3.")
        sys.exit(1)

    num_hidden_layers_fr=8
    temperature_fr=4.0
    num_epochs_fr=25
    experiment_type_str = None
    if type_of_distillation_experiment == "1":
        experiment_type_str = "Hidden_Layers"
        num_hidden_layers_fr= int(sys.argv[3])
    elif type_of_distillation_experiment == "2":
        experiment_type_str = "Epochs"
        num_epochs_fr = int(sys.argv[3])
    elif type_of_distillation_experiment == "3":
        experiment_type_str = "Temperature"
        temperature_fr = float(sys.argv[3])
        
        

    # 1) Load your pre-extracted data (outside the class!)
    feats_np = np.load("/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/all_features.npy")   # [N, seq_len, H]
    logs = pd.read_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/logits_output.csv")
    #get the logits column from the csv file
    
    logs['Logits'] = logs['Logits'].apply(literal_eval)  # Convert string to list
    logs_np= np.array(logs['Logits'].tolist())  # Convert to a NumPy array


    # 2) Convert to torch tensors
    feats  = torch.tensor(feats_np, dtype=torch.float32)
    tlogs  = torch.tensor(logs_np, dtype=torch.float32)

    # 3) Instantiate and distill
    distiller = DistillModelVIT(
        num_classes=logs_np.shape[1],
        num_hidden_layers= num_hidden_layers_fr,
        hidden_size=768,
        attention_heads=12,
        temperature= temperature_fr,
        learning_rate=1e-4,
        num_epochs= num_epochs_fr,
        batch_size=32,
    )
    student = distiller.distill(feats, tlogs)

    # 4) Save the distilled student
    distiller.save_model(save_directory_model + "/distilled_student" + "_" + experiment_type_str + "_" + str(sys.argv[3]) + "_epochs_" + str(num_epochs_fr) + "_hidden_layers_" + str(num_hidden_layers_fr) + "_temperature_" + str(temperature_fr) )
    print(f"Distilled model saved to {save_directory_model}/distilled_student" + "_" + experiment_type_str + "_" + str(sys.argv[3]) + "_epochs_" + str(num_epochs_fr) + "_hidden_layers_" + str(num_hidden_layers_fr) + "_temperature_" + str(temperature_fr))

    # 5) test the accuracy of the distilled model
    labels_test = pd.read_csv("/deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/extracted_labels_with_ids.csv")
    #get the id column from the labels_test
    labels_test = labels_test[['ID']] # Select only the 'ID' column
    # Convert the 'ID' column to a NumPy array
    labels_test = np.array(labels_test)  # Convert to a NumPy array

    # Convert labels_test to PyTorch tensor
    labels_test = torch.tensor(labels_test, dtype=torch.long)

    #load distilled model
    model  = student
    model.eval()

    with torch.no_grad():
            outputs = model(feats)  # Forward pass through the model the features
            predicted = torch.argmax(outputs.logits, dim=1)  # Get the predicted class indices
            accuracy = accuracy_score(labels_test, predicted)
            print(f"Accuracy: {accuracy * 100:.2f}%")

    #save the results to a csv file that has the following columns:
    # num_hidden_layers
    # hidden_size
    # temperature
    # num_epochs
    # accuracy
    # if the file already exists, append to it
    # if not, create it
    results_file = os.path.join(save_directory, "distillation_results.csv")
    if os.path.exists(results_file):
        # Append to the existing file
        with open(results_file, "a") as f:
            f.write(f"{num_hidden_layers_fr},{768},{temperature_fr},{num_epochs_fr},{accuracy}\n")
    else:
        # Create a new file and write the header
        with open(results_file, "w") as f:
            f.write("num_hidden_layers,hidden_size,temperature,num_epochs,accuracy\n")
            f.write(f"{num_hidden_layers_fr},{768},{temperature_fr},{num_epochs_fr},{accuracy}\n")
    print(f"Results saved to {results_file}")

    
if __name__ == "__main__":
    main()