#!/bin/bash
#SBATCH --job-name="csc373-DM"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16GB
#SBATCH --time=01-00:00:00
#SBATCH --output="OUTPUT-%j.o"
#SBATCH --error="ERROR-%j.e"
#SBATCH --mail-user=passta23@wfu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

# Load a software module
module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8

# Go into your work path
cd /deac/csc/classes/csc373/${USER}/model_distillation_backup/Scripts/Model_Distillation

# Run the program
# python ... ... ...
python Distilling_Models.py /deac/csc/classes/csc373/passta23/model_distillation_backup/Data/imagenet-sample-images-master/extracted_labels.csv /deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/logits_output.csv /deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/all_features.npy /deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Testing_Results /deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Systematic_Testing/Hidden_Layer_Models
#python code_snippets.py