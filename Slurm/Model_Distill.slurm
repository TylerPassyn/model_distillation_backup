#!/bin/bash
#SBATCH --job-name="csc373-DM"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=248GB
#SBATCH --time=01-00:00:00
#SBATCH --output="OUTPUT-%j.o"
#SBATCH --error="ERROR-%j.e"
#SBATCH --mail-user=passta23@wfu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

# Load the required module
module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8

# Define the parameter values
hidden_layers=(2 8 18)
epochs=(10 25 40)
temperatures=(2 4 6)


# Directories
save_model_directory="/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Systematic_Testing"
results_save_directory="/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Testing_Results"

# Submit one job for each hidden_layer
for hidden_layer in "${hidden_layers[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="DM_h${hidden_layer}"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=248GB
#SBATCH --time=01-00:00:00
#SBATCH --output="OUTPUT-h${hidden_layer}-%j.o"
#SBATCH --error="ERROR-h${hidden_layer}-%j.e"
#SBATCH --mail-user=passta23@wfu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8
cd /deac/csc/classes/csc373/${USER}/model_distillation_backup/Scripts/Model_Distillation

# Run the Python script for hidden_layer
python Better_Distillation.py ${save_model_directory} ${results_save_directory} 1 ${hidden_layer}
EOF
done

# Submit one job for each epoch
for epoch in "${epochs[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="DM_e${epoch}"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=248GB
#SBATCH --time=01-00:00:00
#SBATCH --output="OUTPUT-e${epoch}-%j.o"
#SBATCH --error="ERROR-e${epoch}-%j.e"
#SBATCH --mail-user=passta23@wfu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8
cd /deac/csc/classes/csc373/${USER}/model_distillation_backup/Scripts/Model_Distillation

# Run the Python script for epoch
python Better_Distillation.py ${save_model_directory} ${results_save_directory} 2 ${epoch}
EOF
done

# Submit one job for each temperature
for temperature in "${temperatures[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="DM_t${temperature}"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=248GB
#SBATCH --time=01-00:00:00
#SBATCH --output="OUTPUT-t${temperature}-%j.o"
#SBATCH --error="ERROR-t${temperature}-%j.e"
#SBATCH --mail-user=passta23@wfu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8
cd /deac/csc/classes/csc373/${USER}/model_distillation_backup/Scripts/Model_Distillation

# Run the Python script for temperature
python Better_Distillation.py ${save_model_directory} ${results_save_directory} 3 ${temperature}
EOF
done