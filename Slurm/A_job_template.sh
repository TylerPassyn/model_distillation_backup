#!/bin/bash
#SBATCH --job-name=DM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128GB
#SBATCH --time=01-00:00:00
#SBATCH --output="OUTPUT-%j.o"
#SBATCH --error="ERROR-%j.e"
#SBATCH --mail-user=passta23@wfu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8
cd /deac/csc/classes/csc373/${USER}/model_distillation_backup/Scripts/Model_Distillation

# args: $1 = save_model_dir, $2 = results_dir, $3 = mode, $4 = value
python Better_Distillation.py "$1" "$2" "$3" "$4"
