#!/bin/bash
# no SBATCH here—run this from a login node

save_model_directory="/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Distilled_Models/Systematic_Testing"
results_save_directory="/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Testing_Results"

hidden_layers=(2 8 18)
epochs=(10 25 40)
temperatures=(2 4 6)

for hl in "${hidden_layers[@]}"; do
  # Submit jobs for each hidden layer
  #ecgi 
  sbatch A_job_template.sh "$save_model_directory" "$results_save_directory" 1 "$hl"
  #echo "Submitted job for hidden layer $hl"
  echo "Submitted job for hidden layer $hl"

done

for ep in "${epochs[@]}"; do
  sbatch A_job_template.sh "$save_model_directory" "$results_save_directory" 2 "$ep"
  echo "Submitted job for epcohs $ep"
done

for t in "${temperatures[@]}"; do
  sbatch A_job_template.sh "$save_model_directory" "$results_save_directory" 3 "$t"
  echo "Submitted job for temperature $t"
done
