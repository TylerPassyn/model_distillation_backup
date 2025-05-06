Traceback (most recent call last):
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/numpy/lib/npyio.py", line 465, in load
    return pickle.load(fid, **pickle_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to int

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 155, in <module>
    main()
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 134, in main
    logits = np.load(logits_file, allow_pickle=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/numpy/lib/npyio.py", line 467, in load
    raise pickle.UnpicklingError(
_pickle.UnpicklingError: Failed to interpret file '/deac/csc/classes/csc373/passta23/model_distillation_backup/Output/Main_Model_Outputs/logits_output.csv' as a pickle
