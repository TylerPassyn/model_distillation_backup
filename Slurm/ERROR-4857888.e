Traceback (most recent call last):
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7081, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7089, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'class'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 157, in <module>
    if __name__ == "__main__":
        ^^^^^^
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 149, in main
    class_labels = pd.read_csv(labels_file)
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/pandas/core/frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 'class'
