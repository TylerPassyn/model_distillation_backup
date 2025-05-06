Traceback (most recent call last):
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 166, in <module>
    main()
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 162, in main
    testing_num_hidden_layers(logits_train, extracted_features_train, class_labels, labels_train, logits_test, labels_test, num_classes, model_save_directory, testing_results_directory)
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 60, in testing_num_hidden_layers
    model, accuracy = train_and_test_model(
                      ^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 24, in train_and_test_model
    distiller = Distill_VIT(
                ^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Image_Distiller.py", line 18, in __init__
    self.model = self.init_model()
                 ^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Image_Distiller.py", line 39, in init_model
    id2label={i: name for i, name in enumerate(self.class_names)},
                                               ^^^^^^^^^^^^^^^^
AttributeError: 'Distill_Model_VIT_to_VIT' object has no attribute 'class_names'
