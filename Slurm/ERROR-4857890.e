Traceback (most recent call last):
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 158, in <module>
    main()
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 154, in main
    testing_num_hidden_layers(logits_train, extracted_features_train, class_labels, labels_train, logits_test, labels_test, num_classes, model_save_directory, testing_results_directory)
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Distilling_Models.py", line 60, in testing_num_hidden_layers
    model, accuracy = train_and_test_model(logits_train, extracted_features, class_labels, labels_train, logits_test, labels_test, num_classes, model_save_directory, num_hidden_layers=size)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: train_and_test_model() got multiple values for argument 'num_hidden_layers'
