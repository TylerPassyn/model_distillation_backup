Traceback (most recent call last):
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Better_Distillation.py", line 267, in <module>
    main()
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Better_Distillation.py", line 210, in main
    distiller = DistillModelVIT(
                ^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Better_Distillation.py", line 45, in __init__
    self.student = ViTForImageClassification(cfg).to(self.device)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/transformers/models/vit/modeling_vit.py", line 823, in __init__
    self.vit = ViTModel(config, add_pooling_layer=False)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/transformers/models/vit/modeling_vit.py", line 573, in __init__
    self.encoder = ViTEncoder(config)
                   ^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/transformers/models/vit/modeling_vit.py", line 440, in __init__
    self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'tuple' object cannot be interpreted as an integer
