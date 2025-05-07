Epoch 1/2:   0%|          | 0/32 [00:00<?, ?batch/s]Epoch 1/2:   0%|          | 0/32 [00:00<?, ?batch/s]
Traceback (most recent call last):
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Better_Distillation.py", line 266, in <module>
    main()
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Better_Distillation.py", line 219, in main
    student = distiller.distill(feats, tlogs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/passta23/model_distillation_backup/Scripts/Model_Distillation/Better_Distillation.py", line 91, in distill
    out   = self.student(inputs_embeds=feat_batch)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/deac/csc/classes/csc373/software/csc373/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ViTForImageClassification.forward() got an unexpected keyword argument 'inputs_embeds'
