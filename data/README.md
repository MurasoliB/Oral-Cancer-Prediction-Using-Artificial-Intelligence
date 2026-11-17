# Dataset

The training dataset is intentionally not committed to this repository. It may be redistributed separately depending on its source license.

Expected structure:

```text
train/
  normal/
  pre_cancer/
  oral_cancer/
validation/
  normal/
  pre_cancer/
  oral_cancer/
test/
  normal/
  pre_cancer/
  oral_cancer/
```

The bundled application uses the trained model in `model/densenet201_model.keras` for inference. Use the training scripts only with a dataset you are legally permitted to use.
