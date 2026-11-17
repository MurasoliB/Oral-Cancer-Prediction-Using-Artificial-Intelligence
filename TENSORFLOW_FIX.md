# TensorFlow / model loading fix

The bundled `model/densenet201_model.keras` was successfully loaded with Python 3.11 and TensorFlow 2.13.1. TensorFlow 2.15.0 previously failed while restoring the model weights (`Layer 'conv1/conv' expected 1 variables, but received 0`).

Use the pinned `requirements.txt` and the clean `venv_fixed` environment. Do not install a second TensorFlow version into an existing environment. `setup_windows.bat` installs dependencies and runs the model-loading diagnostic.

The app retains the original classes (`Normal`, `Oral Cancer`, `Pre-Cancer`), preprocessing (RGB, 224x224, float32 divided by 255), uncertainty rule, and bundled model.
