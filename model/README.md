# DenseNet201 Model

`densenet201_model.keras` is the trained model bundled with the application.

## Input
- RGB image
- Resized to 224 × 224
- Float32 pixel values scaled to [0, 1]

## Output classes
1. Normal
2. Oral Cancer
3. Pre-Cancer

The application class order is tied to the training dataset folders:
`normal`, `oral_cancer`, `pre_cancer`.

Do not describe the confidence score as a probability of disease or as disease severity.
