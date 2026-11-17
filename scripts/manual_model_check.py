#!/usr/bin/env python3
"""
Test script to check model loading and make a simple prediction
"""

import os
import numpy as np
from PIL import Image

def test_model_loading():
    """Test if the model can be loaded"""
    print("🧪 Testing Model Loading")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ Keras available: {hasattr(keras, 'models')}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check if model file exists
    model_path = 'model/densenet201_model.keras'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✅ Model file found: {file_size:.1f} MB")
    
    # Try to load the model
    try:
        print("Loading model...")
        model = keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_prediction(model):
    """Test making a prediction with the model"""
    print("\n🔮 Testing Prediction")
    print("=" * 40)
    
    if not model:
        print("❌ No model available for testing")
        return
    
    # Create a dummy image
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    print("✅ Created dummy image (224x224x3)")
    
    try:
        prediction = model.predict(dummy_image, verbose=0)
        print("✅ Prediction successful!")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction values: {prediction[0]}")
        
        # Interpret prediction
        class_names = ["Normal", "Oral Cancer", "Pre-Cancer"]
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        print(f"Predicted class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_with_real_image():
    """Test with a real uploaded image if available"""
    print("\n📸 Testing with Real Images")
    print("=" * 40)
    
    upload_dir = 'instance/uploads'
    if not os.path.exists(upload_dir):
        print("❌ Upload directory not found")
        return
    
    image_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ No test images found in uploads directory")
        return
    
    print(f"✅ Found {len(image_files)} test images")
    
    # Test with first image
    test_image = image_files[0]
    print(f"Testing with: {test_image}")
    
    try:
        # Load and preprocess image
        img_path = os.path.join(upload_dir, test_image)
        img = Image.open(img_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"✅ Image preprocessed: {img_array.shape}")
        return img_array
        
    except Exception as e:
        print(f"❌ Image preprocessing failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 DenseNet201 Model Test Suite")
    print("=" * 50)
    
    # Test model loading
    model = test_model_loading()
    if model is False:
        raise SystemExit(1)

    # This is a technical smoke test only, not a meaningful medical prediction.
    if not test_prediction(model):
        raise SystemExit(1)

    # Test with a real uploaded image if one exists; absence is not a setup failure.
    real_image = test_with_real_image()
    if real_image is not None:
        print("\n🎯 Testing with Real Image")
        print("=" * 40)
        try:
            prediction = model.predict(real_image, verbose=0)
            class_names = ["Normal", "Oral Cancer", "Pre-Cancer"]
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class] * 100
            print(f"Technical inference completed. Output class index: {predicted_class}")
            print(f"Output score: {confidence:.2f}% (not a clinical probability)")
            print(f"Raw output scores: {prediction[0]}")
        except Exception as e:
            print(f"❌ Real image inference failed: {e}")
            raise SystemExit(1)

    print("\n" + "=" * 50)
    print("Model loading and inference smoke tests passed.")