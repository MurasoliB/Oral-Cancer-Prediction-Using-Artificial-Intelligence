#!/usr/bin/env python3
"""
DenseNet201 Model Training Script for Lip/Tongue Cancer Detection
This script will train a new model using your dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime

# Configuration
CONFIG = {
    'IMAGE_SIZE': (224, 224),
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.0001,
    'VALIDATION_SPLIT': 0.2,
    'TEST_SPLIT': 0.1,
    'NUM_CLASSES': 3,
    'CLASS_NAMES': ['Normal', 'Oral Cancer', 'Pre-Cancer'],
    'MODEL_NAME': 'densenet201_model.keras',
    'DATASET_PATH': 'dataset',  # Change this to your dataset path
    'SAVE_HISTORY': True,
    'EARLY_STOPPING_PATIENCE': 10,
    'REDUCE_LR_PATIENCE': 5
}

def setup_gpu():
    """Setup GPU if available"""
    print("🔧 Setting up GPU...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
    else:
        print("ℹ️  No GPU found, using CPU")

def create_dataset_structure():
    """Create proper dataset structure if it doesn't exist"""
    print("📁 Checking dataset structure...")
    
    dataset_path = CONFIG['DATASET_PATH']
    class_names = CONFIG['CLASS_NAMES']
    
    if not os.path.exists(dataset_path):
        print(f"Creating dataset directory: {dataset_path}")
        os.makedirs(dataset_path)
        
        for class_name in class_names:
            class_dir = os.path.join(dataset_path, class_name.lower())
            os.makedirs(class_dir, exist_ok=True)
            print(f"Created: {class_dir}")
        
        print("\n📋 Dataset Structure Created:")
        print(f"{dataset_path}/")
        for class_name in class_names:
            print(f"  ├── {class_name.lower()}/  (put {class_name.lower()} images here)")
        
        print("\n⚠️  Please organize your images into the above folders and run this script again.")
        return False
    
    return True

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("📊 Loading dataset...")
    
    dataset_path = CONFIG['DATASET_PATH']
    image_size = CONFIG['IMAGE_SIZE']
    
    # Check if dataset exists and has images
    total_images = 0
    class_counts = {}
    
    for class_name in CONFIG['CLASS_NAMES']:
        class_dir = os.path.join(dataset_path, class_name.lower())
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            class_counts[class_name] = len(images)
            total_images += len(images)
        else:
            class_counts[class_name] = 0
    
    print(f"📈 Dataset Statistics:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    print(f"  Total: {total_images} images")
    
    if total_images == 0:
        print("❌ No images found in dataset!")
        return None, None, None, None
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=CONFIG['VALIDATION_SPLIT']
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation data
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Class indices: {train_generator.class_indices}")
    
    return train_generator, validation_generator, train_generator.class_indices, class_counts

def create_densenet201_model():
    """Create DenseNet201 model for medical image classification"""
    print("🏗️  Building DenseNet201 model...")
    
    # Load pre-trained DenseNet201
    base_model = DenseNet201(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['IMAGE_SIZE'], 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(CONFIG['NUM_CLASSES'], activation='softmax', name='predictions')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("✅ Model created successfully!")
    print(f"📊 Total parameters: {model.count_params():,}")
    print(f"📊 Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model

def setup_callbacks():
    """Setup training callbacks"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['EARLY_STOPPING_PATIENCE'],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=CONFIG['REDUCE_LR_PATIENCE'],
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"model/best_{CONFIG['MODEL_NAME']}",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def train_model(model, train_gen, val_gen):
    """Train the model"""
    print("🚀 Starting training...")
    
    callbacks = setup_callbacks()
    
    # Calculate steps
    steps_per_epoch = train_gen.samples // CONFIG['BATCH_SIZE']
    validation_steps = val_gen.samples // CONFIG['BATCH_SIZE']
    
    print(f"📊 Steps per epoch: {steps_per_epoch}")
    print(f"📊 Validation steps: {validation_steps}")
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG['EPOCHS'],
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, train_gen, val_gen, history):
    """Fine-tune the model by unfreezing some layers"""
    print("🔧 Fine-tuning model...")
    
    # Unfreeze the top layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 50
    
    # Freeze all the layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"📊 Trainable parameters after fine-tuning: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    # Continue training
    fine_tune_epochs = 20
    total_epochs = len(history.history['loss']) + fine_tune_epochs
    
    callbacks = setup_callbacks()
    
    history_fine = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // CONFIG['BATCH_SIZE'],
        epochs=total_epochs,
        initial_epoch=len(history.history['loss']),
        validation_data=val_gen,
        validation_steps=val_gen.samples // CONFIG['BATCH_SIZE'],
        callbacks=callbacks,
        verbose=1
    )
    
    return history_fine

def save_model_and_history(model, history, class_indices):
    """Save the trained model and training history"""
    print("💾 Saving model...")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save model
    model_path = f"model/{CONFIG['MODEL_NAME']}"
    model.save(model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save training history
    if CONFIG['SAVE_HISTORY']:
        history_path = f"model/training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        history_dict = {
            'history': {k: [float(v) for v in values] for k, values in history.history.items()},
            'config': CONFIG,
            'class_indices': class_indices,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"✅ History saved: {history_path}")

def plot_training_history(history):
    """Plot training history"""
    print("📊 Creating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plot_path = f"model/training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved: {plot_path}")
    plt.show()

def test_trained_model():
    """Test the newly trained model"""
    print("🧪 Testing trained model...")
    
    try:
        model_path = f"model/{CONFIG['MODEL_NAME']}"
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test with a dummy image
        dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(dummy_image, verbose=0)
        
        print(f"✅ Test prediction successful!")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction probabilities: {prediction[0]}")
        
        class_names = CONFIG['CLASS_NAMES']
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        print(f"Predicted class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main training function"""
    print("🏥 DenseNet201 Medical Image Classifier Training")
    print("=" * 60)
    
    # Setup
    setup_gpu()
    
    # Check dataset structure
    if not create_dataset_structure():
        return
    
    # Load data
    train_gen, val_gen, class_indices, class_counts = load_and_preprocess_data()
    if train_gen is None:
        return
    
    # Create model
    model = create_densenet201_model()
    
    # Train model
    print(f"\n🎯 Starting training for {CONFIG['EPOCHS']} epochs...")
    history = train_model(model, train_gen, val_gen)
    
    # Fine-tune (optional)
    print("\n🔧 Starting fine-tuning...")
    history_fine = fine_tune_model(model, train_gen, val_gen, history)
    
    # Save everything
    save_model_and_history(model, history_fine, class_indices)
    
    # Plot results
    plot_training_history(history_fine)
    
    # Test model
    if test_trained_model():
        print("\n🎉 Training completed successfully!")
        print(f"✅ Your new DenseNet201 model is ready: model/{CONFIG['MODEL_NAME']}")
        print("✅ You can now restart your Flask app to use the real model!")
    else:
        print("\n❌ Training completed but model testing failed")
    
    print("\n📋 Next Steps:")
    print("1. Restart your Flask application")
    print("2. The app will automatically load your new trained model")
    print("3. Test with real medical images")
    print("4. Monitor performance and retrain if needed")

if __name__ == "__main__":
    main()