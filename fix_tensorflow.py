#!/usr/bin/env python3
"""
Script to help fix TensorFlow/Keras compatibility issues
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✅ Python version is compatible with TensorFlow")
        return True
    else:
        print("❌ Python version may not be compatible with TensorFlow")
        print("Recommended: Python 3.8-3.11")
        return False

def uninstall_conflicting_packages():
    """Uninstall potentially conflicting packages"""
    print("\n🧹 Cleaning up potentially conflicting packages...")
    
    packages_to_remove = [
        "tensorflow",
        "tensorflow-intel", 
        "tensorflow-cpu",
        "tensorflow-gpu",
        "keras",
        "tf-keras"
    ]
    
    for package in packages_to_remove:
        print(f"Uninstalling {package}...")
        success, stdout, stderr = run_command(f"pip uninstall -y {package}")
        if success:
            print(f"✅ {package} uninstalled")
        else:
            print(f"ℹ️  {package} was not installed or already removed")

def install_tensorflow():
    """Install TensorFlow with proper version"""
    print("\n📦 Installing TensorFlow...")
    
    # Try installing TensorFlow 2.13.0 (stable version)
    success, stdout, stderr = run_command("pip install tensorflow==2.13.0")
    
    if success:
        print("✅ TensorFlow 2.13.0 installed successfully")
        return True
    else:
        print("❌ Failed to install TensorFlow 2.13.0")
        print("Trying latest stable version...")
        
        # Try latest version
        success, stdout, stderr = run_command("pip install tensorflow")
        if success:
            print("✅ TensorFlow (latest) installed successfully")
            return True
        else:
            print("❌ Failed to install TensorFlow")
            print("Error:", stderr)
            return False

def test_tensorflow():
    """Test TensorFlow installation"""
    print("\n🧪 Testing TensorFlow installation...")
    
    test_code = '''
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Test Keras import
try:
    from tensorflow import keras
    print("✅ tf.keras imported successfully")
except ImportError as e:
    print("❌ tf.keras import failed:", e)
    try:
        import keras
        print("✅ standalone keras imported successfully")
    except ImportError as e2:
        print("❌ standalone keras import failed:", e2)

# Test model creation
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    print("✅ Model creation successful")
except Exception as e:
    print("❌ Model creation failed:", e)
'''
    
    success, stdout, stderr = run_command(f'python -c "{test_code}"')
    
    if success:
        print("✅ TensorFlow test passed!")
        print(stdout)
        return True
    else:
        print("❌ TensorFlow test failed!")
        print("Error:", stderr)
        return False

def main():
    """Main function to fix TensorFlow issues"""
    print("🔧 TensorFlow/Keras Compatibility Fixer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n⚠️  Consider upgrading to Python 3.8-3.11 for best compatibility")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Uninstall conflicting packages
    uninstall_conflicting_packages()
    
    # Install TensorFlow
    if not install_tensorflow():
        print("\n❌ Failed to install TensorFlow. Please check your internet connection and try again.")
        return
    
    # Test installation
    if test_tensorflow():
        print("\n🎉 TensorFlow is now working correctly!")
        print("\nNext steps:")
        print("1. Restart your Flask application")
        print("2. The app should now load your DenseNet201 model")
        print("3. Remove the mock system warnings")
    else:
        print("\n❌ TensorFlow installation has issues. You may need to:")
        print("1. Check your Python environment")
        print("2. Try installing in a virtual environment")
        print("3. Check for system-specific installation guides")

if __name__ == "__main__":
    main()