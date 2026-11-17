"""
Optimization script to reduce TensorFlow startup warnings and improve performance
"""
import os
import warnings

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings

# Suppress other common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set TensorFlow to use CPU efficiently
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage (faster startup)

print("🔧 TensorFlow optimizations applied")
print("📝 Startup warnings suppressed")
print("💻 Configured for CPU-only inference")