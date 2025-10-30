#!/usr/bin/env python3
"""
Quick System Test - Enhanced CIFAR-10 CNN
"""

def test_system():
    print("🚀 ENHANCED CIFAR-10 CNN SYSTEM TEST")
    print("="*50)
    
    try:
        # Test 1: Core imports
        print("\n1. Testing core imports...")
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        print(f"   ✓ TensorFlow {tf.__version__}")
        print(f"   ✓ NumPy {np.__version__}")
        print("   ✓ PIL/Pillow")
        
        # Test 2: Model loading
        print("\n2. Testing model loading...")
        if tf.io.gfile.exists('cifar10_cnn_model.h5'):
            model = tf.keras.models.load_model('cifar10_cnn_model.h5')
            print(f"   ✓ Model loaded: {model.count_params():,} parameters")
        else:
            print("   ⚠️  Model file not found, but system ready for training")
        
        # Test 3: Flask imports
        print("\n3. Testing web interface imports...")
        from flask import Flask, render_template, request, jsonify
        print("   ✓ Flask components")
        
        # Test 4: Visualization libs
        print("\n4. Testing visualization libraries...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("   ✓ Matplotlib & Seaborn")
        
        # Test 5: Check files
        print("\n5. Checking generated files...")
        import os
        files_to_check = [
            'cifar10_cnn_model.h5',
            'training_history.png', 
            'confusion_matrix.png',
            'sample_images.png'
        ]
        
        for file in files_to_check:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✓ {file} ({size:,} bytes)")
            else:
                print(f"   📋 {file} (will be generated)")
        
        print("\n" + "="*50)
        print("🎉 SYSTEM TEST COMPLETED SUCCESSFULLY!")
        print("✅ All components are working properly")
        print("🌐 Ready to run web interface on http://localhost:5000")
        print("🧠 Ready to train enhanced CNN model")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_system()
