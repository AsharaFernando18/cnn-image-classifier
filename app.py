#!/usr/bin/env python3
"""
CIFAR-10 CNN Super User Interface
Modern Web Application with Flask
Production-ready with error handling and security

Copyright (c) 2026 Ashara Fernando
All rights reserved.

This project is provided as-is for educational and research purposes.
"""

import os
import io
import base64
import warnings
import logging
from datetime import datetime

# Suppress all warnings for clean output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import numpy as np
import tensorflow as tf

# Fix OpenCV/OpenGL issues in headless environments
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':99'
os.environ['MPLBACKEND'] = 'Agg'

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, send_file
import json

# Set matplotlib backend and suppress warnings
plt.switch_backend('Agg')
warnings.filterwarnings('ignore')

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FLASK APPLICATION SETUP
# ============================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['JSON_SORT_KEYS'] = False

# Add security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# ============================================
# CIFAR-10 WEB PREDICTOR CLASS
# ============================================

class CIFAR10WebPredictor:
    def __init__(self):
        """Initialize the web predictor"""
        self.model = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_names_display = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        self.class_names_sinhala = ['ගුවන් යානය', 'මෝටර් රථය', 'කුරුල්ලා', 'බළලා', 'මුවා',
                                   'බල්ලා', 'ගෙම්බා', 'අශ්වයා', 'නෞකාව', 'ට්‍රක් රථය']
        self.class_emojis = ['✈️', '🚗', '🐦', '🐱', '🦌', '🐕', '🐸', '🐴', '🚢', '🚛']
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model('cifar10_cnn_model.h5')
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_data):
        """Preprocess image for prediction"""
        try:
            # Convert PIL image to numpy array
            img = np.array(image_data)
            
            # Resize to CIFAR-10 format using PIL
            img_pil = Image.fromarray(img)
            img_resized_pil = img_pil.resize((32, 32))
            img_resized = np.array(img_resized_pil)
            
            # Normalize
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch, img_resized
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return None, None
    
    def ensure_json_serializable(self, obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: self.ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj
    
    def predict_image(self, image_data):
        """Make prediction on image"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess
            img_batch, img_display = self.preprocess_image(image_data)
            if img_batch is None:
                return {"error": "Failed to process image"}
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx]) * 100
            
            # Get all predictions
            all_predictions = []
            for i, (class_name, class_display, class_si, emoji) in enumerate(zip(
                self.class_names, self.class_names_display, self.class_names_sinhala, self.class_emojis)):
                all_predictions.append({
                    'class': str(class_display),  # Use display name (capitalized English)
                    'class_key': str(class_name),  # Keep original for internal use
                    'class_sinhala': str(class_si),
                    'emoji': str(emoji),
                    'confidence': float(predictions[0][i]) * 100,
                    'is_predicted': bool(i == predicted_class_idx)
                })
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Create visualization
            chart_data = self.create_prediction_chart(predictions[0])
            
            result = {
                'success': True,
                'predicted_class': str(self.class_names_display[predicted_class_idx]),
                'predicted_class_key': str(self.class_names[predicted_class_idx]),
                'predicted_class_sinhala': str(self.class_names_sinhala[predicted_class_idx]),
                'predicted_emoji': str(self.class_emojis[predicted_class_idx]),
                'confidence': float(confidence),
                'all_predictions': all_predictions,
                'chart_data': chart_data,
                'timestamp': str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }
            
            # Ensure everything is JSON serializable
            return self.ensure_json_serializable(result)
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def create_prediction_chart(self, predictions):
        """Create chart data for visualization"""
        chart_data = []
        for i, (class_display, confidence) in enumerate(zip(self.class_names_display, predictions)):
            chart_data.append({
                'class': str(class_display),  # Use display names for chart
                'confidence': float(confidence) * 100,
                'color': f'hsl({i * 36}, 70%, 60%)'
            })
        return chart_data

# Initialize predictor
predictor = CIFAR10WebPredictor()

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to load interface'}), 500

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({'success': False, 'error': 'File is too large (max 16 MB)'}), 413

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    try:
        # Check request data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        image_data = data['image']
        if not image_data or not isinstance(image_data, str):
            return jsonify({'success': False, 'error': 'Invalid image format'})
        
        # Handle base64 image data
        try:
            # Extract base64 data
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            import base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to parse image: {str(e)}'})
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        result = predictor.predict_image(image)
        
        return jsonify(predictor.ensure_json_serializable(result))
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/model-info')
def model_info():
    """Get model information"""
    try:
        if predictor.model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        # Get model summary
        total_params = int(predictor.model.count_params())
        trainable_params = int(sum([tf.keras.backend.count_params(w) for w in predictor.model.trainable_weights]))
        
        layers_info = []
        for i, layer in enumerate(predictor.model.layers):
            layer_info = {
                'index': int(i + 1),
                'name': str(layer.__class__.__name__),
                'output_shape': str(layer.output_shape)
            }
            
            if hasattr(layer, 'filters'):
                layer_info['filters'] = int(layer.filters)
            if hasattr(layer, 'units'):
                layer_info['units'] = int(layer.units)
                
            layers_info.append(layer_info)
        
        result = {
            'success': True,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_layers': len(predictor.model.layers),
            'layers': layers_info,
            'classes': len(predictor.class_names),
            'class_names': [str(name) for name in predictor.class_names_display],  # Use display names
            'class_names_internal': [str(name) for name in predictor.class_names],
            'class_names_sinhala': [str(name) for name in predictor.class_names_sinhala]
        }
        
        return jsonify(predictor.ensure_json_serializable(result))
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to get model info: {str(e)}'})

@app.route('/sample-images')
def sample_images():
    """Get sample CIFAR-10 images"""
    try:
        # Load a few sample images from CIFAR-10 dataset
        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), _ = cifar10.load_data()
        
        samples = []
        classes_shown = set()
        
        for i in range(len(x_train)):
            class_idx = int(y_train[i][0])
            if class_idx not in classes_shown and len(samples) < 10:
                # Convert image to base64
                img = x_train[i]
                img_pil = Image.fromarray(img)
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                samples.append({
                    'image': f'data:image/png;base64,{img_base64}',
                    'class': str(predictor.class_names_display[class_idx]),  # Use display names
                    'class_key': str(predictor.class_names[class_idx]),
                    'class_sinhala': str(predictor.class_names_sinhala[class_idx]),
                    'emoji': str(predictor.class_emojis[class_idx])
                })
                classes_shown.add(class_idx)
        
        result = {
            'success': True,
            'samples': samples
        }
        
        return jsonify(predictor.ensure_json_serializable(result))
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to load samples: {str(e)}'})

if __name__ == '__main__':
    # Suppress Flask development server warnings
    import logging as werkzeug_logging
    werkzeug_log = werkzeug_logging.getLogger('werkzeug')
    werkzeug_log.setLevel(werkzeug_logging.ERROR)
    
    print("\n" + "="*70)
    print("🚀 CIFAR-10 CNN SUPER UI - INITIALIZING")
    print("="*70)
    print("✅ Model loaded successfully!")
    print("🌐 Web interface available at: http://localhost:5000")
    print("✨ Features:")
    print("   • Drag & Drop image upload")
    print("   • Real-time AI classification")
    print("   • Detailed prediction confidence scores")
    print("   • Model architecture visualization")
    print("📱 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    logger.info("Starting Flask application on http://0.0.0.0:5000")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
