/**
 * CIFAR-10 CNN Classifier - Modern UI JavaScript
 * Professional image classification interface
 * 
 * Copyright (c) 2026 Ashara Fernando
 * All rights reserved.
 * 
 * This project is provided as-is for educational and research purposes.
 */

document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    loadModelInfo();
});

// ============================================
// STATE MANAGEMENT
// ============================================

const state = {
    isLoading: false,
    currentImageData: null,
    modelInfo: null,
    lastPrediction: null
};

// ============================================
// DOM ELEMENTS
// ============================================

const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    previewContainer: document.getElementById('previewContainer'),
    previewImg: document.getElementById('previewImg'),
    loading: document.getElementById('loading'),
    errorMessage: document.getElementById('errorMessage'),
    predictionSection: document.getElementById('predictionSection'),
    predictionClass: document.getElementById('predictionClass'),
    predictionEmoji: document.getElementById('predictionEmoji'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceFill: document.getElementById('confidenceFill'),
    allPredictions: document.getElementById('allPredictions'),
    buttonGroup: document.getElementById('buttonGroup'),
    modelStats: document.getElementById('modelStats')
};

// ============================================
// INITIALIZATION
// ============================================

function initializeUI() {
    // Drag and drop
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());

    elements.uploadArea.addEventListener('dragover', () => {
        elements.uploadArea.classList.add('dragover');
    });

    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });

    elements.uploadArea.addEventListener('drop', (e) => {
        elements.uploadArea.classList.remove('dragover');
        handleDrop(e.dataTransfer.files);
    });

    // Click to upload
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Keyboard shortcut for upload
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            elements.fileInput.click();
        }
    });
}

// ============================================
// FILE HANDLING
// ============================================

function handleDrop(files) {
    if (files.length > 0) {
        const file = files[0];
        if (isValidImageFile(file)) {
            handleFile(file);
        } else {
            showError('Please drop a valid image file (JPEG, PNG, GIF, WebP)');
        }
    }
}

function handleFile(file) {
    if (!isValidImageFile(file)) {
        showError('Invalid file type. Please upload an image.');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('File is too large. Maximum size is 16 MB.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        state.currentImageData = e.target.result;
        displayPreview(e.target.result);
        predictImage(e.target.result);
    };
    reader.onerror = () => {
        showError('Error reading file. Please try again.');
    };
    reader.readAsDataURL(file);
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    return validTypes.includes(file.type);
}

// ============================================
// PREVIEW
// ============================================

function displayPreview(dataUrl) {
    elements.previewImg.src = dataUrl;
    elements.previewContainer.classList.add('show');
    clearResults();
}

// ============================================
// PREDICTION
// ============================================

function predictImage(dataUrl) {
    if (state.isLoading) return;

    state.isLoading = true;
    elements.loading.classList.add('show');
    elements.errorMessage.classList.remove('show');

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: dataUrl }),
        timeout: 30000
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            elements.loading.classList.remove('show');
            state.isLoading = false;

            if (data.success === false) {
                showError(data.error || 'Prediction failed');
            } else if (data.error) {
                showError(data.error);
            } else {
                displayPredictionResults(data);
            }
        })
        .catch(error => {
            elements.loading.classList.remove('show');
            state.isLoading = false;
            console.error('Prediction error:', error);
            showError(`Error: ${error.message || 'Failed to process image'}`);
        });
}

function displayPredictionResults(data) {
    state.lastPrediction = data;

    // Main prediction
    const confidence = Math.round(data.confidence);
    elements.predictionClass.textContent = data.predicted_class;
    elements.predictionEmoji.textContent = data.predicted_emoji;
    elements.confidenceValue.textContent = confidence;

    // Animate confidence bar
    setTimeout(() => {
        elements.confidenceFill.style.width = confidence + '%';
    }, 100);

    // Display all predictions
    displayAllPredictions(data.all_predictions);

    // Show results
    elements.predictionSection.classList.add('show');
    elements.errorMessage.classList.remove('show');
}

function displayAllPredictions(predictions) {
    if (!predictions || predictions.length === 0) return;

    elements.allPredictions.innerHTML = '';

    predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = `prediction-item ${index === 0 ? 'top-1' : ''}`;

        const confidence = Math.round(pred.confidence);

        item.innerHTML = `
            <div class="prediction-name">
                <span class="prediction-emoji">${pred.emoji}</span>
                <span>${pred.class}</span>
            </div>
            <div class="prediction-confidence-bar">
                <div class="prediction-confidence-bar-fill" style="width: ${confidence}%"></div>
            </div>
            <div class="prediction-percentage">${confidence}%</div>
        `;

        elements.allPredictions.appendChild(item);

        // Animate bar
        setTimeout(() => {
            const fill = item.querySelector('.prediction-confidence-bar-fill');
            if (fill) {
                fill.style.width = confidence + '%';
            }
        }, 100 + index * 50);
    });
}

// ============================================
// MODEL INFO
// ============================================

function loadModelInfo() {
    fetch('/model-info')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                state.modelInfo = data;
                displayModelStats(data);
            }
        })
        .catch(error => {
            console.warn('Could not load model info:', error);
        });
}

function displayModelStats(data) {
    if (!elements.modelStats) return;

    const statsHTML = `
        <div class="stat-card">
            <div class="stat-icon">🧠</div>
            <div class="stat-label">Total Parameters</div>
            <div class="stat-value">${formatNumber(data.total_parameters)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-label">Trainable Params</div>
            <div class="stat-value">${formatNumber(data.trainable_parameters)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🔢</div>
            <div class="stat-label">Layers</div>
            <div class="stat-value">${data.total_layers}</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-label">Classes</div>
            <div class="stat-value">${data.classes}</div>
        </div>
    `;

    elements.modelStats.innerHTML = statsHTML;
}

// ============================================
// ERROR HANDLING
// ============================================

function showError(message) {
    elements.errorMessage.textContent = `❌ ${message}`;
    elements.errorMessage.classList.add('show');
    console.error('UI Error:', message);
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function clearImage() {
    elements.previewContainer.classList.remove('show');
    elements.fileInput.value = '';
    elements.predictionSection.classList.remove('show');
    elements.errorMessage.classList.remove('show');
    elements.allPredictions.innerHTML = '';
    elements.uploadArea.classList.remove('dragover');
    state.currentImageData = null;
}

function uploadAnother() {
    clearImage();
    elements.fileInput.click();
}

function clearResults() {
    elements.predictionSection.classList.remove('show');
    elements.errorMessage.classList.remove('show');
    elements.allPredictions.innerHTML = '';
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// ============================================
// KEYBOARD SHORTCUTS
// ============================================

document.addEventListener('keydown', (e) => {
    // Clear on Escape
    if (e.key === 'Escape' && state.currentImageData) {
        clearImage();
    }

    // New upload on N
    if (e.key === 'n' && state.currentImageData) {
        uploadAnother();
    }
});

// ============================================
// ACCESSIBILITY ENHANCEMENTS
// ============================================

// Add ARIA labels for screen readers
elements.uploadArea.setAttribute('role', 'button');
elements.uploadArea.setAttribute('aria-label', 'Click or drag image to upload');
elements.uploadArea.setAttribute('tabindex', '0');

elements.uploadArea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        elements.fileInput.click();
    }
});

// ============================================
// PROGRESSIVE ENHANCEMENT
// ============================================

// Check for fetch support
if (!window.fetch) {
    showError('Your browser does not support modern features required for this application.');
}

// Check for localStorage
const hasLocalStorage = (() => {
    try {
        const test = '__localstorage_test__';
        localStorage.setItem(test, test);
        localStorage.removeItem(test);
        return true;
    } catch (e) {
        return false;
    }
})();

// ============================================
// CONSOLE INFO
// ============================================

console.log('%cCIFAR-10 CNN Classifier', 'color: #667eea; font-size: 16px; font-weight: bold;');
console.log('%cPowered by TensorFlow & Flask', 'color: #764ba2; font-style: italic;');
