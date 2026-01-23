# Paid Version Deepfake Detection System Documentation

## 1. Problem Statement
With the rise of generative AI, deepfakes have become increasingly sophisticated, posing threats to privacy and security. Traditional CNN-based detection methods often fail against modern generation techniques that leave subtle artifacts. This "Paid Version" system addresses these limitations by leveraging **Transformer** architectures to model both spatial and temporal inconsistencies.

## 2. System Architecture
The system follows a modern full-stack architecture:

- **Frontend**: A "Premium" UI built with HTML5, CSS3 (Glassmorphism), and Vanilla JavaScript. It handles:
    - Drag-and-drop for Images and Videos.
    - Real-time preview.
    - Dynamic result visualization with confidence bars.

- **Backend**: A scalable Flask API serving as the inference engine.
    - `/predict-image`: Handles single-image analysis.
    - `/predict-video`: Handles video frame extraction and sequence analysis.
    - Model Loading: Loads heavy Transformer models once at startup.

## 3. Core AI Models (The "Paid" Value)
Unlike free/basic versions that use simple CNNs (ResNet/EfficientNet), this system employs **Transformers**:

### 3.1 Vision Transformer (ViT) - For Images
- **Why?** CNNs look at local features (edges, textures). Deepfakes often have *global* structural inconsistencies (e.g., mismatched symmetry). ViTs process images as patches and use Self-Attention to model long-range dependencies, capturing these global flaws.
- **Implementation**: We use `timm`'s `vit_base_patch16_224`, pre-trained on ImageNet and fine-tuned for deepfake binary classification.

### 3.2 Temporal Transformer - For Videos
- **Why?** A video is not just a pile of images. A deepfake might look real in individual frames but jitter or flicker over time.
- **Implementation**:
    1. **Frame Sampling**: We extract 1-3 FPS from the video to balance speed and accuracy.
    2. **Feature Extraction**: Each frame is passed through the ViT (without the head) to get a 768-dimensional embedding.
    3. **Temporal Attention**: These embeddings are fed into a generic `TransformerEncoder`. The model looks at the *sequence* of frames to detect temporal anomalies.
    4. **Decision**: The system outputs a probability score based on the entire sequence.

## 4. Dataset & Training Strategy
- **Data**: Combined datasets from Kaggle (DeepFake Detection Challenge) and self-collected real-world samples to ensure robustness.
- **Augmentation**: Applied random crops, brightness adjustments, and compression artifacts during training to mimic real-world conditions.

## 5. Usage
1. **Start Backend**: `python backend/app.py`
2. **Open Frontend**: Open `frontend/index.html` in a browser.
3. **Upload**: Select an Image (`.jpg`, `.png`) or Video (`.mp4`, `.avi`).
4. **View Results**: The system displays "REAL" or "FAKE" with a confidence percentage and explanation (e.g., "Analysis based on 10 sampled frames").

## 6. Limitations
- **Compute**: Processing videos requires significant CPU/GPU resources.
- **Face Detection**: Reliance on MTCNN means if a face is not detected (e.g., extreme angle), analysis stops.
