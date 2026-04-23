import os
import sys
import logging
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import DeepfakeViT, TemporalTransformer, DeepfakeEfficientNet
from src.face_extraction import FaceExtractor
from transformers import VideoMAEImageProcessor, TimesformerForVideoClassification

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app) # Enable CORS for all routes

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Running on device: {DEVICE}")

# --- Load Models ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. Try Loading ViT (Preferred if weights exist)
    model_path_vit = os.path.join(base_dir, '..', 'models', 'deepfake_vit_best.pth')
    model_path_eff = os.path.join(base_dir, '..', 'models', 'deepfake_efficientnet.pth')
    
    if os.path.exists(model_path_vit):
        logger.info(f"Loading ViT weights from {model_path_vit}...")
        image_model = DeepfakeViT(num_classes=2).to(DEVICE)
        image_model.load_state_dict(torch.load(model_path_vit, map_location=DEVICE))
        input_dim = 768 # ViT-Base dimension
    elif os.path.exists(model_path_eff):
        logger.info(f"Loading EfficientNet weights from {model_path_eff}...")
        image_model = DeepfakeEfficientNet(num_classes=2).to(DEVICE)
        image_model.model.load_state_dict(torch.load(model_path_eff, map_location=DEVICE))
        input_dim = 1280 # EfficientNet-B0 dimension
    else:
        logger.warning("No trained weights found! Defaulting to ViT with random weights. Predictions will be inaccurate.")
        image_model = DeepfakeViT(num_classes=2).to(DEVICE)
        input_dim = 768

    image_model.eval()
    
    # 2. Initialize Video Model (TimeSformer)
    logger.info("Initializing TimeSformer for advanced video analysis...")
    video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    video_model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400").to(DEVICE)
    video_model.eval()
    
    # Face Extractor (Keeping it for image tasks)
    face_extractor = FaceExtractor(device=DEVICE)
    
    logger.info("Models loaded successfully. TimeSformer is active for video detection.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    sys.exit(1)

# --- Preprocessing ---
# Standard ImageNet normalization for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_single_image(pil_image):
    """
    Helper to run inference on a single PIL image.
    """
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]
    with torch.no_grad():
        logits = image_model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        
        return pred_idx.item(), conf.item()

def extract_features_for_video(frames):
    """
    Extract features for a list of PIL images using the Image Model (ViT).
    """
    features_list = []
    # Process in batches if necessary, but loop is fine for small num_frames
    for frame in frames:
        tensor = transform(frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Use the extract_features method we added
            feats = image_model.extract_features(tensor) 
            # ViT features might be [1, N_patches, Dim] or [1, Dim] depending on pooling.
            # timm forward_features usually returns unpooled: [B, N, C]
            # We want a global vector per frame? Or keep tokens?
            # TemporalTransformer expects [Batch, Seq_Len, Features].
            # Let's assume we pool the frame features into one vector per frame.
            
            # Simple Global Average Pooling across patches if unpooled
            if len(feats.shape) == 3:
                feats = feats.mean(dim=1) # [1, Dim]
            
            features_list.append(feats)
            
    if not features_list:
        return None
        
    return torch.cat(features_list, dim=0).unsqueeze(0) # [1, Seq_Len, Dim]

# --- Routes ---

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": DEVICE})

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/about')
def about():
    # Placeholder or redirect to home for now
    return render_template('landing.html') # Or a separate about page

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save temp
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # 1. Face Extraction
        face = face_extractor.process_image(filepath)
        if face is None:
            # Fallback: Use original image if no face detected (or error)
            # But "Paid Version" implies robustness. Let's try raw image if needed,
            # but ideally we tell user "No face found".
            # For robustness, we'll try to process the raw image (maybe it's a tight crop already)
            img_raw = Image.open(filepath).convert('RGB')
            face = img_raw
            # return jsonify({"error": "No face detected in image"}), 400

        # 2. Prediction
        pred_idx, confidence = predict_single_image(face)
        
        # Cleanup
        os.remove(filepath)
        
        # --- CORRECTED PREDICTION LOGIC ---
        # Based on dataset.py mapping:
        # Index 0 = REAL, Index 1 = FAKE
        
        if pred_idx == 1:
            label = "FAKE"
        else:
            label = "REAL"
        
        # Simple version for UI:
        display_label = label

        return jsonify({
            "prediction": label, # More descriptive for logs/advanced users
            "display_prediction": display_label, # Binary for simple UI
            "confidence": float(confidence),
            "type": "image",
            "raw_prediction": int(pred_idx)
        })

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/predict-video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # 1. Extract Frames for TimeSformer
        # We need a sequence of frames (e.g., 8 or 16)
        import cv2
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
             return jsonify({"error": "Invalid video file"}), 400
             
        # Extract 8 frames evenly
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        if len(frames) < 8:
            return jsonify({"error": "Video too short for analysis"}), 400

        # 2. TimeSformer Inference
        inputs = video_processor(list(frames), return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = video_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            max_conf = torch.max(probs).item()
        
        # 3. Logic based on threshold 0.7
        # Note: Kinetics-400 doesn't have a direct "Fake" label, 
        # so we use the user's logic: High confidence in a pattern -> Real, else Fake.
        if max_conf > 0.7:
            label = "REAL"
            confidence = max_conf
        else:
            label = "FAKE"
            confidence = 1.0 - max_conf # Or just max_conf depending on how you want to show it

        # --- PREPARE FRAMES FOR UI (Compatibility with original design) ---
        import base64
        from io import BytesIO
        from PIL import Image
        
        encoded_frames = []
        for frame_arr in frames:
            pil_img = Image.fromarray(frame_arr)
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            encoded_frames.append(img_str)

        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            "prediction": label,
            "confidence": float(confidence),
            "frames_analyzed": len(frames),
            "sampled_frames": encoded_frames,
            "type": "video",
            "model_used": "TimeSformer",
            "device": DEVICE
        })

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
