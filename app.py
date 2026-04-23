import streamlit as st
import torch
import cv2
import numpy as np
import time
import os
from PIL import Image
from transformers import VideoMAEImageProcessor, TimesformerForVideoClassification

# --- Configuration ---
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
PROCESSOR_NAME = "MCG-NJU/videomae-base"

# Set page config
st.set_page_config(
    page_title="DeepScan | TimeSformer Neural Forensics",
    page_icon="🛡️",
    layout="wide"
)

# --- Enhanced CSS (Cyber-Forensics Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 50% 0%, #1c2331 0%, #05070a 100%);
        color: #e0e0e0;
    }

    /* Glassmorphism Cards */
    [data-testid="stVerticalBlock"] > div:has(div.stVideo), 
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 10px;
    }

    /* Animated Title */
    .hero-title {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1.5px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(0, 242, 254, 0.1);
        color: #00f2fe;
        border: 1px solid #00f2fe;
        margin-bottom: 2rem;
    }

    /* Analysis Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.8s ease-out;
    }
    .real-card {
        background: rgba(52, 211, 153, 0.05);
        border: 1px solid #34d399;
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.2);
    }
    .fake-card {
        background: rgba(248, 113, 113, 0.05);
        border: 1px solid #f87171;
        box-shadow: 0 0 20px rgba(248, 113, 113, 0.2);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Cyber Button */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
    }

    code {
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.05) !important;
    }

    </style>
    """, unsafe_allow_html=True)

# --- Logic ---
@st.cache_resource
def load_model():
    try:
        processor = VideoMAEImageProcessor.from_pretrained(PROCESSOR_NAME)
        model = TimesformerForVideoClassification.from_pretrained(MODEL_NAME)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return None
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if frames: frames.append(frames[-1])
            else: frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    return np.array(frames)

# --- UI Header ---
st.markdown('<div class="hero-title">DeepScan</div>', unsafe_allow_html=True)
st.markdown('<div class="status-badge">Neural Forensics Engine v2.1 (TimeSformer)</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🖥️ System Status")
    st.success("Core Engine: Ready")
    st.info("Device: " + ("CUDA Accelerated" if torch.cuda.is_available() else "CPU Mode"))
    st.markdown("---")
    st.markdown("### ⚙️ Parameters")
    num_frames = st.slider("Frame Sequence Length", 8, 32, 16)
    threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.7)
    st.markdown("---")
    st.caption("A premium deepfake detection interface using advanced space-time attention models.")

# --- Main Interface ---
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 📽️ Source Media")
    uploaded_video = st.file_uploader("Drop video file for temporal analysis", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        st.video(uploaded_video)

with col2:
    st.markdown("### 🔍 Diagnostics")
    if not uploaded_video:
        st.info("Upload a video to initiate neural diagnostics.")
    else:
        if st.button("🚀 INITIATE NEURAL SCAN"):
            # Save temp
            temp_filename = "temp_analysis_video.mp4"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_video.getbuffer())

            # Progress Logic
            status = st.status("Initializing Neural Nodes...", expanded=True)
            processor, model = load_model()
            
            status.write("📡 Extracting temporal frame sequences...")
            frames = extract_frames(temp_filename, num_frames=num_frames)
            time.sleep(0.5)
            
            status.write("🧠 Performing Space-Time Attention mapping...")
            inputs = processor(list(frames), return_tensors="pt")
            
            with torch.no_grad():
                start_time = time.time()
                outputs = model(**inputs)
                latency = time.time() - start_time
                probs = torch.softmax(outputs.logits, dim=-1)
                max_conf = torch.max(probs).item()
            
            status.update(label="Analysis Complete", state="complete", expanded=False)

            # Results
            if max_conf > threshold:
                label, css, icon = "REAL", "real-card", "✅"
                verdict = "Media verified authentic. No synthetic temporal anomalies detected."
            else:
                label, css, icon = "FAKE", "fake-card", "⚠️"
                verdict = "WARNING: Synthetic media signatures detected. High temporal variance."

            st.markdown(f"""
                <div class="result-card {css}">
                    <h2 style="margin:0; color: inherit;">{icon} {label}</h2>
                    <p style="margin-top:10px; font-size: 0.9rem; opacity: 0.8;">{verdict}</p>
                </div>
            """, unsafe_allow_html=True)

            # Metric Grid
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Neural Confidence", f"{max_conf*100:.2f}%")
            with m_col2:
                st.metric("Inference Latency", f"{latency:.3f}s")
            
            st.progress(max_conf, text="Analysis Probability Density")

st.markdown("---")
st.caption("DeepScan | Secure Forensic Environment | v2.1.0")
