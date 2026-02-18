// --- Global Selectors ---
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const loadingState = document.getElementById('loadingState');
const resultCard = document.getElementById('resultCard');
const resetBtn = document.getElementById('resetBtn');

// --- Webcam Selectors ---
const startWebcamBtn = document.getElementById('startWebcamBtn');
const stopWebcamBtn = document.getElementById('stopWebcamBtn');
const webcamContainer = document.getElementById('webcamContainer');
const webcamView = document.getElementById('webcamView');
const captureCanvas = document.getElementById('captureCanvas');
const detectionContainer = document.querySelector('.detection-container');

// --- API Endpoints ---
const API_URL_IMAGE = '/predict-image';
const API_URL_VIDEO = '/predict-video';
const API_URL_LIVE = '/predict-live';

// --- State ---
let webcamStream = null;
let analysisInterval = null;

// --- Webcam Logic ---
startWebcamBtn.addEventListener('click', startWebcam);
stopWebcamBtn.addEventListener('click', stopWebcam);

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        });
        webcamView.srcObject = webcamStream;

        // UI Transitions
        uploadArea.style.display = 'none';
        detectionContainer.classList.add('live-mode');
        webcamContainer.style.display = 'block';
        resultCard.style.display = 'block';

        // Reset/Update Preview UI for live mode
        const previewContainer = document.querySelector('.image-preview-container');
        if (previewContainer) previewContainer.innerHTML = '';

        document.querySelector('.disclaimer span').textContent = "Live analysis active. Facial features are analyzed in real-time.";
        document.getElementById('framesGrid').style.display = 'none';

        // Start analysis loop (every 1.5s for stability)
        analysisInterval = setInterval(captureAndPredict, 1500);

    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please ensure permissions are granted.");
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (analysisInterval) {
        clearInterval(analysisInterval);
        analysisInterval = null;
    }

    webcamContainer.style.display = 'none';
    detectionContainer.classList.remove('live-mode');
    resultCard.style.display = 'none';
    uploadArea.style.display = 'block';

    // Reset badge and bar
    const badge = document.getElementById('predictionBadge');
    badge.textContent = 'PENDING';
    badge.className = 'badge';
    document.getElementById('confidenceValue').textContent = '0%';
    document.getElementById('confidenceBar').style.width = '0%';
}

async function captureAndPredict() {
    if (!webcamStream || !webcamView.videoWidth) return;

    const context = captureCanvas.getContext('2d');
    captureCanvas.width = webcamView.videoWidth;
    captureCanvas.height = webcamView.videoHeight;
    context.drawImage(webcamView, 0, 0, captureCanvas.width, captureCanvas.height);

    const imageData = captureCanvas.toDataURL('image/jpeg', 0.8);

    try {
        const response = await fetch(API_URL_LIVE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        if (response.ok) {
            const data = await response.json();
            updateLiveUI(data);
        }
    } catch (error) {
        console.error("Live analysis error:", error);
    }
}

function updateLiveUI(data) {
    const { prediction, confidence } = data;
    const statusEl = document.getElementById('liveStatus');

    if (prediction === "NO FACE DETECTED") {
        if (statusEl) statusEl.innerHTML = '<span class="pulse-dot" style="background-color: #ff9800; animation: none;"></span> SEARCHING FOR FACE...';
        return;
    }

    if (statusEl) statusEl.innerHTML = '<span class="pulse-dot"></span> LIVE ANALYSIS ACTIVE';

    const isFake = prediction.toLowerCase() === 'fake';
    const percentage = Math.round(confidence * 100);

    const badge = document.getElementById('predictionBadge');
    badge.textContent = isFake ? 'FAKE' : 'REAL';
    badge.className = isFake ? 'badge fake' : 'badge real';

    document.getElementById('confidenceValue').textContent = `${percentage}%`;
    const bar = document.getElementById('confidenceBar');
    bar.style.backgroundColor = isFake ? 'var(--error-color)' : 'var(--success-color)';
    bar.style.width = `${percentage}%`;
}

// --- File Upload Logic ---
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

uploadArea.addEventListener('click', (e) => {
    // Prevent trigger if clicking the webcam button inside the area
    if (e.target.closest('#startWebcamBtn')) return;
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

resetBtn.addEventListener('click', () => {
    resultCard.style.display = 'none';
    uploadArea.style.display = 'block';
    fileInput.value = '';
    const previewContainer = document.querySelector('.image-preview-container');
    if (previewContainer) previewContainer.innerHTML = '';
});

async function handleFile(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');

    if (!isImage && !isVideo) {
        alert("Please upload a valid image or video file.");
        return;
    }

    uploadArea.style.display = 'none';
    loadingState.style.display = 'block';

    const previewContainer = document.querySelector('.image-preview-container');
    previewContainer.innerHTML = '';

    if (isImage) {
        const img = document.createElement('img');
        img.id = 'imagePreview';
        img.alt = 'Preview';
        const reader = new FileReader();
        reader.onload = (e) => { img.src = e.target.result; };
        reader.readAsDataURL(file);
        previewContainer.appendChild(img);
    } else {
        const video = document.createElement('video');
        video.id = 'videoPreview';
        video.controls = true;
        const url = URL.createObjectURL(file);
        video.src = url;
        previewContainer.appendChild(video);
    }

    const endpoint = isImage ? API_URL_IMAGE : API_URL_VIDEO;
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errBody = await response.json();
            throw new Error(errBody.error || `API Error: ${response.statusText}`);
        }

        const data = await response.json();
        showResult(data);

    } catch (error) {
        console.error(error);
        alert("Error analyzing file: " + error.message);
        loadingState.style.display = 'none';
        uploadArea.style.display = 'block';
    }
}

function showResult(data) {
    loadingState.style.display = 'none';
    resultCard.style.display = 'block';

    const { prediction, confidence, frames_analyzed, type } = data;
    const isFake = prediction.toLowerCase() === 'fake';
    const percentage = Math.round(confidence * 100);

    const badge = document.getElementById('predictionBadge');
    badge.textContent = isFake ? 'FAKE' : 'REAL';
    badge.className = isFake ? 'badge fake' : 'badge real';

    document.getElementById('confidenceValue').textContent = `${percentage}%`;
    const bar = document.getElementById('confidenceBar');
    bar.style.backgroundColor = isFake ? 'var(--error-color)' : 'var(--success-color)';
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = `${percentage}%`; }, 100);

    const disclaimer = document.querySelector('.disclaimer span');
    const framesGrid = document.getElementById('framesGrid');
    const framesContainer = document.getElementById('framesContainer');

    if (type === 'video') {
        disclaimer.textContent = `Analysis based on ${frames_analyzed} sampled frames using Temporal Transformer.`;
        if (data.sampled_frames && data.sampled_frames.length > 0) {
            framesGrid.style.display = 'block';
            framesContainer.innerHTML = '';
            data.sampled_frames.forEach((frameBase64, index) => {
                const img = document.createElement('img');
                img.src = `data:image/jpeg;base64,${frameBase64}`;
                img.alt = `Frame ${index + 1}`;
                img.className = 'sampled-frame';
                framesContainer.appendChild(img);
            });
        }
    } else {
        disclaimer.textContent = 'Analysis based on global spatial inconsistencies using Vision Transformer.';
        framesGrid.style.display = 'none';
    }
}

// --- Initialization ---
window.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('mode') === 'live') {
        // Delay slightly to ensure browser is ready for media request
        setTimeout(startWebcam, 500);
    }
});
