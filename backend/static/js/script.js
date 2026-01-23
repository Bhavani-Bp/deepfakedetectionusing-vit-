const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const loadingState = document.getElementById('loadingState');
const resultCard = document.getElementById('resultCard');
const resetBtn = document.getElementById('resetBtn');

// API Endpoints - Updated to be relative since we are serving from the same origin now
const API_URL_IMAGE = '/predict-image';
const API_URL_VIDEO = '/predict-video';

// Drag & Drop Handling
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

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File Input Handling
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Reset Handling
resetBtn.addEventListener('click', () => {
    resultCard.style.display = 'none';
    uploadArea.style.display = 'block';
    fileInput.value = ''; // clear input
    // Reset preview
    const previewContainer = document.querySelector('.image-preview-container');
    if (previewContainer) {
        previewContainer.innerHTML = '';
    }
});

async function handleFile(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');

    if (!isImage && !isVideo) {
        alert("Please upload a valid image or video file.");
        return;
    }

    // Show Loading
    uploadArea.style.display = 'none';
    loadingState.style.display = 'block';

    // Show Preview
    const previewContainer = document.querySelector('.image-preview-container');
    previewContainer.innerHTML = ''; // Clear previous

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
        video.style.width = '100%';
        video.style.height = '100%';
        video.style.objectFit = 'contain';
        const url = URL.createObjectURL(file);
        video.src = url;
        previewContainer.appendChild(video);
    }

    // Determine Endpoint
    const endpoint = isImage ? API_URL_IMAGE : API_URL_VIDEO;

    // Prepare Request
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
        // Reset UI
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

    // Update Badge
    const badge = document.getElementById('predictionBadge');
    badge.textContent = isFake ? 'FAKE' : 'REAL';
    badge.className = isFake ? 'badge fake' : 'badge real';

    // Update Confidence Stats
    document.getElementById('confidenceValue').textContent = `${percentage}%`;
    const bar = document.getElementById('confidenceBar');

    // Set color based on result
    bar.style.backgroundColor = isFake ? 'var(--error-color)' : 'var(--success-color)';

    // Animate bar width
    bar.style.width = '0%';
    setTimeout(() => {
        bar.style.width = `${percentage}%`;
    }, 100);

    // Update Details (Video Frames or Image Disclaimer)
    const disclaimer = document.querySelector('.disclaimer span');
    const framesGrid = document.getElementById('framesGrid');
    const framesContainer = document.getElementById('framesContainer');

    if (type === 'video') {
        disclaimer.textContent = `Analysis based on ${frames_analyzed} sampled frames using Temporal Transformer.`;

        // Show Sampled Frames if available
        if (data.sampled_frames && data.sampled_frames.length > 0) {
            framesGrid.style.display = 'block';
            framesContainer.innerHTML = ''; // Clear old

            data.sampled_frames.forEach((frameBase64, index) => {
                const img = document.createElement('img');
                img.src = `data:image/jpeg;base64,${frameBase64}`;
                img.alt = `Frame ${index + 1}`;
                img.title = `Frame ${index + 1}`;
                img.className = 'sampled-frame';
                framesContainer.appendChild(img);
            });
        }
    } else {
        disclaimer.textContent = 'Analysis based on global spatial inconsistencies using Vision Transformer.';
        framesGrid.style.display = 'none';
    }
}
