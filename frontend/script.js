document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const processingState = document.getElementById('processing-state');
    const resultsSection = document.getElementById('results-section');
    const statusText = document.getElementById('status-text');
    const resetBtn = document.getElementById('reset-btn');

    // Drag and Drop Effects
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('dragover');
        });
    });

    // Handle File Drop or Select
    uploadZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Restart process
    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        fileInput.value = '';
    });

    function handleFile(file) {
        if (!file) return;

        // Transition UI to processing state
        uploadZone.classList.add('hidden');
        processingState.classList.remove('hidden');

        // Initial Status
        statusText.innerText = "Extracting and Analyzing Audio...";

        // Simulate small delay for UI UX even if it's fast
        setTimeout(() => {
            uploadFileToAPI(file);
        }, 1000);
    }

    // Environment Config
    const API_BASE_URL = window.location.hostname === "localhost"
        ? "http://localhost:8000"
        : "https://audioguard2026mp-production.up.railway.app";

    // Cloudinary Config (Unsigned Upload)
    const CLOUDINARY_URL = "https://api.cloudinary.com/v1_1/dug4wbgmo/video/upload";
    const CLOUDINARY_PRESET = "ml_default";

    async function uploadFileToAPI(file) {
        try {
            // 1. Upload to Cloudinary
            statusText.innerText = "Uploading to Secure Cloud Storage...";
            const cloudFormData = new FormData();
            cloudFormData.append('file', file);
            cloudFormData.append('upload_preset', CLOUDINARY_PRESET);

            const cloudResponse = await fetch(CLOUDINARY_URL, {
                method: 'POST',
                body: cloudFormData
            });

            if (!cloudResponse.ok) {
                const errData = await cloudResponse.json();
                throw new Error(`Cloudinary Upload Failed: ${errData.error.message}`);
            }

            const cloudData = await cloudResponse.json();
            const videoUrl = cloudData.secure_url;
            console.log("Cloudinary URL:", videoUrl);

            // 2. Send URL to Railway Backend for Analysis
            statusText.innerText = "Running Deep Learning Models (TCA & SER)...";
            const backendFormData = new FormData();
            backendFormData.append('video_url', videoUrl);

            const response = await fetch(`${API_BASE_URL}/api/analyze`, {
                method: 'POST',
                body: backendFormData
            });

            if (!response.ok) {
                throw new Error("Failed to process file.");
            }

            const data = await response.json();

            // Artificial delay to mimic heavy processing for "wow" factor if it responds too fast locally
            setTimeout(() => {
                showResults(data);
            }, 1000);

        } catch (error) {
            console.error(error);
            alert("Error processing the video/audio. Check console.");
            processingState.classList.add('hidden');
            uploadZone.classList.remove('hidden');
        }
    }

    function showResults(data) {
        processingState.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Update DOM elements
        document.getElementById('transcription-text').innerText = `"${data.transcription}"`;
        document.getElementById('emotion-text').innerText = data.detected_emotion.charAt(0).toUpperCase() + data.detected_emotion.slice(1);
        document.getElementById('confidence-text').innerText = data.confidence;
        
        // Language & Translation Logic
        const langBadge = document.getElementById('lang-badge');
        langBadge.innerText = (data.original_language || 'EN').toUpperCase();
        
        const translationCard = document.getElementById('translation-card');
        if (data.original_language && data.original_language !== 'en' && data.translation_en) {
            translationCard.classList.remove('hidden');
            document.getElementById('translation-text').innerText = `"${data.translation_en}"`;
        } else {
            translationCard.classList.add('hidden');
        }

        const badge = document.getElementById('decision-badge');
        if (data.is_hatespeech) {
            badge.className = 'badge danger';
            badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Hate Speech Detected';
            // Also add a red tint to the glow background for extra effect
            document.querySelector('.glow-bg').style.background = 'radial-gradient(circle at 50% 50%, rgba(239, 68, 68, 0.3) 0%, transparent 60%)';
        } else {
            badge.className = 'badge safe';
            badge.innerHTML = '<i class="fa-solid fa-shield-check"></i> Safe Content';
            // Restore normal glow background
            document.querySelector('.glow-bg').style.background = 'radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.3) 0%, transparent 60%)';
        }
    }
});
