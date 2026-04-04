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

    // Environment Config: Replace this with your Railway deployment URL
    const API_BASE_URL = window.location.hostname === "localhost" 
                         ? "http://localhost:8000" 
                         : "https://your-app-name-production.up.railway.app";

    async function uploadFileToAPI(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            statusText.innerText = "Running Deep Learning Models (TCA & SER)...";
            const response = await fetch(`${API_BASE_URL}/api/analyze`, {
                method: 'POST',
                body: formData
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
        document.getElementById('emotion-text').innerText = data.detected_emotion;
        document.getElementById('confidence-text').innerText = data.confidence;

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
