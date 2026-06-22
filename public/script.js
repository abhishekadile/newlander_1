document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const resultContainer = document.getElementById('resultContainer');
    const detectionCanvas = document.getElementById('detectionCanvas');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const statsPanel = document.getElementById('statsPanel');
    const detectBtn = document.getElementById('detectBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    // Params
    const thresholdSlider = document.getElementById('threshold_value');
    const threshValDisplay = document.getElementById('threshValDisplay');
    const minRadiusInput = document.getElementById('min_radius');
    const maxRadiusInput = document.getElementById('max_radius');
    const colorGroupingInput = document.getElementById('enable_color_grouping');
    
    // State
    let currentFile = null;
    let currentImage = null;
    let detectionResults = null;

    // --- Event Listeners ---

    // File Upload
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Controls
    thresholdSlider.addEventListener('input', (e) => {
        threshValDisplay.textContent = e.target.value;
    });

    detectBtn.addEventListener('click', () => {
        if (currentFile) runDetection();
    });

    clearBtn.addEventListener('click', () => {
        resetWorkspace();
    });

    // --- Functions ---

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;
        
        // Load image to canvas immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImage = new Image();
            currentImage.onload = () => {
                showCanvas(currentImage);
                detectBtn.classList.remove('blocked'); // Enable detection
            };
            currentImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function showCanvas(img) {
        // Switch view
        dropZone.style.display = 'none';
        resultContainer.classList.remove('hidden');
        
        // Setup canvas sizing
        // We'll draw the image at its natural size or scaled to fit?
        // Let's render at natural size but use CSS to fit container.
        detectionCanvas.width = img.naturalWidth;
        detectionCanvas.height = img.naturalHeight;
        
        const ctx = detectionCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
    }

    async function runDetection() {
        if (!currentFile) return;

        showLoading(true);
        
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('threshold_value', thresholdSlider.value);
        formData.append('min_radius', minRadiusInput.value);
        formData.append('max_radius', maxRadiusInput.value);
        formData.append('enable_color_grouping', colorGroupingInput.checked);
        
        // Get radio value
        const type = document.querySelector('input[name="threshold_type"]:checked').value;
        formData.append('threshold_type', type);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                detectionResults = data;
                drawResults(detectionResults.colonies);
                updateStats(detectionResults);
            } else {
                alert('Detection failed: ' + data.error);
            }

        } catch (error) {
            console.error(error);
            alert('An error occurred during detection.');
        } finally {
            showLoading(false);
        }
    }

    function drawResults(colonies) {
        const ctx = detectionCanvas.getContext('2d');
        
        // Redraw image first to clear old markings
        ctx.drawImage(currentImage, 0, 0);

        // Style
        ctx.lineWidth = 2; // Fixed width, maybe scale with image size?
        ctx.strokeStyle = '#00ff00';
        ctx.font = "bold 16px Arial";
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        
        colonies.forEach((colony, index) => {
            const { x, y, radius } = colony;
            
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.stroke();

            // Draw number
            ctx.fillText(index + 1, x, y - radius - 5);
        });
    }

    function updateStats(results) {
        document.getElementById('countValue').textContent = results.colonyCount;
        
        const avgRadius = results.colonies.length > 0 
            ? results.colonies.reduce((acc, c) => acc + c.radius, 0) / results.colonies.length 
            : 0;
            
        document.getElementById('radiusValue').textContent = avgRadius.toFixed(2) + ' px';
    }

    function showLoading(active) {
        if (active) {
            loadingOverlay.classList.remove('hidden');
        } else {
            loadingOverlay.classList.add('hidden');
        }
    }

    function resetWorkspace() {
        currentFile = null;
        currentImage = null;
        detectionResults = null;
        
        dropZone.style.display = 'flex';
        resultContainer.classList.add('hidden');
        detectBtn.classList.add('blocked');
        
        // Clear inputs if needed
        document.getElementById('fileInput').value = '';
    }
});
