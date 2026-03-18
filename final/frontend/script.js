/* Configuration */
const API_BASE_URL = 'http://127.0.0.1:5000';

/* Elements */
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const removeFileBtn = document.getElementById('removeFileBtn');

/* Buttons */
const btnResnet = document.getElementById('btnResnet');
const btnVit = document.getElementById('btnVit');
const btnYolo = document.getElementById('btnYolo');
const btnHybrid = document.getElementById('btnHybrid');
const allButtons = [btnResnet, btnVit, btnYolo, btnHybrid];

/* States */
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');

/* Result Elements */
const processedImage = document.getElementById('processedImage');
const scanModelType = document.getElementById('scanModelType');
const predictionHighlight = document.getElementById('predictionHighlight');
const predictionLabel = document.getElementById('predictionLabel');
const clinicalNote = document.getElementById('clinicalNote');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const accuracyValue = document.getElementById('accuracyValue');
const inferenceTime = document.getElementById('inferenceTime');
const analyticsSection = document.getElementById('analyticsSection');

// Extended Metrics DOM
const precisionPerc = document.getElementById('precisionPerc');
const precisionBar = document.getElementById('precisionBar');
const recallPerc = document.getElementById('recallPerc');
const recallBar = document.getElementById('recallBar');
const f1Perc = document.getElementById('f1Perc');
const f1Bar = document.getElementById('f1Bar');

let chartInstance = null;
let selectedFile = null;

/* ==========================================
   File Handling
================================================== */
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', function() {
    if (this.files && this.files.length > 0) {
        handleFile(this.files[0]);
    }
});

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
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

removeFileBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewArea.style.display = 'none';
    dropZone.style.display = 'block';
    resultsSection.style.display = 'none';
    analyticsSection.style.display = 'none';
    toggleButtons(false);
});

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png'];
    const validExts = ['.jpg', '.jpeg', '.png', '.dcm'];
    
    const isDicom = file.name.toLowerCase().endsWith('.dcm');
    
    if (!validTypes.includes(file.type) && !isDicom) {
        showToast('Invalid file format. Please upload JPG, PNG, or DICOM.', 'error');
        return;
    }

    selectedFile = file;
    
    // Show Preview
    if (!isDicom) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.style.display = 'none';
            previewArea.style.display = 'inline-block';
        };
        reader.readAsDataURL(file);
    } else {
        imagePreview.src = 'https://via.placeholder.com/224x224.png?text=DICOM+Scan';
        dropZone.style.display = 'none';
        previewArea.style.display = 'inline-block';
    }

    toggleButtons(true);
    resultsSection.style.display = 'none';
    analyticsSection.style.display = 'none';
}

function toggleButtons(enabled) {
    allButtons.forEach(btn => btn.disabled = !enabled);
}

/* ==========================================
   Analysis Triggers
================================================== */
btnResnet.addEventListener('click', () => runAnalysis('ResNet-50 Analysis'));
btnVit.addEventListener('click', () => runAnalysis('ViT SOTA Analysis'));
btnYolo.addEventListener('click', () => runAnalysis('YOLOv8 Localization'));
btnHybrid.addEventListener('click', () => runAnalysis('Hybrid Architecture Combined'));

async function runAnalysis(modelName) {
    if (!selectedFile) return;

    toggleButtons(false);
    loadingIndicator.style.display = 'flex';
    resultsSection.style.display = 'none';
    analyticsSection.style.display = 'none';
    
    const startTime = performance.now();
    const formData = new FormData();
    formData.append('file', selectedFile);
    // Passing the model name to the backend (even if it's currently processing everything with one pipeline, it's good practice for this UI)
    formData.append('model', modelName);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed.');
        }

        const endTime = performance.now();
        const inferenceMs = Math.round(endTime - startTime);

        displayResults(data, modelName, inferenceMs);
        showToast(`Analysis completed successfully using ${modelName}.`, 'success');

    } catch (error) {
        console.error('Fetch error:', error);
        showToast(error.message || 'Error communicating with backend.', 'error');
    } finally {
        toggleButtons(true);
        loadingIndicator.style.display = 'none';
    }
}

/* ==========================================
   Results Rendering
================================================== */
function displayResults(data, modelName, inferenceMs) {
    resultsSection.style.display = 'block';
    
    scanModelType.textContent = `Model: ${modelName}`;
    predictionLabel.textContent = data.prediction || "Unknown";
    
    const isFatty = data.prediction && data.prediction.toLowerCase().includes('fatty');
    const confidence = parseFloat(data.confidence || 0).toFixed(1);
    
    predictionHighlight.className = 'prediction-highlight ' + (isFatty ? 'danger' : 'success');
    
    if (isFatty) {
        clinicalNote.textContent = "Indicates significant fat deposition. Clinical correlation required.";
        confidenceFill.className = 'progress-fill danger';
        confidenceValue.className = 'gauge-value high';
    } else {
        clinicalNote.textContent = "No significant abnormalities detected in this view.";
        confidenceFill.className = 'progress-fill success';
        confidenceValue.className = 'gauge-value low';
    }

    confidenceValue.textContent = `${confidence}%`;
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);

    accuracyValue.textContent = data.accuracy ? `${parseFloat(data.accuracy).toFixed(1)}%` : '--%';
    // Adding some mock variance in inference time based on the model if desired, but here we just use the real ms.
    inferenceTime.textContent = `${inferenceMs} ms`;

    // Reset image
    processedImage.src = '';
    processedImage.style.display = 'none';

    if (data.heatmap_image) {
        const cacheBuster = `?t=${new Date().getTime()}`;
        if (data.heatmap_image.startsWith('/uploads/')) {
            processedImage.src = API_BASE_URL + data.heatmap_image + cacheBuster;
        } else if (data.heatmap_image.startsWith('data:')) {
            processedImage.src = data.heatmap_image;
        } else {
            processedImage.src = `data:image/jpeg;base64,${data.heatmap_image}`;
        }
        processedImage.style.display = 'block';
    } else {
        processedImage.src = imagePreview.src;
        processedImage.style.display = 'block';
    }
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 100);

    // Extended metrics rendering dynamic values straight from backend
    if (data.precision !== undefined && data.recall !== undefined && data.f1_score !== undefined) {
        analyticsSection.style.display = 'block';

        const precision = parseFloat(data.precision || 0).toFixed(1);
        const recall = parseFloat(data.recall || 0).toFixed(1);
        const f1 = parseFloat(data.f1_score || 0).toFixed(1);

        precisionPerc.textContent = `${precision}%`;
        precisionBar.style.width = `${precision}%`;

        recallPerc.textContent = `${recall}%`;
        recallBar.style.width = `${recall}%`;

        f1Perc.textContent = `${f1}%`;
        f1Bar.style.width = `${f1}%`;

        // Render Confusion Matrix Chart
        if (data.confusion_matrix) {
            renderConfusionMatrix(data.confusion_matrix, data.classes || data.cm_labels);
        }
    } else {
        analyticsSection.style.display = 'none';
    }
}

/* ==========================================
   Chart.js Confusion Matrix Visualization
================================================== */
function renderConfusionMatrix(matrixData, classes) {
    if (!matrixData || matrixData.length < 2) return;

    if (chartInstance) {
        chartInstance.destroy();
    }

    const ctx = document.getElementById('cmCanvas').getContext('2d');
    const labels = classes && classes.length === matrixData.length ? classes : matrixData.map((_, i) => `Class ${i}`);

    const correctData = [];
    const incorrectData = [];

    for (let i = 0; i < matrixData.length; i++) {
        correctData.push(matrixData[i][i]);
        let incorrectSum = 0;
        for (let j = 0; j < matrixData.length; j++) {
            if (i !== j) {
                incorrectSum += matrixData[i][j]; // False negatives (Actual i, Predicted j)
            }
        }
        incorrectData.push(incorrectSum);
    }

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Correct Predictions',
                    data: correctData,
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Incorrect Predictions',
                    data: incorrectData,
                    backgroundColor: 'rgba(239, 68, 68, 0.7)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Samples'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Prediction Performance by Class'
                }
            }
        }
    });
}

/* ==========================================
   Toast Notifications
================================================== */
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-check';
    
    toast.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Trigger reflow
    void toast.offsetWidth;
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
