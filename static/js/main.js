/* ==========================================================================
   AI NON-CONTACT IOP SCREENING SYSTEM - FRONTEND INTERACTION LOGIC
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Navigation Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetTab = document.getElementById(btn.dataset.tab);
            if (targetTab) {
                targetTab.classList.add('active');
            }
        });
    });

    // 2. Camera Controls & Stream State
    let mediaStream = null;
    const videoElem = document.getElementById('webcam-stream');
    const startCamBtn = document.getElementById('btn-start-cam');
    const captureCamBtn = document.getElementById('btn-capture-cam');
    const stopCamBtn = document.getElementById('btn-stop-cam');
    const dropzone = document.getElementById('file-dropzone');
    const fileInput = document.getElementById('file-input');

    if (startCamBtn) {
        startCamBtn.addEventListener('click', async () => {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 600 }, height: { ideal: 600 }, facingMode: 'user' }
                });
                videoElem.srcObject = mediaStream;
                videoElem.style.display = 'block';
                document.getElementById('scanner-placeholder').style.display = 'none';

                startCamBtn.style.display = 'none';
                captureCamBtn.style.display = 'inline-flex';
                stopCamBtn.style.display = 'inline-flex';
            } catch (err) {
                alert('Camera access denied or webcam unavailable: ' + err.message);
            }
        });
    }

    if (stopCamBtn) {
        stopCamBtn.addEventListener('click', stopWebcam);
    }

    function stopWebcam() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        videoElem.style.display = 'none';
        document.getElementById('scanner-placeholder').style.display = 'flex';
        startCamBtn.style.display = 'inline-flex';
        captureCamBtn.style.display = 'none';
        stopCamBtn.style.display = 'none';
    }

    if (captureCamBtn) {
        captureCamBtn.addEventListener('click', () => {
            if (!videoElem.srcObject) return;
            const canvas = document.createElement('canvas');
            canvas.width = videoElem.videoWidth || 600;
            canvas.height = videoElem.videoHeight || 600;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

            analyzeImageData(dataUrl);
            stopWebcam();
        });
    }

    // 3. File Upload Dropzone
    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--accent-cyan)';
        });
        dropzone.addEventListener('dragleave', () => {
            dropzone.style.borderColor = 'rgba(0, 229, 255, 0.3)';
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'rgba(0, 229, 255, 0.3)';
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                processUploadedFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                processUploadedFile(e.target.files[0]);
            }
        });
    }

    function processUploadedFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            analyzeImageData(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    // 4. Preset Sample Click Listeners
    const presetCards = document.querySelectorAll('.preset-card');
    presetCards.forEach(card => {
        card.addEventListener('click', () => {
            const presetName = card.dataset.preset;
            fetchPresetAnalysis(presetName);
        });
    });

    // 5. Fetch API Calls
    async function analyzeImageData(base64Data) {
        showLoadingState(true);
        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data })
            });
            const data = await res.json();
            if (data.success) {
                renderAnalysisResults(data);
                // Switch to Telemetry Tab automatically
                document.querySelector('[data-tab="tab-telemetry"]').click();
            } else {
                alert('Analysis failed: ' + data.error);
            }
        } catch (err) {
            alert('Error communicating with analysis backend: ' + err.message);
        } finally {
            showLoadingState(false);
        }
    }

    async function fetchPresetAnalysis(presetName) {
        showLoadingState(true);
        try {
            const res = await fetch(`/api/preset/${presetName}`);
            const data = await res.json();
            if (data.success) {
                renderAnalysisResults(data);
                document.querySelector('[data-tab="tab-telemetry"]').click();
            } else {
                alert('Failed to process preset: ' + data.error);
            }
        } catch (err) {
            alert('Error loading preset: ' + err.message);
        } finally {
            showLoadingState(false);
        }
    }

    function showLoadingState(isLoading) {
        const loader = document.getElementById('global-loader');
        if (loader) {
            loader.style.display = isLoading ? 'flex' : 'none';
        }
    }

    // 6. Render Results to UI
    function renderAnalysisResults(data) {
        const { annotated_image, features, prediction } = data;

        // Annotated Overlay Image
        const imgElem = document.getElementById('result-annotated-img');
        if (imgElem) {
            imgElem.src = annotated_image;
        }

        // Risk Level Badge
        const badgeElem = document.getElementById('result-risk-badge');
        if (badgeElem) {
            badgeElem.textContent = prediction.risk_level;
            badgeElem.className = `risk-badge ${prediction.badge_class || getBadgeClass(prediction.risk_level)}`;
        }

        // IOP Range Display
        const iopElem = document.getElementById('result-iop-range');
        if (iopElem) {
            iopElem.textContent = prediction.iop_range_str;
        }

        // Confidence Score
        const confElem = document.getElementById('result-confidence');
        if (confElem) {
            confElem.textContent = `${prediction.confidence_pct}%`;
        }

        // Feature Telemetry Values
        setMetric('val-pupil-ratio', features.pupil_iris_ratio);
        setMetric('val-pupil-dia', `${features.pupil_diameter_px} px`);
        setMetric('val-iris-dia', `${features.iris_diameter_px} px`);
        setMetric('val-circularity', features.circularity_index);
        setMetric('val-symmetry', `${features.symmetry_score_pct}%`);
        setMetric('val-brightness', `${features.brightness_level} / 255`);

        // Clinical Recommendations
        const recText = document.getElementById('result-recommendation');
        if (recText) recText.textContent = prediction.recommendation;

        const actionText = document.getElementById('result-action');
        if (actionText) actionText.textContent = prediction.suggested_action;

        // Feature Importances Progress Bars
        renderFeatureImportances(prediction.feature_importances);
    }

    function setMetric(id, val) {
        const elem = document.getElementById(id);
        if (elem) elem.textContent = val;
    }

    function getBadgeClass(level) {
        if (level.includes('Normal')) return 'badge-normal';
        if (level.includes('Moderate')) return 'badge-moderate';
        return 'badge-high';
    }

    function renderFeatureImportances(importances) {
        const container = document.getElementById('feature-importance-container');
        if (!container || !importances) return;

        container.innerHTML = '';
        const labels = {
            'pupil_iris_ratio': 'Pupil-to-Iris Ratio',
            'circularity_index': 'Pupil Circularity',
            'symmetry_score_pct': 'Concentricity / Symmetry',
            'pupil_diameter_px': 'Pupil Diameter',
            'iris_diameter_px': 'Iris Diameter',
            'brightness_level': 'Reflectance / Luminance'
        };

        for (const [key, val] of Object.entries(importances)) {
            const pct = Math.round(val * 100);
            const row = document.createElement('div');
            row.style.marginBottom = '0.75rem';
            row.innerHTML = `
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.3rem;">
                    <span>${labels[key] || key}</span>
                    <span style="font-family: var(--font-mono); color: var(--accent-cyan);">${pct}%</span>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden;">
                    <div style="width: ${pct}%; height: 100%; background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)); border-radius: 4px;"></div>
                </div>
            `;
            container.appendChild(row);
        }
    }

    // 7. Load Datasets Tab Content
    async function loadDatasetsInfo() {
        try {
            const res = await fetch('/api/datasets');
            const data = await res.json();
            if (data.success && data.datasets) {
                const grid = document.getElementById('dataset-grid');
                if (!grid) return;
                grid.innerHTML = '';
                data.datasets.forEach(ds => {
                    const card = document.createElement('div');
                    card.className = 'glass-card';
                    card.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                            <h3 style="font-size: 1.1rem; color: var(--accent-cyan);">${ds.name}</h3>
                            <span class="flow-tag">${ds.status}</span>
                        </div>
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">${ds.full_name}</p>
                        <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent-emerald); margin-bottom: 0.5rem;">
                            📊 Volume: ${ds.samples}
                        </div>
                        <p style="font-size: 0.85rem; color: var(--text-muted);">${ds.utility}</p>
                    `;
                    grid.appendChild(card);
                });
            }
        } catch (err) {
            console.error('Error fetching datasets info:', err);
        }
    }

    loadDatasetsInfo();

    // 8. FAQ Accordion Click Listeners
    document.querySelectorAll('.faq-question').forEach(q => {
        q.addEventListener('click', () => {
            const answer = q.nextElementSibling;
            const isVisible = answer.style.display === 'block';
            document.querySelectorAll('.faq-answer').forEach(a => a.style.display = 'none');
            answer.style.display = isVisible ? 'none' : 'block';
        });
    });

    // Auto-load normal preset on start to populate telemetry default view
    fetchPresetAnalysis('normal');
});
