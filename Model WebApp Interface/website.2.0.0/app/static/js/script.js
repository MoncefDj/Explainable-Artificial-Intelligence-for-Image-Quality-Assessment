// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    // Control Elements
    const imageIndexInput = document.getElementById('imageIndexInput');
    const maxIndexDisplay = document.getElementById('maxIndexDisplay');
    const datasetStatusEl = document.getElementById('datasetStatus');
    const useLlmCheckbox = document.getElementById('useLlmCheckbox');
    const llmAvailabilityStatusEl = document.getElementById('llmAvailabilityStatus');
    const analyzeButton = document.getElementById('analyzeButton');
    const analyzeButtonText = document.getElementById('analyzeButtonText');
    const analyzeButtonSpinner = document.getElementById('analyzeButtonSpinner');
    const darkModeToggle = document.getElementById('darkModeToggle');
    const imageUploadInput = document.getElementById('imageUploadInput');
    const uploadedImagePreview = document.getElementById('uploadedImagePreview');
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    const imageIndexContainer = document.getElementById('imageIndexContainer');
    const orSeparator = document.getElementById('orSeparator');
    const clearUploadedImageButton = document.getElementById('clearUploadedImageButton');

    // Progress Elements
    const progressSection = document.getElementById('progressSection');
    const progressBar = document.getElementById('progressBar');
    const progressStatusText = document.getElementById('progressStatusText');

    // Result Display Elements
    const resultsContainer = document.getElementById('resultsContainer');
    const initialMessageDiv = document.getElementById('initialMessage');
    const originalImageEl = document.getElementById('originalImage');
    const segmentationOverlayEl = document.getElementById('segmentationOverlay');
    const saliencyOverlayEl = document.getElementById('saliencyOverlay');
    const segmentationCaptionEl = document.getElementById('segmentationCaption');
    const saliencyCaptionEl = document.getElementById('saliencyCaption');
    const scoreAllObjectsEl = document.getElementById('scoreAllObjects');
    const numAllObjectsEl = document.getElementById('numAllObjects');
    const scoreFilteredObjectsEl = document.getElementById('scoreFilteredObjects');
    const numFilteredObjectsEl = document.getElementById('numFilteredObjects');
    const llmStatusReportEl = document.getElementById('llmStatusReport');
    const textSummaryReportDiv = document.getElementById('textSummaryReport'); 
    const llmExplanationDiv = document.getElementById('llmExplanation'); 
    const llmTabButton = document.getElementById('llm-tab');
    const exportPdfButton = document.getElementById('exportPdfButton');

    // Zoom Modal Elements
    const imageZoomModalEl = document.getElementById('imageZoomModal');
    let imageZoomModal = null; 
    if (imageZoomModalEl) {
        imageZoomModal = new bootstrap.Modal(imageZoomModalEl);
    }
    const zoomedImageEl = document.getElementById('zoomedImage');
    const imageZoomModalLabel = document.getElementById('imageZoomModalLabel');
    const zoomSpinner = document.getElementById('zoomSpinner');


    const defaultImagePlaceholder = "data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22256%22%20height%3D%22256%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20256%20256%22%20preserveAspectRatio%3D%22none%22%3E%3Cdefs%3E%3Cstyle%20type%3D%22text%2Fcss%22%3E%23holder_1%20text%20%7B%20fill%3A%23AAA%3Bfont-weight%3Abold%3Bfont-family%3AArial%2C%20Helvetica%2C%20Open%20Sans%2C%20sans-serif%2C%20monospace%3Bfont-size%3A13pt%20%7D%20%3C%2Fstyle%3E%3C%2Fdefs%3E%3Cg%20id%3D%22holder_1%22%3E%3Crect%20width%3D%22256%22%20height%3D%22256%22%20fill%3D%22%23EEE%22%3E%3C%2Frect%3E%3Cg%3E%3Ctext%20x%3D%2290%22%20y%3D%22133.6%22%3EWaiting...%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E";
    let currentMaxIndex = -1;
    let currentAnalysisDataForPdf = null; 

    if (window.marked && window.hljs) {
        marked.setOptions({
            highlight: function (code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language, ignoreIllegals: true }).value;
            },
            gfm: true, breaks: false, pedantic: false
        });
    } else { console.warn("Marked.js or Highlight.js not found."); }

    function renderJupyterMarkdown(markdownString, targetElement) {
        if (!targetElement) return;
        let htmlContent = (window.marked && typeof window.marked.parse === 'function') ? marked.parse(markdownString || "") : (markdownString || "").replace(/\n/g, '<br>');
        targetElement.innerHTML = htmlContent;
        if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
            Promise.resolve().then(() => window.MathJax.typesetPromise([targetElement])).catch((err) => console.error('MathJax typesetting error:', err));
        }
    }
    function setProgress(percentage, text = "") {
        const clampedPercentage = Math.min(Math.max(percentage, 0), 100);
        if (progressBar) {
            progressBar.style.width = `${clampedPercentage}%`;
            progressBar.setAttribute('aria-valuenow', clampedPercentage);
            const progressTextDisplay = document.getElementById('progressBarText');
            if(progressTextDisplay) progressTextDisplay.textContent = `${clampedPercentage}%`;
        }
        if (progressStatusText) progressStatusText.textContent = text;
        if (progressSection && clampedPercentage > 0 && progressSection.style.display === 'none') {
            progressSection.style.display = 'block';
        }
    }

    function updateUIWithResults(data) {
        currentAnalysisDataForPdf = data; 
        console.log("Data received in updateUIWithResults. Full-scale image data presence:", 
            `Original: ${!!data.full_scale_original_b64}`, 
            `Segmentation: ${!!data.full_scale_segmentation_b64}`, 
            `Saliency: ${!!data.full_scale_saliency_b64}`
        ); 
        
        if(exportPdfButton) exportPdfButton.style.display = 'block';

        if(originalImageEl) originalImageEl.src = data.original_image ? `data:image/png;base64,${data.original_image}` : defaultImagePlaceholder;
        if(segmentationOverlayEl) segmentationOverlayEl.src = data.segmentation_overlay ? `data:image/png;base64,${data.segmentation_overlay}` : defaultImagePlaceholder;
        if(saliencyOverlayEl) saliencyOverlayEl.src = data.saliency_overlay ? `data:image/png;base64,${data.saliency_overlay}` : defaultImagePlaceholder;

        if (segmentationCaptionEl) {
            if (data.segmentation_caption && data.segmentation_caption.trim() !== "") {
                segmentationCaptionEl.textContent = data.segmentation_caption;
                segmentationCaptionEl.style.display = 'block';
            } else {
                segmentationCaptionEl.style.display = 'none'; 
            }
        }
        if (saliencyCaptionEl) {
            if (data.saliency_caption && data.saliency_caption.trim() !== "") {
                saliencyCaptionEl.textContent = data.saliency_caption;
                saliencyCaptionEl.style.display = 'block';
            } else {
                saliencyCaptionEl.style.display = 'none'; 
            }
        }
        const metrics = data.quality_metrics || {};
        if(scoreAllObjectsEl) scoreAllObjectsEl.textContent = metrics.overall_image_quality_all !== null && metrics.overall_image_quality_all !== undefined ? metrics.overall_image_quality_all.toFixed(3) : 'N/A';
        if(numAllObjectsEl) numAllObjectsEl.textContent = `${metrics.num_objects_all === null || metrics.num_objects_all === undefined ? '-' : metrics.num_objects_all} objs`;
        if(scoreFilteredObjectsEl) scoreFilteredObjectsEl.textContent = metrics.overall_image_quality_filtered !== null && metrics.overall_image_quality_filtered !== undefined ? metrics.overall_image_quality_filtered.toFixed(3) : 'N/A';
        if(numFilteredObjectsEl) numFilteredObjectsEl.textContent = `${metrics.num_objects_filtered === null || metrics.num_objects_filtered === undefined ? '-' : metrics.num_objects_filtered} objs`;

        renderJupyterMarkdown(data.text_summary_report || "No summary report available.", textSummaryReportDiv);
        renderJupyterMarkdown(data.llm_explanation_text || "AI explanation not available or not requested.", llmExplanationDiv);
        
        if(llmStatusReportEl) llmStatusReportEl.textContent = `AI Explanation Status: ${data.llm_status || 'N/A'}`;
        if(resultsContainer) resultsContainer.style.display = 'block';
        if(initialMessageDiv) initialMessageDiv.style.display = 'none';
    }

    function resetUIForNewAnalysis() {
        currentAnalysisDataForPdf = null;
        if(exportPdfButton) exportPdfButton.style.display = 'none';
        if(originalImageEl) originalImageEl.src = defaultImagePlaceholder;
        if(segmentationOverlayEl) segmentationOverlayEl.src = defaultImagePlaceholder;
        if(saliencyOverlayEl) saliencyOverlayEl.src = defaultImagePlaceholder;
        if(segmentationCaptionEl) {segmentationCaptionEl.textContent = ''; segmentationCaptionEl.style.display = 'none';}
        if(saliencyCaptionEl) {saliencyCaptionEl.textContent = ''; saliencyCaptionEl.style.display = 'none';}
        if(scoreAllObjectsEl) scoreAllObjectsEl.textContent = "-";
        if(numAllObjectsEl) numAllObjectsEl.textContent = "- objs";
        if(scoreFilteredObjectsEl) scoreFilteredObjectsEl.textContent = "-";
        if(numFilteredObjectsEl) numFilteredObjectsEl.textContent = "- objs";
        renderJupyterMarkdown("Report will appear here...", textSummaryReportDiv);
        renderJupyterMarkdown("AI explanation will appear here...", llmExplanationDiv);
        if(llmStatusReportEl) llmStatusReportEl.textContent = "";
        if(resultsContainer) resultsContainer.style.display = 'none';
        if(initialMessageDiv) initialMessageDiv.style.display = 'block';
        if(progressSection) progressSection.style.display = 'none';
        if(progressBar) {
            progressBar.style.width = '0%'; progressBar.classList.remove('bg-danger', 'bg-success');
            const progressTextDisplay = document.getElementById('progressBarText');
            if(progressTextDisplay) progressTextDisplay.textContent = '0%';
        }
        if(progressStatusText) progressStatusText.textContent = "";
    }
    
    async function fetchInitialConfig() {
        console.log("fetchInitialConfig called");
        try {
            const response = await fetch('/get_initial_config');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const config = await response.json();
            console.log("Initial config received:", config);

            if (config.dataset_ok && config.max_index >= 0) {
                currentMaxIndex = config.max_index;
                if(maxIndexDisplay) maxIndexDisplay.textContent = currentMaxIndex;
                if(imageIndexInput) { imageIndexInput.max = currentMaxIndex; imageIndexInput.disabled = false; }
                if(analyzeButton && (!imageUploadInput || !imageUploadInput.files || !imageUploadInput.files.length)) analyzeButton.disabled = false;
                if(datasetStatusEl) { datasetStatusEl.textContent = config.dataset_message || `Dataset ready (${currentMaxIndex + 1} images).`; datasetStatusEl.className = "form-text text-success"; }
                if(imageIndexContainer) imageIndexContainer.style.display = 'block';
                if(orSeparator) orSeparator.style.display = 'block';
                if(imageUploadInput) imageUploadInput.disabled = false;
            } else {
                if(datasetStatusEl) { datasetStatusEl.textContent = config.dataset_message || "Error: Dataset not available."; datasetStatusEl.className = "form-text text-danger"; }
                if(imageIndexInput) imageIndexInput.disabled = true;
                if(imageIndexContainer) imageIndexContainer.style.display = 'none';
                if(orSeparator) orSeparator.style.display = 'none';
                if(analyzeButton && (!imageUploadInput || !imageUploadInput.files || !imageUploadInput.files.length)) analyzeButton.disabled = true; 
            }
            if (config.llm_available) {
                if(llmAvailabilityStatusEl) { llmAvailabilityStatusEl.textContent = "AI engine available for explanations."; llmAvailabilityStatusEl.className = "form-text text-success small"; }
                if(useLlmCheckbox) { useLlmCheckbox.disabled = false; useLlmCheckbox.checked = config.llm_initially_enabled; }
                if (llmTabButton) llmTabButton.style.display = '';
            } else {
                if(llmAvailabilityStatusEl) { llmAvailabilityStatusEl.textContent = "AI engine (LLM) not available on server."; llmAvailabilityStatusEl.className = "form-text text-muted small"; }
                if(useLlmCheckbox) { useLlmCheckbox.checked = false; useLlmCheckbox.disabled = true; }
                if (llmTabButton) llmTabButton.style.display = 'none';
            }
            const canAnalyzeDataset = config.dataset_ok && config.max_index >= 0;
            if (!canAnalyzeDataset) { 
                if(initialMessageDiv) { initialMessageDiv.textContent = 'Dataset not loaded. You can upload an image for analysis.'; initialMessageDiv.className = "text-center mt-3 alert alert-warning lead"; }
                if(analyzeButton && (!imageUploadInput || !imageUploadInput.files || !imageUploadInput.files.length)) {
                    analyzeButton.disabled = true;
                } else if (analyzeButton && imageUploadInput && imageUploadInput.files && imageUploadInput.files.length > 0) {
                    analyzeButton.disabled = false;
                }
            } else { 
                if(initialMessageDiv) { initialMessageDiv.textContent = 'Select an image index or upload an image, then click "Start Analysis".'; initialMessageDiv.className = "text-center mt-3 alert alert-info lead"; }
                if (analyzeButton) {
                     analyzeButton.disabled = !( (imageIndexInput && imageIndexInput.value !== "") || (imageUploadInput && imageUploadInput.files && imageUploadInput.files.length > 0) );
                }
            }
        } catch (error) {
            console.error("Error fetching initial config:", error);
            if(datasetStatusEl) { datasetStatusEl.textContent = "Failed to load server configuration."; datasetStatusEl.className = "form-text text-danger"; }
            if(imageIndexInput) imageIndexInput.disabled = true;
            if(analyzeButton) analyzeButton.disabled = true;
            if(useLlmCheckbox) useLlmCheckbox.disabled = true;
            if(imageUploadInput) imageUploadInput.disabled = true;
            if(llmAvailabilityStatusEl) llmAvailabilityStatusEl.textContent = "AI status unknown (config error).";
            if (llmTabButton) llmTabButton.style.display = 'none';
            if(initialMessageDiv) { initialMessageDiv.textContent = "Error connecting to the server. Please try again later."; initialMessageDiv.className = "text-center mt-3 alert alert-danger lead"; }
        }
    }

    if (imageIndexInput) {
        imageIndexInput.addEventListener('input', () => {
            let valStr = imageIndexInput.value;
            if (valStr === "") { 
                 if (imageUploadInput && imageUploadInput.files.length > 0) { analyzeButton.disabled = false; } 
                 else { analyzeButton.disabled = true; } 
                return;
            }
            let val = parseInt(valStr);
            if (isNaN(val) || val < 0) { imageIndexInput.value = 0; }
            else if (currentMaxIndex >= 0 && val > currentMaxIndex) { imageIndexInput.value = currentMaxIndex; }
            if (imageUploadInput && imageUploadInput.files.length > 0) { clearUploadedImage(); } 
            analyzeButton.disabled = false; 
        });
    }

    if (imageUploadInput) {
        imageUploadInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    if(uploadedImagePreview) uploadedImagePreview.src = e.target.result;
                    if(uploadedImageContainer) uploadedImageContainer.style.display = 'block';
                    if(analyzeButton) analyzeButton.disabled = false; 
                    if (imageIndexInput) imageIndexInput.value = ""; 
                    if (datasetStatusEl && currentMaxIndex >=0 ) datasetStatusEl.textContent = `Using uploaded image. (Dataset of ${currentMaxIndex + 1} images available via index)`;
                    else if (datasetStatusEl) datasetStatusEl.textContent = "Using uploaded image.";
                }
                reader.readAsDataURL(file);
            } else { clearUploadedImage(); } 
        });
    }

    function clearUploadedImage() {
        if (imageUploadInput) imageUploadInput.value = null;
        if (uploadedImagePreview) uploadedImagePreview.src = '#';
        if (uploadedImageContainer) uploadedImageContainer.style.display = 'none';
        if (datasetStatusEl && currentMaxIndex >=0 ) datasetStatusEl.textContent = `Dataset ready (${currentMaxIndex + 1} images).`;
        else if (datasetStatusEl) datasetStatusEl.textContent = "Dataset not available."; 
        if (imageIndexInput && !imageIndexInput.value) {
            if (analyzeButton) analyzeButton.disabled = true;
        } else if (imageIndexInput && imageIndexInput.value && !imageIndexInput.disabled) { 
            if (analyzeButton) analyzeButton.disabled = false;
        }
    }
    if (clearUploadedImageButton) { clearUploadedImageButton.addEventListener('click', clearUploadedImage); }

    if (analyzeButton) {
        analyzeButton.addEventListener('click', async () => {
            const selectedIndexStr = imageIndexInput ? imageIndexInput.value : "";
            const uploadedFile = imageUploadInput ? imageUploadInput.files[0] : null;
            let analysisSourceDescription = "";
            const formData = new FormData();
            formData.append('use_llm', useLlmCheckbox ? useLlmCheckbox.checked : false);

            if (uploadedFile) {
                formData.append('image_file', uploadedFile);
                analysisSourceDescription = `uploaded image "${uploadedFile.name}"`;
            } else if (selectedIndexStr !== "") {
                const selectedIndex = parseInt(selectedIndexStr);
                if (isNaN(selectedIndex) || selectedIndex < 0 || (currentMaxIndex >= 0 && selectedIndex > currentMaxIndex) ) {
                    alert("Please enter a valid image index (0 to " + (currentMaxIndex >= 0 ? currentMaxIndex : 'N/A') + ") or upload an image.");
                    if(imageIndexInput) imageIndexInput.focus(); return;
                }
                formData.append('image_index', selectedIndex);
                analysisSourceDescription = `image ${selectedIndex}`;
            } else { alert("Please select an image index or upload an image for analysis."); return; }

            resetUIForNewAnalysis();
            if(progressSection) progressSection.style.display = 'block';
            setProgress(5, `Requesting analysis for ${analysisSourceDescription}...`);
            if(analyzeButton) analyzeButton.disabled = true;
            if(analyzeButtonText) analyzeButtonText.style.display = 'none';
            if(analyzeButtonSpinner) analyzeButtonSpinner.style.display = 'inline-block';
            let progress = 5;
            const progressInterval = setInterval(() => {
                progress += Math.floor(Math.random() * 5) + 1;
                if (progress <= 90) { setProgress(progress, `Processing ${analysisSourceDescription} on server...`);}
                else { setProgress(90, `Finalizing analysis for ${analysisSourceDescription}...`);}
            }, 300);
            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                clearInterval(progressInterval);
                if (response.ok) {
                    const results = await response.json();
                    setProgress(100, `Analysis complete for ${results.image_source_analyzed || analysisSourceDescription}.`);
                    if(progressBar) { progressBar.classList.remove('bg-danger'); progressBar.classList.add('bg-success'); }
                    updateUIWithResults(results); 
                    setTimeout(() => { if(progressSection) progressSection.style.display = 'none'; }, 2000);
                } else {
                    const errorData = await response.json().catch(() => ({error: "Unknown server error"}));
                    setProgress(100, `Error: ${errorData.error || response.statusText}`);
                    if(progressBar) { progressBar.classList.remove('bg-success'); progressBar.classList.add('bg-danger'); }
                    console.error('Analysis Error:', errorData);
                }
            } catch (error) {
                clearInterval(progressInterval); console.error('Analysis request failed:', error);
                setProgress(100, 'Request failed. Check console/server logs.');
                if(progressBar) { progressBar.classList.remove('bg-success'); progressBar.classList.add('bg-danger'); }
            } finally {
                if (imageUploadInput && imageUploadInput.files.length > 0) { analyzeButton.disabled = false; }
                else if (imageIndexInput && imageIndexInput.value !== "" && !imageIndexInput.disabled) { analyzeButton.disabled = false; }
                else { analyzeButton.disabled = true; }
                if(analyzeButtonText) analyzeButtonText.style.display = 'inline-block';
                if(analyzeButtonSpinner) analyzeButtonSpinner.style.display = 'none';
            }
        });
    }

    if (darkModeToggle) { 
        const moonIcon = darkModeToggle.querySelector('.fa-moon');
        const sunIcon = darkModeToggle.querySelector('.fa-sun');
        const highlightLinkLight = document.getElementById('highlight-theme-light');
        const highlightLinkDark = document.getElementById('highlight-theme-dark');
        const applyTheme = (theme) => {
            document.body.setAttribute('data-bs-theme', theme);
            if (moonIcon && sunIcon) {
                if (theme === 'dark') {
                    moonIcon.style.display = 'none'; sunIcon.style.display = 'inline-block';
                    if (highlightLinkDark) highlightLinkDark.removeAttribute('disabled');
                    if (highlightLinkLight) highlightLinkLight.setAttribute('disabled', 'true');
                } else {
                    moonIcon.style.display = 'inline-block'; sunIcon.style.display = 'none';
                    if (highlightLinkLight) highlightLinkLight.removeAttribute('disabled');
                    if (highlightLinkDark) highlightLinkDark.setAttribute('disabled', 'true');
                }
            }
        };
        let preferredTheme = localStorage.getItem('theme') || 'light'; 
        applyTheme(preferredTheme);
        darkModeToggle.addEventListener('click', () => {
            let currentTheme = document.body.getAttribute('data-bs-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('theme', newTheme); applyTheme(newTheme);
        });
    }

    // --- PDF EXPORT (Reverted to Server-Side Call) ---
    if (exportPdfButton) {
        exportPdfButton.addEventListener('click', async () => {
            if (!currentAnalysisDataForPdf || !currentAnalysisDataForPdf.image_source_analyzed_for_filename) {
                alert("No analysis data available to export. Please run an analysis first.");
                return;
            }

            exportPdfButton.disabled = true;
            const originalButtonText = exportPdfButton.innerHTML;
            exportPdfButton.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Requesting PDF...`;
            console.log("PDF Export: Requesting server-side PDF generation.");

            try {
                const response = await fetch('/export_report', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ analysis_data: currentAnalysisDataForPdf })
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const filenameFromServer = response.headers.get('Content-Disposition')?.split('filename=')[1]?.replace(/"/g, '');
                    const defaultFilename = `IQA_Report_ServerSide_${currentAnalysisDataForPdf.image_source_analyzed_for_filename}.pdf`;
                    const filename = filenameFromServer || defaultFilename;
                    
                    const link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = filename;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(link.href);
                    console.log("PDF Export: Download initiated from server response.");
                } else {
                    const errorData = await response.json().catch(() => ({ error: "PDF export failed on server with status: " + response.status }));
                    alert(`Error exporting PDF: ${errorData.error}`);
                    console.error("PDF Export Error (Server):", errorData);
                }
            } catch (error) {
                alert("Failed to request PDF export. Check console or network tab.");
                console.error("PDF Export Request Error:", error);
            } finally {
                exportPdfButton.disabled = false;
                exportPdfButton.innerHTML = originalButtonText;
                console.log("PDF Export: Process finished. Button re-enabled.");
            }
        });
    }
    
    // --- Zoom Functionality ---
    document.querySelectorAll('.zoom-icon').forEach(icon => {
        icon.addEventListener('click', async (event) => { 
            if (!currentAnalysisDataForPdf || !imageZoomModal) {
                alert("Please run an analysis first to enable zoom, or modal not found.");
                return;
            }
            const plotType = event.currentTarget.dataset.plotType;
            if(imageZoomModalLabel) imageZoomModalLabel.textContent = `Zoomed: ${plotType.charAt(0).toUpperCase() + plotType.slice(1)}`;
            
            let imageBase64ForZoom = null;
            switch(plotType) {
                case 'original':
                    imageBase64ForZoom = currentAnalysisDataForPdf.full_scale_original_b64;
                    break;
                case 'segmentation':
                    imageBase64ForZoom = currentAnalysisDataForPdf.full_scale_segmentation_b64;
                    break;
                case 'saliency':
                    imageBase64ForZoom = currentAnalysisDataForPdf.full_scale_saliency_b64;
                    break;
            }

            if (imageBase64ForZoom) { 
                if(zoomedImageEl) {
                    zoomedImageEl.src = `data:image/png;base64,${imageBase64ForZoom}`;
                    zoomedImageEl.style.display = 'block';
                }
                if(zoomSpinner) zoomSpinner.style.display = 'none';
                imageZoomModal.show();
            } else { 
                console.warn(`Full-scale image for zoom (${plotType}) not found directly in analysis data. Using preview as fallback if available, or attempting fetch.`);
                let fallbackImageBase64 = null;
                 switch(plotType) {
                    case 'original': fallbackImageBase64 = currentAnalysisDataForPdf.original_image; break;
                    case 'segmentation': fallbackImageBase64 = currentAnalysisDataForPdf.segmentation_overlay; break;
                    case 'saliency': fallbackImageBase64 = currentAnalysisDataForPdf.saliency_overlay; break;
                }
                if (fallbackImageBase64) {
                    console.log(`Zoom: Using preview image for ${plotType} as full-scale was missing.`);
                    if(zoomedImageEl) {
                        zoomedImageEl.src = `data:image/png;base64,${fallbackImageBase64}`;
                        zoomedImageEl.style.display = 'block';
                    }
                    if(zoomSpinner) zoomSpinner.style.display = 'none';
                    imageZoomModal.show();
                } else {
                    console.error(`Full-scale AND preview image for zoom (${plotType}) not found. No fallback available in current data. You might need to re-enable on-demand fetching if this occurs.`);
                     if(zoomedImageEl) { zoomedImageEl.src = defaultImagePlaceholder; zoomedImageEl.style.display = 'block'; } // Show placeholder
                    if(zoomSpinner) zoomSpinner.style.display = 'none';
                    imageZoomModal.show();
                    alert(`Could not load high-resolution image for ${plotType}. Displaying placeholder if available.`);
                }
            }
        });
    });

    resetUIForNewAnalysis();
    fetchInitialConfig();   
});