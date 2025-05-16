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
    const scoreAllObjectsEl = document.getElementById('scoreAllObjects');
    const numAllObjectsEl = document.getElementById('numAllObjects');
    const scoreFilteredObjectsEl = document.getElementById('scoreFilteredObjects');
    const numFilteredObjectsEl = document.getElementById('numFilteredObjects');
    const llmStatusReportEl = document.getElementById('llmStatusReport');
    const textSummaryReportDiv = document.getElementById('textSummaryReport'); 
    const llmExplanationDiv = document.getElementById('llmExplanation');    
    const llmTabButton = document.getElementById('llm-tab'); 

    const defaultImagePlaceholder = "data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22256%22%20height%3D%22256%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20256%20256%22%20preserveAspectRatio%3D%22none%22%3E%3Cdefs%3E%3Cstyle%20type%3D%22text%2Fcss%22%3E%23holder_1%20text%20%7B%20fill%3A%23AAA%3Bfont-weight%3Abold%3Bfont-family%3AArial%2C%20Helvetica%2C%20Open%20Sans%2C%20sans-serif%2C%20monospace%3Bfont-size%3A13pt%20%7D%20%3C%2Fstyle%3E%3C%2Fdefs%3E%3Cg%20id%3D%22holder_1%22%3E%3Crect%20width%3D%22256%22%20height%3D%22256%22%20fill%3D%22%23EEE%22%3E%3C%2Frect%3E%3Cg%3E%3Ctext%20x%3D%2290%22%20y%3D%22133.6%22%3EWaiting...%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E";
    let currentMaxIndex = -1;

    // Configure Marked.js to use Highlight.js
    if (window.marked && window.hljs) {
        marked.setOptions({
            highlight: function (code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language, ignoreIllegals: true }).value;
            },
            gfm: true,      // Enable GitHub Flavored Markdown
            breaks: false,  // Convert single newlines in paragraphs to <br> (false for standard behavior)
            pedantic: false // Don't be too strict
        });
        console.log("Marked.js initialized with Highlight.js.");
    } else {
        console.warn("Marked.js or Highlight.js not found. Code block highlighting may not work.");
    }

    function renderJupyterMarkdown(markdownString, targetElement) {
        if (!targetElement) {
            // console.error("Target element for Markdown rendering not found.");
            return;
        }
    
        let htmlContent = "";
        if (window.marked && typeof window.marked.parse === 'function') {
            htmlContent = marked.parse(markdownString || ""); // Ensure string is not null/undefined
        } else {
            htmlContent = (markdownString || "").replace(/\n/g, '<br>');
        }
        targetElement.innerHTML = htmlContent;
    
        // No need to manually call hljs.highlightElement if using marked's highlight option
    
        if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
            setTimeout(() => { 
                 window.MathJax.typesetPromise([targetElement])
                    .catch((err) => console.error('MathJax typesetting error on element:', targetElement, err));
            }, 0); 
        }
    }


    function setProgress(percentage, text = "") { /* ... (same as before) ... */ 
        const clampedPercentage = Math.min(Math.max(percentage, 0), 100);
        if (progressBar) {
            progressBar.style.width = `${clampedPercentage}%`;
            progressBar.textContent = `${clampedPercentage}%`;
            progressBar.setAttribute('aria-valuenow', clampedPercentage);
        }
        if (progressStatusText) {
            progressStatusText.textContent = text;
        }
        if (progressSection && clampedPercentage > 0 && progressSection.style.display === 'none') {
            progressSection.style.display = 'block';
        }
    }
    
    function updateUIWithResults(data) { /* ... (image and score updates same as before) ... */ 
        if(originalImageEl) originalImageEl.src = data.original_image ? `data:image/png;base64,${data.original_image}` : defaultImagePlaceholder;
        if(segmentationOverlayEl) segmentationOverlayEl.src = data.segmentation_overlay ? `data:image/png;base64,${data.segmentation_overlay}` : defaultImagePlaceholder;
        if(saliencyOverlayEl) saliencyOverlayEl.src = data.saliency_overlay ? `data:image/png;base64,${data.saliency_overlay}` : defaultImagePlaceholder;

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

    function resetUIForNewAnalysis() { /* ... (image and score resets same as before) ... */ 
        if(originalImageEl) originalImageEl.src = defaultImagePlaceholder;
        if(segmentationOverlayEl) segmentationOverlayEl.src = defaultImagePlaceholder;
        if(saliencyOverlayEl) saliencyOverlayEl.src = defaultImagePlaceholder;
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
            progressBar.style.width = '0%'; progressBar.textContent = '0%';
            progressBar.classList.remove('bg-danger', 'bg-success'); 
        }
        if(progressStatusText) progressStatusText.textContent = "";
    }
    
    async function fetchInitialConfig() { /* ... (same as before) ... */ 
        try {
            const response = await fetch('/get_initial_config');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const config = await response.json();

            if (config.dataset_ok && config.max_index >= 0) {
                currentMaxIndex = config.max_index;
                if(maxIndexDisplay) maxIndexDisplay.textContent = currentMaxIndex;
                if(imageIndexInput) imageIndexInput.max = currentMaxIndex; 
                if(imageIndexInput) imageIndexInput.disabled = false;
                if(analyzeButton) analyzeButton.disabled = false;
                if(datasetStatusEl) datasetStatusEl.textContent = config.dataset_message || `Dataset ready (${currentMaxIndex + 1} images).`;
                if(datasetStatusEl) datasetStatusEl.className = "form-text text-success";
            } else {
                if(datasetStatusEl) datasetStatusEl.textContent = config.dataset_message || "Error: Dataset not available.";
                if(datasetStatusEl) datasetStatusEl.className = "form-text text-danger";
                if(imageIndexInput) imageIndexInput.disabled = true; 
                if(analyzeButton) analyzeButton.disabled = true;
            }

            if (config.llm_available) {
                if(llmAvailabilityStatusEl) llmAvailabilityStatusEl.textContent = "AI engine available for explanations.";
                if(llmAvailabilityStatusEl) llmAvailabilityStatusEl.className = "form-text text-success small";
                if(useLlmCheckbox) useLlmCheckbox.disabled = false;
                if(useLlmCheckbox) useLlmCheckbox.checked = config.llm_initially_enabled;
                if (llmTabButton) llmTabButton.style.display = '';
            } else {
                if(llmAvailabilityStatusEl) llmAvailabilityStatusEl.textContent = "AI engine (LLM) not available on server.";
                if(llmAvailabilityStatusEl) llmAvailabilityStatusEl.className = "form-text text-muted small";
                if(useLlmCheckbox) useLlmCheckbox.checked = false; 
                if(useLlmCheckbox) useLlmCheckbox.disabled = true;
                if (llmTabButton) llmTabButton.style.display = 'none';
            }
            if (imageIndexInput && imageIndexInput.disabled) { 
                if(initialMessageDiv) initialMessageDiv.textContent = "Dataset not loaded. Please check server status.";
                if(initialMessageDiv) initialMessageDiv.className = "text-center mt-5 alert alert-danger lead";
            } else {
                if(initialMessageDiv) initialMessageDiv.textContent = 'Please enter an image index and click "Start Analysis" to begin.';
                if(initialMessageDiv) initialMessageDiv.className = "text-center mt-5 alert alert-info lead";
            }
        } catch (error) {
            console.error("Error fetching initial config:", error);
            if(datasetStatusEl) { datasetStatusEl.textContent = "Failed to load server configuration."; datasetStatusEl.className = "form-text text-danger"; }
            if(imageIndexInput) imageIndexInput.disabled = true; 
            if(analyzeButton) analyzeButton.disabled = true;
            if(useLlmCheckbox) useLlmCheckbox.disabled = true;
            if(llmAvailabilityStatusEl) llmAvailabilityStatusEl.textContent = "AI status unknown (config error).";
            if (llmTabButton) llmTabButton.style.display = 'none';
            if(initialMessageDiv) { initialMessageDiv.textContent = "Error connecting to the server. Please try again later."; initialMessageDiv.className = "text-center mt-5 alert alert-danger lead"; }
        }
    }

    if (imageIndexInput) { /* ... (same as before) ... */ 
        imageIndexInput.addEventListener('change', () => { 
            let val = parseInt(imageIndexInput.value);
            if (isNaN(val) || val < 0) { imageIndexInput.value = 0; }
            else if (currentMaxIndex >= 0 && val > currentMaxIndex) { 
                imageIndexInput.value = currentMaxIndex; 
            }
        });
    }

    if (analyzeButton) { /* ... (same as before) ... */ 
        analyzeButton.addEventListener('click', async () => {
            const selectedIndex = parseInt(imageIndexInput.value);
            if (isNaN(selectedIndex) || selectedIndex < 0 || (currentMaxIndex >= 0 && selectedIndex > currentMaxIndex) ) {
                alert("Please enter a valid image index within the range 0 to " + (currentMaxIndex >= 0 ? currentMaxIndex : 'N/A') + ".");
                if(imageIndexInput) imageIndexInput.focus(); return;
            }
            resetUIForNewAnalysis(); 
            if(progressSection) progressSection.style.display = 'block';
            setProgress(5, `Requesting analysis for image ${selectedIndex}...`);
            if(analyzeButton) analyzeButton.disabled = true; 
            if(analyzeButtonText) analyzeButtonText.style.display = 'none'; 
            if(analyzeButtonSpinner) analyzeButtonSpinner.style.display = 'inline-block'; 
            let progress = 5;
            const progressInterval = setInterval(() => {
                progress += Math.floor(Math.random() * 5) + 1; 
                if (progress <= 90) { setProgress(progress, `Processing image ${selectedIndex} on server...`);}
                else { setProgress(90, `Finalizing analysis for image ${selectedIndex}...`);}
            }, 300);
            try {
                const response = await fetch('/analyze', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_index: selectedIndex, use_llm: useLlmCheckbox ? useLlmCheckbox.checked : false }),
                });
                clearInterval(progressInterval);
                if (response.ok) {
                    const results = await response.json();
                    setProgress(100, `Analysis complete for image ${results.image_index_analyzed}.`);
                    if(progressBar) { progressBar.classList.remove('bg-danger'); progressBar.classList.add('bg-success'); }
                    updateUIWithResults(results);
                    setTimeout(() => { if(progressSection) progressSection.style.display = 'none'; }, 2000);
                } else {
                    const errorData = await response.json();
                    setProgress(100, `Error: ${errorData.error || response.statusText}`);
                    if(progressBar) { progressBar.classList.remove('bg-success'); progressBar.classList.add('bg-danger'); }
                    console.error('Analysis Error:', errorData);
                }
            } catch (error) {
                clearInterval(progressInterval); console.error('Analysis request failed:', error);
                setProgress(100, 'Request failed. Check console/server logs.');
                if(progressBar) { progressBar.classList.remove('bg-success'); progressBar.classList.add('bg-danger'); }
            } finally {
                if(analyzeButton) analyzeButton.disabled = false; 
                if(analyzeButtonText) analyzeButtonText.style.display = 'inline-block'; 
                if(analyzeButtonSpinner) analyzeButtonSpinner.style.display = 'none'; 
            }
        });
    }

    // Dark Mode Toggle Logic
    if (darkModeToggle) { /* ... (same as before) ... */ 
        const moonIcon = darkModeToggle.querySelector('.fa-moon');
        const sunIcon = darkModeToggle.querySelector('.fa-sun');

        const applyTheme = (theme) => {
            document.body.setAttribute('data-bs-theme', theme);
            if (moonIcon && sunIcon) { 
                if (theme === 'dark') {
                    moonIcon.style.display = 'none';
                    sunIcon.style.display = 'inline-block';
                } else {
                    moonIcon.style.display = 'inline-block';
                    sunIcon.style.display = 'none';
                }
            }
        };
        
        let preferredTheme = localStorage.getItem('theme');
        if (!preferredTheme) {
            preferredTheme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        applyTheme(preferredTheme);

        darkModeToggle.addEventListener('click', () => {
            let currentTheme = document.body.getAttribute('data-bs-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('theme', newTheme);
            applyTheme(newTheme);
        });
    }

    fetchInitialConfig();
    resetUIForNewAnalysis();
});