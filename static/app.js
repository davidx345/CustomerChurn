// Modern JavaScript for enhanced frontend functionality
// [Intent] Clean, modular JavaScript with modern ES6+ features
// [UX] Smooth interactions and real-time feedback

class ChurnPredictionApp {
    constructor() {
        this.apiBase = '';
        this.charts = {};
        this.predictionHistory = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupFormValidation();
        await this.loadInitialData();
        this.updateUI();
        this.startHealthMonitoring();
    }

    setupEventListeners() {
        // Form submissions
        document.getElementById('churnForm')?.addEventListener('submit', this.handlePrediction.bind(this));
        document.getElementById('batchChurnForm')?.addEventListener('submit', this.handleBatchPrediction.bind(this));
        
        // Clear history
        document.getElementById('clearHistory')?.addEventListener('click', this.clearHistory.bind(this));
        
        // History item clicks
        document.addEventListener('click', (e) => {
            if (e.target.closest('.history-item')) {
                this.loadHistoryItem(e.target.closest('.history-item'));
            }
        });

        // Checkbox handling for form
        const checkboxes = document.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                e.target.value = e.target.checked ? '1' : '0';
            });
        });
    }

    setupFormValidation() {
        const forms = document.querySelectorAll('.needs-validation');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!form.checkValidity()) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                form.classList.add('was-validated');
            });
        });
    }

    async loadInitialData() {
        try {
            // Load model info
            await this.loadModelInfo();
            
            // Load feature importance
            await this.loadFeatureImportance();
            
            // Load performance metrics
            await this.loadPerformanceMetrics();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/api/model_info');
            const data = await response.json();
            
            document.getElementById('modelType').textContent = data.model_type || 'Unknown';
            document.getElementById('modelVersion').textContent = data.metadata?.version || 'Unknown';
            document.getElementById('featureCount').textContent = data.feature_count || 'Unknown';
            
        } catch (error) {
            console.error('Failed to load model info:', error);
        }
    }

    async loadFeatureImportance() {
        try {
            const response = await fetch('/api/feature_importance');
            const data = await response.json();
            
            if (data.features && data.importances) {
                this.createFeatureImportanceChart(data.features, data.importances);
            }
            
        } catch (error) {
            console.error('Failed to load feature importance:', error);
        }
    }

    async loadPerformanceMetrics() {
        try {
            const response = await fetch('/api/model_performance');
            const data = await response.json();
            
            if (data.accuracy) {
                document.getElementById('accuracyMetric').textContent = (data.accuracy * 100).toFixed(1) + '%';
                document.getElementById('modelAccuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
            }
            
            if (data.roc_auc) {
                document.getElementById('rocAucMetric').textContent = data.roc_auc.toFixed(3);
            }
            
            if (data.confusion_matrix) {
                const cm = data.confusion_matrix;
                document.getElementById('tnValue').textContent = cm[0][0];
                document.getElementById('fpValue').textContent = cm[0][1];
                document.getElementById('fnValue').textContent = cm[1][0];
                document.getElementById('tpValue').textContent = cm[1][1];
            }
            
        } catch (error) {
            console.error('Failed to load performance metrics:', error);
        }
    }

    async handlePrediction(event) {
        event.preventDefault();
        
        const form = event.target;
        if (!form.checkValidity()) return;

        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        
        try {
            // Show loading state
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Predicting...';
            submitBtn.disabled = true;
            
            // Collect form data
            const formData = new FormData(form);
            const data = {};
            
            for (let [key, value] of formData.entries()) {
                if (['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'].includes(key)) {
                    data[key] = parseInt(value);
                } else if (['Balance', 'EstimatedSalary'].includes(key)) {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            }

            // Handle checkboxes
            data.HasCrCard = document.getElementById('hasCrCard').checked ? 1 : 0;
            data.IsActiveMember = document.getElementById('isActiveMember').checked ? 1 : 0;

            // Make prediction
            const startTime = Date.now();
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const responseTime = Date.now() - startTime;

            // Update response time metric
            document.getElementById('responseTime').textContent = responseTime;
            document.getElementById('avgResponseTime').textContent = responseTime + 'ms';

            if (response.ok) {
                this.displayPredictionResult(result);
                this.addToHistory(data, result);
                this.updatePredictionCount();
            } else {
                this.displayError(result.error || 'Prediction failed');
            }

        } catch (error) {
            this.displayError('Network error: ' + error.message);
        } finally {
            // Reset button
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    }

    async handleBatchPrediction(event) {
        event.preventDefault();
        
        const form = event.target;
        const fileInput = document.getElementById('csvFile');
        const resultDiv = document.getElementById('batchResult');
        
        if (!fileInput.files[0]) {
            this.showAlert('Please select a CSV file', 'warning');
            return;
        }

        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        
        try {
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Processing...';
            submitBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('csvFile', fileInput.files[0]);
            
            const response = await fetch('/batch_predict', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                
                resultDiv.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>Batch prediction completed!
                        <a href="${url}" download="batch_predictions.csv" class="btn btn-sm btn-success ms-2">
                            <i class="fas fa-download me-1"></i>Download Results
                        </a>
                    </div>
                `;
                resultDiv.style.display = 'block';
                
                // Auto-hide after 10 seconds
                setTimeout(() => {
                    resultDiv.style.display = 'none';
                    URL.revokeObjectURL(url);
                }, 10000);
                
            } else {
                const error = await response.json();
                this.showAlert('Batch prediction failed: ' + error.error, 'danger');
            }
            
        } catch (error) {
            this.showAlert('Network error: ' + error.message, 'danger');
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    }

    displayPredictionResult(result) {
        const resultDiv = document.getElementById('predictionResult');
        const shapDiv = document.getElementById('shapExplanation');
        
        const isChurn = result.prediction === 1;
        const probability = (result.probability * 100).toFixed(1);
        
        const resultClass = isChurn ? 'danger' : 'success';
        const resultIcon = isChurn ? 'fa-exclamation-triangle' : 'fa-check-circle';
        const resultText = isChurn ? 'High Churn Risk' : 'Low Churn Risk';
        
        let badgesHtml = '';
        if (result.transparency) {
            badgesHtml += `<span class="badge bg-info ms-2">${result.transparency}</span>`;
        }
        if (result.harmfulness_flag) {
            badgesHtml += `<span class="badge bg-warning ms-2">Overconfident Prediction</span>`;
        }
        if (result.bias_flags && Object.keys(result.bias_flags).length > 0) {
            badgesHtml += `<span class="badge bg-danger ms-2">Bias Detected</span>`;
        }

        resultDiv.className = `prediction-result ${resultClass}`;
        resultDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas ${resultIcon} fa-2x me-3"></i>
                <div>
                    <h4 class="mb-1">${resultText}</h4>
                    <p class="mb-1">Probability: ${probability}%</p>
                    ${badgesHtml}
                </div>
            </div>
        `;
        resultDiv.style.display = 'block';

        // Show feature importance if available
        if (result.feature_importance && !result.feature_importance.explainability_error) {
            this.createFeatureExplanationChart(result.feature_importance);
            shapDiv.style.display = 'block';
        } else {
            shapDiv.style.display = 'none';
        }
    }

    displayError(message) {
        const resultDiv = document.getElementById('predictionResult');
        resultDiv.className = 'prediction-result danger';
        resultDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-times-circle fa-2x me-3"></i>
                <div>
                    <h4 class="mb-1">Prediction Failed</h4>
                    <p class="mb-0">${message}</p>
                </div>
            </div>
        `;
        resultDiv.style.display = 'block';
    }

    createFeatureImportanceChart(features, importances) {
        const ctx = document.getElementById('globalFeatureChart');
        if (!ctx) return;

        if (this.charts.featureImportance) {
            this.charts.featureImportance.destroy();
        }

        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features.slice(0, 10), // Top 10 features
                datasets: [{
                    label: 'Importance',
                    data: importances.slice(0, 10),
                    backgroundColor: 'rgba(37, 99, 235, 0.7)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Top 10 Most Important Features'
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    }
                }
            }
        });
    }

    createFeatureExplanationChart(featureImportance) {
        const ctx = document.getElementById('shapChart');
        if (!ctx) return;

        if (this.charts.shap) {
            this.charts.shap.destroy();
        }

        const features = Object.keys(featureImportance);
        const values = Object.values(featureImportance);

        this.charts.shap = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Feature Impact',
                    data: values,
                    backgroundColor: values.map(val => 
                        val > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)'
                    ),
                    borderColor: values.map(val => 
                        val > 0 ? 'rgba(239, 68, 68, 1)' : 'rgba(16, 185, 129, 1)'
                    ),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Feature Impact on This Prediction'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Impact Score (positive = increases churn risk)'
                        }
                    }
                }
            }
        });
    }

    addToHistory(inputData, result) {
        const historyItem = {
            timestamp: new Date().toISOString(),
            input: inputData,
            result: {
                prediction: result.prediction,
                probability: result.probability,
                harmfulness_flag: result.harmfulness_flag,
                bias_flags: result.bias_flags
            }
        };

        this.predictionHistory.unshift(historyItem);
        if (this.predictionHistory.length > 10) {
            this.predictionHistory = this.predictionHistory.slice(0, 10);
        }

        localStorage.setItem('predictionHistory', JSON.stringify(this.predictionHistory));
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const historyContainer = document.getElementById('predictionHistory');
        const clearBtn = document.getElementById('clearHistory');
        
        if (this.predictionHistory.length === 0) {
            historyContainer.innerHTML = '<p class="text-muted text-center">No predictions yet</p>';
            clearBtn.style.display = 'none';
            return;
        }

        clearBtn.style.display = 'block';
        
        const historyHtml = this.predictionHistory.map((item, index) => {
            const isChurn = item.result.prediction === 1;
            const resultClass = isChurn ? 'churn' : 'no-churn';
            const resultText = isChurn ? 'High Risk' : 'Low Risk';
            const probability = (item.result.probability * 100).toFixed(1);
            
            return `
                <div class="history-item" data-index="${index}">
                    <div class="history-time">${new Date(item.timestamp).toLocaleString()}</div>
                    <div class="history-result ${resultClass}">${resultText} (${probability}%)</div>
                    <div class="text-muted small">
                        Age: ${item.input.Age}, Balance: $${item.input.Balance?.toLocaleString()}
                    </div>
                </div>
            `;
        }).join('');
        
        historyContainer.innerHTML = historyHtml;
    }

    loadHistoryItem(element) {
        const index = parseInt(element.dataset.index);
        const item = this.predictionHistory[index];
        if (!item) return;

        const form = document.getElementById('churnForm');
        const inputs = item.input;

        // Populate form fields
        Object.keys(inputs).forEach(key => {
            const element = form.elements[key];
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = inputs[key] === 1;
                } else {
                    element.value = inputs[key];
                }
            }
        });

        // Scroll to form
        form.scrollIntoView({ behavior: 'smooth' });
        
        // Reset validation state
        form.classList.remove('was-validated');
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear the prediction history?')) {
            this.predictionHistory = [];
            localStorage.removeItem('predictionHistory');
            this.updateHistoryDisplay();
        }
    }

    updatePredictionCount() {
        const current = parseInt(document.getElementById('totalPredictions').textContent) || 0;
        document.getElementById('totalPredictions').textContent = current + 1;
        
        const today = parseInt(document.getElementById('todayPredictions').textContent) || 0;
        document.getElementById('todayPredictions').textContent = today + 1;
    }

    async startHealthMonitoring() {
        const checkHealth = async () => {
            try {
                const startTime = Date.now();
                const response = await fetch('/health');
                const responseTime = Date.now() - startTime;
                
                const healthStatus = document.getElementById('healthStatus');
                
                if (response.ok) {
                    healthStatus.innerHTML = '<span class="status-badge status-healthy"><i class="fas fa-check-circle me-1"></i>Healthy</span>';
                    document.getElementById('responseTime').textContent = responseTime;
                } else {
                    healthStatus.innerHTML = '<span class="status-badge status-unhealthy"><i class="fas fa-times-circle me-1"></i>Unhealthy</span>';
                }
            } catch (error) {
                const healthStatus = document.getElementById('healthStatus');
                healthStatus.innerHTML = '<span class="status-badge status-unhealthy"><i class="fas fa-times-circle me-1"></i>Offline</span>';
            }
        };

        // Check health immediately and then every 30 seconds
        checkHealth();
        setInterval(checkHealth, 30000);
    }

    updateUI() {
        this.updateHistoryDisplay();
    }

    showAlert(message, type = 'info') {
        // Create temporary alert
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.churnApp = new ChurnPredictionApp();
});

// Service Worker registration for PWA capabilities (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => console.log('SW registered'))
            .catch(error => console.log('SW registration failed'));
    });
}
