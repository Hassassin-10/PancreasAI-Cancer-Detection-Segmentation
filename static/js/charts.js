/**
 * charts.js — Chart.js Visualizations
 * ======================================
 * Creates all charts on the results dashboard:
 *   - Probability pie chart (Cancer vs Normal)
 *   - Metrics bar chart (Dice, IoU, Precision, Recall, F1)
 *   - Feature importance horizontal bar chart
 *   - Clinical summary text population
 *
 * Data is passed via window.analysisResults (set in result.html).
 */

document.addEventListener('DOMContentLoaded', () => {
    // Wait for data to be available
    if (typeof window.analysisResults === 'undefined') return;

    const results = window.analysisResults;

    initProbabilityChart(results);
    initMetricsChart(results);
    initFeatureChart(results);
    initClinicalSummary(results);
    animateGauge(results);
});


// ---------------------------------------------------------------------------
// Chart.js Global Configuration
// ---------------------------------------------------------------------------

Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 16;


// ---------------------------------------------------------------------------
// Probability Pie Chart
// ---------------------------------------------------------------------------

function initProbabilityChart(results) {
    const ctx = document.getElementById('probabilityChart');
    if (!ctx) return;

    const isCancer = results.prediction_label === 'Cancer';
    const cancerProb = results.cancer_probability;
    const normalProb = results.normal_probability;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Cancer', 'Normal'],
            datasets: [{
                data: [cancerProb, normalProb],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.85)',
                    'rgba(16, 185, 129, 0.85)'
                ],
                borderColor: [
                    'rgba(239, 68, 68, 1)',
                    'rgba(16, 185, 129, 1)'
                ],
                borderWidth: 2,
                hoverBorderWidth: 3,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { weight: '600' },
                        padding: 20,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: { weight: '700' },
                    callbacks: {
                        label: function(context) {
                            return ` ${context.label}: ${context.parsed.toFixed(2)}%`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration: 1200,
                easing: 'easeOutQuart'
            }
        },
        plugins: [{
            // Center text plugin
            id: 'centerText',
            beforeDraw(chart) {
                const { width, height, ctx } = chart;
                ctx.restore();

                // Main value
                const fontSize = (height / 8).toFixed(2);
                ctx.font = `800 ${fontSize}px Inter, sans-serif`;
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'center';

                const text = `${cancerProb}%`;
                const textX = width / 2;
                const textY = height / 2 - 8;

                // Get theme
                const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
                ctx.fillStyle = isCancer
                    ? '#ef4444'
                    : '#10b981';
                ctx.fillText(text, textX, textY);

                // Label
                const labelSize = (height / 14).toFixed(2);
                ctx.font = `600 ${labelSize}px Inter, sans-serif`;
                ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
                ctx.fillText('Cancer Prob.', textX, textY + parseInt(fontSize) + 4);

                ctx.save();
            }
        }]
    });
}


// ---------------------------------------------------------------------------
// Metrics Bar Chart
// ---------------------------------------------------------------------------

function initMetricsChart(results) {
    const ctx = document.getElementById('metricsChart');
    if (!ctx) return;

    const metrics = results.metrics || {};

    // Select key metrics for the bar chart
    const labels = ['Dice', 'IoU', 'Precision', 'Recall', 'F1', 'Sensitivity', 'Specificity'];
    const keys = ['dice_score', 'iou', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity'];
    const values = keys.map(k => metrics[k] || 0);

    // Color based on value: green > 0.8, yellow 0.5-0.8, red < 0.5
    const colors = values.map(v => {
        if (v >= 0.8) return 'rgba(16, 185, 129, 0.8)';
        if (v >= 0.5) return 'rgba(245, 158, 11, 0.8)';
        return 'rgba(239, 68, 68, 0.8)';
    });

    const borderColors = values.map(v => {
        if (v >= 0.8) return 'rgba(16, 185, 129, 1)';
        if (v >= 0.5) return 'rgba(245, 158, 11, 1)';
        return 'rgba(239, 68, 68, 1)';
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Score',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: v => v.toFixed(1),
                        font: { weight: '500' }
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        font: { weight: '600' }
                    },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: { weight: '700' },
                    callbacks: {
                        label: function(context) {
                            return ` ${context.dataset.label}: ${context.parsed.y.toFixed(4)}`;
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}


// ---------------------------------------------------------------------------
// Feature Importance Chart
// ---------------------------------------------------------------------------

function initFeatureChart(results) {
    const ctx = document.getElementById('featureChart');
    if (!ctx) return;

    // Simulated feature importance data (in real deployment, this would come from the model)
    const features = [
        { name: 'Texture Pattern', value: 0.92 },
        { name: 'Edge Density', value: 0.87 },
        { name: 'Intensity Variance', value: 0.81 },
        { name: 'Shape Regularity', value: 0.74 },
        { name: 'Spatial Context', value: 0.68 },
        { name: 'Boundary Sharpness', value: 0.63 },
        { name: 'Region Homogeneity', value: 0.55 },
        { name: 'Size Ratio', value: 0.48 },
    ];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features.map(f => f.name),
            datasets: [{
                label: 'Importance',
                data: features.map(f => f.value),
                backgroundColor: features.map((_, i) => {
                    const alpha = 0.9 - (i * 0.08);
                    return `rgba(59, 130, 246, ${alpha})`;
                }),
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: v => v.toFixed(1),
                        font: { weight: '500' }
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },
                y: {
                    ticks: {
                        font: { size: 11, weight: '500' }
                    },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return ` Importance: ${(context.parsed.x * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            animation: {
                duration: 1200,
                easing: 'easeOutQuart'
            }
        }
    });
}


// ---------------------------------------------------------------------------
// Confidence Gauge Animation
// ---------------------------------------------------------------------------

function animateGauge(results) {
    const gaugeValue = document.getElementById('gaugeValue');
    if (!gaugeValue) return;

    const targetValue = results.confidence;
    let current = 0;

    const interval = setInterval(() => {
        current += 1;
        if (current >= targetValue) {
            current = targetValue;
            clearInterval(interval);
        }
        gaugeValue.textContent = current.toFixed(1) + '%';
    }, 15);
}


// ---------------------------------------------------------------------------
// Clinical Summary
// ---------------------------------------------------------------------------

function initClinicalSummary(results) {
    const container = document.getElementById('clinicalSummary');
    if (!container) return;

    const isCancer = results.prediction_label === 'Cancer';

    let summary = '';

    if (isCancer) {
        summary = `<span class="summary-heading">AI-ASSISTED ANALYSIS SUMMARY</span>
The deep learning model has identified findings consistent with pancreatic malignancy with a confidence of ${results.confidence}%. The detected lesion is located in the ${results.tumor_location} region, with an estimated area of ${results.tumor_area} cm² and an approximate volume of ${results.tumor_volume} cc.

<span class="summary-heading">STAGING</span>
${results.stage_suggestion}

<span class="summary-heading">RISK ASSESSMENT</span>
${results.risk_level}

<span class="summary-heading">MODEL DETAILS</span>
The segmentation analysis reveals a focal lesion identified by the Attention UNet (PaNSegNet-inspired) architecture. The Grad-CAM attention map highlights the regions most influential in the model's decision. Evaluation metrics are computed following PaNSegNet and PanTS benchmark protocols.

<span class="summary-heading">RECOMMENDATION</span>
Correlation with clinical history, laboratory markers (CA 19-9), and contrast-enhanced CT/MRI is strongly recommended. Referral to a hepatopancreatobiliary specialist is advised for further workup.`;
    } else {
        summary = `<span class="summary-heading">AI-ASSISTED ANALYSIS SUMMARY</span>
The deep learning model has classified this scan as NORMAL with a confidence of ${results.confidence}%. No significant pancreatic lesion was detected in the analyzed image.

<span class="summary-heading">RISK ASSESSMENT</span>
${results.risk_level}

<span class="summary-heading">OBSERVATION</span>
The segmentation analysis did not identify any focal mass lesion in the pancreatic parenchyma. The pancreatic morphology appears within normal limits based on the AI analysis.

<span class="summary-heading">RECOMMENDATION</span>
If clinical suspicion remains high, correlation with clinical presentation and laboratory markers is recommended. Consider follow-up imaging if symptoms persist.`;
    }

    container.innerHTML = summary;
}
