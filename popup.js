// PhishGuard Popup Script

document.addEventListener('DOMContentLoaded', () => {

    chrome.runtime.sendMessage({ type: 'GET_RESULT' }, (response) => {

    const loadingEl = document.getElementById('loading');
    const resultEl = document.getElementById('result');
    const noDataEl = document.getElementById('no-data');
    const footerEl = document.getElementById('footer');

    // ── Loading State — poll until result arrives ─────────────────────────
    if (response && response.status === 'loading') {
        loadingEl.classList.remove('hidden');
        resultEl.classList.add('hidden');
        noDataEl.classList.add('hidden');
        footerEl.classList.add('hidden');

        // Poll every 2 seconds, up to 30 attempts (~60 seconds max)
        // This covers Render free-tier cold starts which can take 30-60s
        let attempts = 0;
        const maxAttempts = 30;
        const pollInterval = setInterval(() => {
            attempts++;
            chrome.runtime.sendMessage({ type: 'GET_RESULT' }, (retryResponse) => {
                if (chrome.runtime.lastError) {
                    clearInterval(pollInterval);
                    loadingEl.classList.add('hidden');
                    noDataEl.classList.remove('hidden');
                    return;
                }
                if (retryResponse && retryResponse.result) {
                    clearInterval(pollInterval);
                    location.reload();
                } else if (attempts >= maxAttempts) {
                    clearInterval(pollInterval);
                    loadingEl.classList.add('hidden');
                    noDataEl.classList.remove('hidden');
                }
            });
        }, 2000);
        return;
    }

    // ── No Data State ────────────────────────────────────────────────────
    loadingEl.classList.add('hidden');

    if (!response || !response.result) {
        noDataEl.classList.remove('hidden');
        footerEl.classList.add('hidden');
        return;
    }

    // ── Result State ─────────────────────────────────────────────────────
    const data = response.result;
    resultEl.classList.remove('hidden');
    footerEl.classList.remove('hidden');

    // ── Scanned URL ──────────────────────────────────────────────────
    const scannedUrlEl = document.getElementById('scanned-url');
    try {
        const urlObj = new URL(data.url);
        scannedUrlEl.textContent = urlObj.hostname;
    } catch {
        scannedUrlEl.textContent = data.url || 'Unknown';
    }

    // ── Threat Indicator ─────────────────────────────────────────────
    const threatIndicator = document.getElementById('threat-indicator');
    const threatIcon = document.getElementById('threat-icon');
    const threatLabel = document.getElementById('threat-label');
    const threatSubtitle = document.getElementById('threat-subtitle');
    const threatMessage = document.getElementById('threat-message');

    const threatConfig = {
        'Low Risk': {
            icon: '✅',
            class: 'threat-safe',
            subtitle: 'This site appears safe.',
            message: ''
        },
        'Suspicious': {
            icon: '⚠️',
            class: 'threat-suspicious',
            subtitle: 'Some suspicious traits detected.',
            message: 'This site has characteristics sometimes associated with phishing. Proceed with caution if you were not expecting to visit this page.'
        },
        'High Risk': {
            icon: '🚨',
            class: 'threat-danger',
            subtitle: 'Strong phishing indicators found.',
            message: 'This site displays multiple strong indicators of a phishing attack. We strongly recommend leaving this page and not entering any personal information.'
        }
    };

    const config = threatConfig[data.finalThreatLevel] ||
        { icon: '❓', class: 'threat-unknown', subtitle: 'Unable to determine risk.', message: '' };

    threatIndicator.className = `threat-indicator ${config.class}`;
    threatIcon.textContent = config.icon;
    threatLabel.textContent = data.finalThreatLevel;
    threatSubtitle.textContent = config.subtitle;

    // Show contextual message for Suspicious and High Risk only
    if (config.message) {
        threatMessage.textContent = config.message;
        threatMessage.classList.remove('hidden');
        // Add color-specific class for the left border
        if (data.finalThreatLevel === 'High Risk') {
            threatMessage.classList.add('msg-danger');
        } else {
            threatMessage.classList.add('msg-suspicious');
        }
    } else {
        threatMessage.classList.add('hidden');
    }

    // ── Heuristic Score Card ─────────────────────────────────────────
    const heuristicScoreEl = document.getElementById('heuristic-score');
    const heuristicBar = document.getElementById('heuristic-bar');

    heuristicScoreEl.textContent = data.heuristicScore;

    const heuristicPct = Math.min((data.heuristicScore / 105) * 100, 100);
    // Use requestAnimationFrame to ensure the bar animates after render
    requestAnimationFrame(() => {
        heuristicBar.style.width = `${heuristicPct}%`;
    });

    if (data.heuristicScore <= 39) {
        heuristicBar.style.background = 'linear-gradient(90deg, #2ecc71, #27ae60)';
    } else if (data.heuristicScore <= 60) {
        heuristicBar.style.background = 'linear-gradient(90deg, #f39c12, #e67e22)';
    } else {
        heuristicBar.style.background = 'linear-gradient(90deg, #e74c3c, #c0392b)';
    }

    // ── ML Model Card ────────────────────────────────────────────────
    const mlStatusEl = document.getElementById('ml-status');
    const mlBarTrack = document.getElementById('ml-bar-track');
    const mlBar = document.getElementById('ml-bar');

    if (data.heuristicScore < 40) {
        // ML was skipped for low-risk heuristic scores
        mlStatusEl.textContent = 'Skipped';
        mlStatusEl.style.color = '#5a5a76';
        mlBarTrack.classList.add('hidden');
    } else if (data.mlLabel === 'unknown' || data.mlLabel === 'error') {
        mlStatusEl.textContent = 'Unavailable';
        mlStatusEl.style.color = '#5a5a76';
        mlBarTrack.classList.add('hidden');
    } else {
        // Show ML probability bar
        mlBarTrack.classList.remove('hidden');
        const mlPct = Math.min(data.mlPhishingProbability * 100, 100);
        requestAnimationFrame(() => {
            mlBar.style.width = `${mlPct}%`;
        });

        if (data.mlPhishingProbability < 0.5) {
            mlStatusEl.textContent = 'Likely Safe';
            mlStatusEl.style.color = '#2ecc71';
            mlBar.style.background = 'linear-gradient(90deg, #2ecc71, #27ae60)';
        } else if (data.mlPhishingProbability < 0.85) {
            mlStatusEl.textContent = 'Uncertain';
            mlStatusEl.style.color = '#f39c12';
            mlBar.style.background = 'linear-gradient(90deg, #f39c12, #e67e22)';
        } else {
            mlStatusEl.textContent = 'Likely Phishing';
            mlStatusEl.style.color = '#e74c3c';
            mlBar.style.background = 'linear-gradient(90deg, #e74c3c, #c0392b)';
        }
    }

    // ── Triggered Rules ──────────────────────────────────────────────
    const rulesSection = document.getElementById('rules-section');
    const rulesList = document.getElementById('rules-list');
    const rulesCount = document.getElementById('rules-count');

    if (data.triggeredRules && data.triggeredRules.length > 0) {
        rulesSection.classList.remove('hidden');
        rulesCount.textContent = data.triggeredRules.length;
        rulesList.innerHTML = '';
        data.triggeredRules.forEach(rule => {
            const li = document.createElement('li');
            li.textContent = rule;
            rulesList.appendChild(li);
        });
    } else {
        rulesSection.classList.add('hidden');
    }

    // ── Footer Timestamp ─────────────────────────────────────────────
    const scanTimeEl = document.getElementById('scan-time');
    if (data.timestamp) {
        const scanDate = new Date(data.timestamp);
        const now = new Date();
        const diffSec = Math.floor((now - scanDate) / 1000);

        if (diffSec < 5) {
            scanTimeEl.textContent = 'Scanned just now';
        } else if (diffSec < 60) {
            scanTimeEl.textContent = `Scanned ${diffSec}s ago`;
        } else if (diffSec < 3600) {
            const mins = Math.floor(diffSec / 60);
            scanTimeEl.textContent = `Scanned ${mins}m ago`;
        } else {
            scanTimeEl.textContent = `Scanned at ${scanDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        }
    } else {
        scanTimeEl.textContent = 'Scanned just now';
    }

    });
});