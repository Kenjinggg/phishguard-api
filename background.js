// PhishGuard Background Service Worker
// Manages state between content script and popup, handles API calls

const API_URL = 'https://phishguard-api-jlzn.onrender.com/predict';
const API_KEY = 'phishguard-api-key-2025';

const tabResults = {};
const tabStatus = {}; // tracks whether analysis is in progress
const urlCache = {};  // #4.4: URL-based result caching

// ─── Combined Scoring Logic (#2.3 — moved from content.js with fixes) ────────
function computeFinalThreatLevel(heuristicLevel, heuristicScore, phishingProbability) {
    // Heuristic scores below 40 are always Low Risk, ML is not consulted
    if (heuristicScore < 40) {
        return 'Low Risk';
    }
    // High Risk from heuristics is always immutable
    if (heuristicLevel === 'High Risk') {
        return 'High Risk';
    }
    // ML upgrades Suspicious → High Risk at extreme confidence (>= 0.95)
    if (heuristicLevel === 'Suspicious' && phishingProbability >= 0.95) {
        return 'High Risk';
    }
    return heuristicLevel;
}

// ─── Cache Cleanup (#4.4) ─────────────────────────────────────────────────────
function cleanupCache() {
    const now = Date.now();
    const maxAge = 600000; // 10 minutes
    for (const url of Object.keys(urlCache)) {
        if (now - urlCache[url].timestamp > maxAge) {
            delete urlCache[url];
        }
    }
}

// Listen for messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    // #1.1 / #6.1: Handle analysis request from content script
    if (message.type === 'ANALYSE_URL' && sender.tab) { // #2.1: null-check on sender.tab
        const tabId = sender.tab.id;
        const { url, allFeatures, heuristicResult } = message.data;

        // #4.4: Check cache first (5-minute TTL)
        const cached = urlCache[url];
        if (cached && (Date.now() - cached.timestamp < 300000)) {
            tabResults[tabId] = cached.result;
            tabStatus[tabId] = 'complete';
            updateBadge(tabId, cached.result.finalThreatLevel);
            sendResponse({ status: 'ok' });
            return;
        }

        // Call ML API with timeout to prevent indefinite hanging
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s for Render cold starts

        fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify(allFeatures),
            signal: controller.signal
        })
            .then(response => {
                clearTimeout(timeoutId)
                if (response.ok) return response.json();
                throw new Error(`API returned ${response.status}`);
            })
            .then(mlResult => {
                const finalThreatLevel = computeFinalThreatLevel(
                    heuristicResult.heuristicLevel,
                    heuristicResult.score,
                    mlResult.phishing_probability
                );

                const analysisResult = {
                    url: url,
                    finalThreatLevel: finalThreatLevel,
                    heuristicScore: heuristicResult.score,
                    heuristicLevel: heuristicResult.heuristicLevel,
                    triggeredRules: heuristicResult.triggeredRules,
                    mlPhishingProbability: mlResult.phishing_probability,
                    mlLabel: mlResult.label,
                    timestamp: new Date().toISOString()
                };

                tabResults[tabId] = analysisResult;
                tabStatus[tabId] = 'complete';
                updateBadge(tabId, finalThreatLevel);

                // #4.4: Cache the result
                urlCache[url] = { result: analysisResult, timestamp: Date.now() };

                sendResponse({ status: 'ok' });
            })
            .catch(apiError => {
                clearTimeout(timeoutId);
                console.warn('PhishGuard: ML API unavailable, using heuristic only.', apiError);

                // Fallback: use heuristic-only result
                const finalThreatLevel = computeFinalThreatLevel(
                    heuristicResult.heuristicLevel,
                    heuristicResult.score,
                    0 // no ML probability available
                );

                const analysisResult = {
                    url: url,
                    finalThreatLevel: finalThreatLevel,
                    heuristicScore: heuristicResult.score,
                    heuristicLevel: heuristicResult.heuristicLevel,
                    triggeredRules: heuristicResult.triggeredRules,
                    mlPhishingProbability: 0,
                    mlLabel: 'unknown',
                    timestamp: new Date().toISOString()
                };

                tabResults[tabId] = analysisResult;
                tabStatus[tabId] = 'complete';
                updateBadge(tabId, finalThreatLevel);

                // Cache even heuristic-only results to avoid repeated failed API calls
                urlCache[url] = { result: analysisResult, timestamp: Date.now() };

                sendResponse({ status: 'ok' });
            });

        return true; // Keep message channel open for async sendResponse
    }

    if (message.type === 'ANALYSIS_STARTED' && sender.tab) { // #2.1: null-check on sender.tab
        const tabId = sender.tab.id;
        tabStatus[tabId] = 'loading';
        sendResponse({ status: 'ok' });
    }

    // Handle content script feature extraction failures
    if (message.type === 'ANALYSIS_FAILED' && sender.tab) {
        const tabId = sender.tab.id;
        console.warn('PhishGuard: Content script analysis failed for', message.data?.url);

        // Clear loading status with a safe fallback so popup doesn't stay stuck
        tabResults[tabId] = {
            url: message.data?.url || 'unknown',
            finalThreatLevel: 'Low Risk',
            heuristicScore: 0,
            heuristicLevel: 'Low Risk',
            triggeredRules: [],
            mlPhishingProbability: 0,
            mlLabel: 'error',
            timestamp: new Date().toISOString()
        };
        tabStatus[tabId] = 'complete';
        updateBadge(tabId, 'Low Risk');
        sendResponse({ status: 'ok' });
    }

    if (message.type === 'GET_RESULT') {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            // #2.2: null-check on tabs[0]
            if (!tabs || tabs.length === 0) {
                sendResponse({ result: null, status: 'idle' });
                return;
            }
            const tabId = tabs[0].id;
            const result = tabResults[tabId] || null;
            const status = tabStatus[tabId] || 'idle';
            sendResponse({ result: result, status: status });
        });
        return true;
    }
});

// Clean up when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
    delete tabResults[tabId];
    delete tabStatus[tabId];
    // #4.4: Run cache cleanup on tab close
    cleanupCache();
});

// Clean up when tab navigates to a new page
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
    if (changeInfo.status === 'loading') {
        delete tabResults[tabId];
        tabStatus[tabId] = 'loading';
        chrome.action.setBadgeText({ text: '...', tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: '#7c83fd', tabId: tabId });
    }
});

// Update the extension icon badge
function updateBadge(tabId, threatLevel) {
    const badgeConfig = {
        'Low Risk': { text: 'SAFE', color: '#2ecc71' },
        'Suspicious': { text: '!', color: '#f39c12' },
        'High Risk': { text: '!!', color: '#e74c3c' }
    };

    const config = badgeConfig[threatLevel] || { text: '?', color: '#95a5a6' };
    chrome.action.setBadgeText({ text: config.text, tabId: tabId });
    chrome.action.setBadgeBackgroundColor({ color: config.color, tabId: tabId });
}