// PhishGuard Content Script
// Runs automatically on every page load

// ─── Top domains for URLSimilarityIndex ──────────────────────────────────────
const TOP_DOMAINS = [
    'google', 'facebook', 'youtube', 'amazon', 'twitter', 'instagram',
    'linkedin', 'microsoft', 'apple', 'paypal', 'netflix', 'whatsapp',
    'tiktok', 'reddit', 'wikipedia', 'yahoo', 'ebay', 'dropbox',
    'spotify', 'github', 'adobe', 'bankofamerica', 'chase', 'wellsfargo',
    'citibank', 'hsbc', 'maybank', 'cimbbank', 'publicbank', 'rhbbank',
    'steam', 'epicgames', 'roblox', 'twitch', 'discord', 'telegram',
    'zoom', 'skype', 'outlook', 'gmail', 'icloud', 'aliexpress',
    'shopee', 'lazada', 'grab', 'airbnb', 'booking', 'expedia'
];

// ─── Warning Interstitial (Feature A) ─────────────────────────────────────────
chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'SHOW_WARNING') {
        injectWarningOverlay(message.data);
    }
});

function injectWarningOverlay(analysisResult) {
    // Avoid double-injection if somehow triggered twice
    if (document.getElementById('phishguard-overlay-host')) return;

    const host = document.createElement('div');
    host.id = 'phishguard-overlay-host';
    host.style.cssText = 'position:fixed;top:0;left:0;width:0;height:0;z-index:2147483647;';
    document.documentElement.appendChild(host);

    const shadow = host.attachShadow({ mode: 'closed' });

    const style = document.createElement('style');
    style.textContent = `
        .pg-overlay {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: #1a0d0d; color: #ffffff;
            display: flex; align-items: center; justify-content: center;
            font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
            z-index: 2147483647;
        }
        .pg-card {
            max-width: 480px; width: 90%; background: #2a1414;
            border: 1px solid #e74c3c; border-radius: 12px;
            padding: 32px; text-align: center;
        }
        .pg-icon { font-size: 48px; margin-bottom: 12px; }
        .pg-title { font-size: 22px; font-weight: 700; margin: 0 0 12px; }
        .pg-desc { font-size: 14px; line-height: 1.5; color: #d9c2c2; margin: 0 0 24px; }
        .pg-buttons { display: flex; gap: 12px; justify-content: center; }
        .pg-btn { padding: 10px 20px; border-radius: 8px; font-size: 14px;
            font-weight: 600; cursor: pointer; border: none; }
        .pg-btn-safe { background: #ffffff; color: #1a0d0d; }
        .pg-btn-risk { background: transparent; color: #d9c2c2; border: 1px solid #5a3a3a; }
    `;

    const wrapper = document.createElement('div');
    wrapper.className = 'pg-overlay';
    wrapper.innerHTML = `
        <div class="pg-card">
            <div class="pg-icon">🚨</div>
            <div class="pg-title">PhishGuard Warning</div>
            <p class="pg-desc">This site shows strong indicators of a phishing attack
            (score: ${analysisResult.heuristicScore}). We recommend leaving this page
            and not entering any personal information.</p>
            <div class="pg-buttons">
                <button class="pg-btn pg-btn-safe" id="pg-go-back">Go back (safe)</button>
                <button class="pg-btn pg-btn-risk" id="pg-proceed">Proceed anyway</button>
            </div>
        </div>
    `;

    shadow.appendChild(style);
    shadow.appendChild(wrapper);

    shadow.getElementById('pg-go-back').addEventListener('click', () => {
        history.back();
    });

    shadow.getElementById('pg-proceed').addEventListener('click', () => {
        host.remove();
    });
}

// ─── String Similarity (Levenshtein) ─────────────────────────────────────────
function computeSimilarity(str1, str2) {
    if (str1 === str2) return 1.0;
    if (str1.length === 0 || str2.length === 0) return 0.0;

    const len1 = str1.length;
    const len2 = str2.length;
    const matrix = Array.from({ length: len1 + 1 }, (_, i) =>
        Array.from({ length: len2 + 1 }, (_, j) =>
            i === 0 ? j : j === 0 ? i : 0)
    );

    for (let i = 1; i <= len1; i++) {
        for (let j = 1; j <= len2; j++) {
            if (str1[i - 1] === str2[j - 1]) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = 1 + Math.min(
                    matrix[i - 1][j],
                    matrix[i][j - 1],
                    matrix[i - 1][j - 1]
                );
            }
        }
    }

    const distance = matrix[len1][len2];
    const maxLen = Math.max(len1, len2);
    return 1 - (distance / maxLen);
}

// ─── URL Feature Extraction ───────────────────────────────────────────────────
function extractURLFeatures(url) {
    const urlObj = new URL(url);
    const fullURL = url;
    const domain = urlObj.hostname;
    const protocol = urlObj.protocol;

    const URLLength = fullURL.length;
    const DomainLength = domain.length;

    // IsDomainIP
    const ipPattern = /^(\d{1,3}\.){3}\d{1,3}$/;
    const IsDomainIP = ipPattern.test(domain) ? 1 : 0;

    // TLD
    const domainParts = domain.split('.');
    const tld = domainParts[domainParts.length - 1];
    const TLDLength = tld.length;
    const NoOfSubDomain = Math.max(0, domainParts.length - 2);

    // IsHTTPS
    const IsHTTPS = protocol === 'https:' ? 1 : 0;

    // Character counts
    const NoOfLettersInURL = (fullURL.match(/[a-zA-Z]/g) || []).length;
    const NoOfDegitsInURL = (fullURL.match(/[0-9]/g) || []).length;
    const NoOfEqualsInURL = (fullURL.match(/=/g) || []).length;
    const NoOfQMarkInURL = (fullURL.match(/\?/g) || []).length;
    const NoOfAmpersandInURL = (fullURL.match(/&/g) || []).length;
    const specialChars = (fullURL.match(/[^a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]/g) || []);
    const NoOfOtherSpecialCharsInURL = specialChars.length;

    // Ratios
    const LetterRatioInURL = URLLength > 0 ? NoOfLettersInURL / URLLength : 0;
    const DegitRatioInURL = URLLength > 0 ? NoOfDegitsInURL / URLLength : 0;
    const SpacialCharRatioInURL = URLLength > 0 ?
        NoOfOtherSpecialCharsInURL / URLLength : 0;

    // Obfuscation
    const obfuscatedChars = (fullURL.match(/%[0-9a-fA-F]{2}/g) || []);
    const HasObfuscation = obfuscatedChars.length > 0 ? 1 : 0;
    const NoOfObfuscatedChar = obfuscatedChars.length;
    const ObfuscationRatio = URLLength > 0 ? NoOfObfuscatedChar / URLLength : 0;

    // URLSimilarityIndex — detects typosquatting, NOT exact matches
    const domainBase = domain.replace(/^www\./, '').split('.')[0].toLowerCase();
    let maxSimilarity = 0;
    for (const topDomain of TOP_DOMAINS) {
        // Exact match = legitimate domain, not suspicious
        if (domainBase === topDomain) {
            maxSimilarity = 0.0;
            break;
        }
        const similarity = computeSimilarity(domainBase, topDomain);
        // Only flag near-matches as suspicious (typosquatting range)
        if (similarity >= 0.7 && similarity < 1.0) {
            if (similarity > maxSimilarity) maxSimilarity = similarity;
        }
    }
    const URLSimilarityIndex = maxSimilarity;

    // CharContinuationRate
    let continuations = 0;
    for (let i = 1; i < fullURL.length; i++) {
        if (fullURL[i] === fullURL[i - 1]) continuations++;
    }
    const CharContinuationRate = URLLength > 1 ?
        continuations / (URLLength - 1) : 0;

    // TLDLegitimateProb
    const legitimateTLDs = {
        'com': 0.95, 'org': 0.85, 'edu': 0.95, 'gov': 0.98,
        'net': 0.80, 'io': 0.75, 'co': 0.70, 'uk': 0.85,
        'my': 0.80, 'sg': 0.80, 'au': 0.85, 'de': 0.85,
        'fr': 0.85, 'jp': 0.85, 'cn': 0.60, 'ru': 0.55,
        'info': 0.50, 'biz': 0.45, 'xyz': 0.30, 'top': 0.25,
        'tk': 0.15, 'ml': 0.15, 'ga': 0.15, 'cf': 0.15
    };
    const TLDLegitimateProb = legitimateTLDs[tld.toLowerCase()] || 0.40;

    // URLCharProb
    const alphanumeric = (fullURL.match(/[a-zA-Z0-9]/g) || []).length;
    const URLCharProb = URLLength > 0 ? alphanumeric / URLLength : 0;

    return {
        URLLength, DomainLength, IsDomainIP, URLSimilarityIndex,
        CharContinuationRate, TLDLegitimateProb, URLCharProb, TLDLength,
        NoOfSubDomain, HasObfuscation, NoOfObfuscatedChar, ObfuscationRatio,
        NoOfLettersInURL, LetterRatioInURL, NoOfDegitsInURL, DegitRatioInURL,
        NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL,
        NoOfOtherSpecialCharsInURL, SpacialCharRatioInURL, IsHTTPS,
        _domain: domain // pass domain for reuse in heuristic scoring
    };
}

// ─── DOM Feature Extraction ───────────────────────────────────────────────────
function extractDOMFeatures() {
    const currentDomain = window.location.hostname.replace(/^www\./, '');
    const pageTitle = document.title.toLowerCase();
    const bodyText = document.body ? document.body.innerText.toLowerCase() : '';

    // Instead of reading full HTML, check specific elements for copyright (#4.3)
    const footerHTML = (document.querySelector('footer')?.innerHTML || '').toLowerCase();
    const bodyHTML = document.body ? document.body.innerHTML.substring(0, 50000).toLowerCase() : '';

    // Collect all links first — used by multiple features below
    const allLinks = document.querySelectorAll('a[href]');

    // Title features
    const HasTitle = document.title.length > 0 ? 1 : 0;

    // DomainTitleMatchScore
    const domainBase = currentDomain.split('.')[0].toLowerCase();
    const DomainTitleMatchScore = pageTitle.includes(domainBase) ? 1.0 : 0.0;

    // URLTitleMatchScore
    const urlWords = window.location.href.toLowerCase()
        .replace(/[^a-z0-9]/g, ' ').split(' ')
        .filter(w => w.length > 3);
    let urlTitleMatches = 0;
    for (const word of urlWords) {
        if (pageTitle.includes(word)) urlTitleMatches++;
    }
    const URLTitleMatchScore = urlWords.length > 0 ?
        urlTitleMatches / urlWords.length : 0;

    // HasFavicon
    const faviconLinks = document.querySelectorAll(
        'link[rel="icon"], link[rel="shortcut icon"], link[rel="apple-touch-icon"]'
    );
    const HasFavicon = faviconLinks.length > 0 ? 1 : 0;

    // Robots meta tag
    const robotsMeta = document.querySelector('meta[name="robots"]');
    const Robots = robotsMeta ? 1 : 0;

    // IsResponsive
    const viewportMeta = document.querySelector('meta[name="viewport"]');
    const IsResponsive = viewportMeta ? 1 : 0;

    // HasDescription
    const descriptionMeta = document.querySelector('meta[name="description"]');
    const HasDescription = descriptionMeta ? 1 : 0;

    // NoOfPopup
    const scripts = document.querySelectorAll('script');
    let popupCount = 0;
    scripts.forEach(script => {
        if (script.innerText && script.innerText.includes('window.open')) {
            popupCount++;
        }
    });
    const NoOfPopup = popupCount;

    // NoOfiFrame
    const NoOfiFrame = document.querySelectorAll('iframe').length;

    // HasExternalFormSubmit
    const forms = document.querySelectorAll('form');
    let hasExternalForm = 0;
    forms.forEach(form => {
        const action = form.getAttribute('action') || '';
        if (action.startsWith('http') && !action.includes(currentDomain)) {
            hasExternalForm = 1;
        }
    });
    const HasExternalFormSubmit = hasExternalForm;

    // HasSocialNet — check all links for social media domains
    const socialDomains = [
        'facebook.com', 'twitter.com', 'x.com', 'instagram.com',
        'linkedin.com', 'youtube.com', 'tiktok.com', 'whatsapp.com',
        'telegram.org', 't.me', 'pinterest.com', 'reddit.com'
    ];
    let hasSocial = 0;
    allLinks.forEach(link => {
        const href = (link.getAttribute('href') || '').toLowerCase();
        if (socialDomains.some(s => href.includes(s))) hasSocial = 1;
    });
    const HasSocialNet = hasSocial;

    // HasSubmitButton
    const submitButtons = document.querySelectorAll(
        'input[type="submit"], button[type="submit"], button'
    );
    const HasSubmitButton = submitButtons.length > 0 ? 1 : 0;

    // HasHiddenFields
    const hiddenFields = document.querySelectorAll('input[type="hidden"]');
    const HasHiddenFields = hiddenFields.length > 0 ? 1 : 0;

    // HasPasswordField
    const passwordFields = document.querySelectorAll('input[type="password"]');
    const HasPasswordField = passwordFields.length > 0 ? 1 : 0;

    // Keyword detection
    const bankKeywords = ['bank', 'banking', 'account', 'balance', 'transfer',
        'deposit', 'withdraw', 'loan', 'mortgage'];
    const payKeywords = ['payment', 'pay now', 'checkout', 'billing',
        'invoice', 'transaction', 'credit card', 'debit card'];
    const cryptoKeywords = ['bitcoin', 'ethereum', 'crypto', 'wallet',
        'blockchain', 'nft', 'binance', 'coinbase'];

    const Bank = bankKeywords.some(k => bodyText.includes(k)) ? 1 : 0;
    const Pay = payKeywords.some(k => bodyText.includes(k)) ? 1 : 0;
    const Crypto = cryptoKeywords.some(k => bodyText.includes(k)) ? 1 : 0;

    // HasCopyrightInfo — uses targeted search instead of full HTML (#4.3)
    const HasCopyrightInfo = (
        bodyText.includes('©') ||
        bodyText.includes('copyright') ||
        bodyText.includes('all rights reserved') ||
        bodyText.includes('rights reserved') ||
        footerHTML.includes('©') ||
        footerHTML.includes('&copy;') ||
        bodyHTML.includes('&copy;')
    ) ? 1 : 0;

    // Element counts
    const NoOfImage = document.querySelectorAll('img').length;
    const NoOfCSS = document.querySelectorAll(
        'link[rel="stylesheet"], link[rel="preload"][as="style"], style'
    ).length;
    const NoOfJS = document.querySelectorAll('script[src]').length;

    // Link type counts
    let selfRefCount = 0;
    let emptyRefCount = 0;
    let externalRefCount = 0;

    allLinks.forEach(link => {
        const href = link.getAttribute('href') || '';
        if (href === '' || href === '#') {
            emptyRefCount++;
        } else if (href.startsWith('http') && !href.includes(currentDomain)) {
            externalRefCount++;
        } else {
            selfRefCount++;
        }
    });

    const NoOfSelfRef = selfRefCount;
    const NoOfEmptyRef = emptyRefCount;
    const NoOfExternalRef = externalRefCount;

    return {
        HasTitle, DomainTitleMatchScore, URLTitleMatchScore, HasFavicon,
        Robots, IsResponsive, HasDescription, NoOfPopup, NoOfiFrame,
        HasExternalFormSubmit, HasSocialNet, HasSubmitButton, HasHiddenFields,
        HasPasswordField, Bank, Pay, Crypto, HasCopyrightInfo,
        NoOfImage, NoOfCSS, NoOfJS, NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef
    };
}

// ─── Heuristic Scoring ────────────────────────────────────────────────────────
// #3.6: Accepts domain parameter to avoid redundant new URL() parsing
function computeHeuristicScore(urlFeatures, url, domain) {
    let score = 0;
    const triggeredRules = [];

    if (urlFeatures.IsDomainIP === 1) {
        score += 25;
        triggeredRules.push('IP address used instead of domain name (+25)');
    }

    if (url.includes('@')) {
        score += 20;
        triggeredRules.push('@ symbol detected in URL (+20)');
    }

    const shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
        'short.link', 'tiny.cc', 'is.gd', 'buff.ly', 'rebrand.ly'];
    if (shorteners.some(s => url.includes(s))) {
        score += 15;
        triggeredRules.push('URL shortening service detected (+15)');
    }

    // #3.6: Use passed-in domain instead of re-parsing with new URL(url).hostname
    if (domain.includes('-')) {
        score += 15;
        triggeredRules.push('Hyphenated domain detected (+15)');
    }

    if (urlFeatures.URLLength > 75) {
        score += 10;
        triggeredRules.push('URL length exceeds 75 characters (+10)');
    }

    const urlPath = url.replace('https://', '').replace('http://', '');
    if (urlPath.includes('//')) {
        score += 10;
        triggeredRules.push('Suspicious redirection detected in URL (+10)');
    }

    if (domain.toLowerCase().includes('https')) {
        score += 10;
        triggeredRules.push('HTTPS token found in domain name (+10)');
    }

    let heuristicLevel;
    if (score <= 39) {
        heuristicLevel = 'Low Risk';
    } else if (score <= 60) {
        heuristicLevel = 'Suspicious';
    } else {
        heuristicLevel = 'High Risk';
    }

    return { score, heuristicLevel, triggeredRules };
}

// ─── Main Analysis Function ───────────────────────────────────────────────────
function analyseCurrentPage() {
    const url = window.location.href;

    if (url.startsWith('chrome://') || url.startsWith('chrome-extension://') ||
        url.startsWith('about:') || url.startsWith('edge://')) {
        return;
    }

    // Notify background that analysis has started
    chrome.runtime.sendMessage({ type: 'ANALYSIS_STARTED' }, () => {
        if (chrome.runtime.lastError) {
            console.warn('PhishGuard: Failed to send ANALYSIS_STARTED.', chrome.runtime.lastError.message);
        }
    });

    try {
        const urlFeatures = extractURLFeatures(url);
        const domFeatures = extractDOMFeatures();

        // Build allFeatures without the internal _domain field
        const { _domain, ...cleanUrlFeatures } = urlFeatures;
        const allFeatures = { ...cleanUrlFeatures, ...domFeatures };

        // #3.6: Pass domain directly instead of re-parsing URL
        const heuristicResult = computeHeuristicScore(urlFeatures, url, urlFeatures._domain);

        // #1.1 / #6.1: Send features to background.js for API call
        // Background handles: API call, final threat level, result storage, badge update
        chrome.runtime.sendMessage(
            {
                type: 'ANALYSE_URL',
                data: {
                    url: url,
                    allFeatures: allFeatures,
                    heuristicResult: heuristicResult
                }
            },
            (response) => {
                // #6.2: Check for sendMessage errors
                if (chrome.runtime.lastError) {
                    console.warn('PhishGuard: Failed to send to background.', chrome.runtime.lastError.message);
                }
            }
        );

    } catch (error) {
        console.error('PhishGuard: Analysis failed.', error);
        // Notify background so status doesn't stay stuck on 'loading'
        chrome.runtime.sendMessage(
            { type: 'ANALYSIS_FAILED', data: { url: url } },
            () => {
                if (chrome.runtime.lastError) {
                    console.warn('PhishGuard: Failed to send failure notice.');
                }
            }
        );
    }
}

// ─── Run on Page Load ─────────────────────────────────────────────────────────
analyseCurrentPage();