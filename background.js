// PhishGuard Background Service Worker
// Manages state between content script and popup, handles API calls
// Includes hybrid edge-cloud inference: on-device Random Forest fallback
// when the cloud ML API (Render) is unreachable.

const API_URL = 'https://phishguard-api-jlzn.onrender.com/predict';
const API_KEY = 'phishguard-api-key-2025';

const tabResults = {};
const tabStatus = {}; // tracks whether analysis is in progress
const urlCache = {};  // #4.4: URL-based result caching

const MAX_HISTORY_ENTRIES = 30; // Feature B: cap on stored scan history entries

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

// ─── Scan History (Feature B) ────────────────────────────────────────────────
function saveToHistory(analysisResult) {
    chrome.storage.local.get(['scanHistory'], (data) => {
        const history = data.scanHistory || [];

        let hostname = analysisResult.url;
        try {
            hostname = new URL(analysisResult.url).hostname;
        } catch (e) {
            // keep raw url as fallback if it somehow isn't a valid URL
        }

        history.unshift({
            url: analysisResult.url,
            hostname: hostname,
            finalThreatLevel: analysisResult.finalThreatLevel,
            heuristicScore: analysisResult.heuristicScore,
            timestamp: analysisResult.timestamp
        });

        // Cap at MAX_HISTORY_ENTRIES, dropping oldest entries first
        const trimmed = history.slice(0, MAX_HISTORY_ENTRIES);

        chrome.storage.local.set({ scanHistory: trimmed });
    });
}

// ─── Push warning to content script for High Risk sites (Feature A) ──────────
function notifyTabIfHighRisk(tabId, analysisResult) {
    if (analysisResult.finalThreatLevel === 'High Risk') {
        chrome.tabs.sendMessage(tabId, {
            type: 'SHOW_WARNING',
            data: analysisResult
        }, () => {
            if (chrome.runtime.lastError) {
                console.warn('PhishGuard: Could not deliver warning to tab.', chrome.runtime.lastError.message);
            }
        });
    }
}

// ─── Edge Model Support: Feature Column Order & Scaler Parameters ────────────
// Column order MUST match the order used during training (validated against
// Python sklearn output — see Sprint 5 diagnostic validation).
const EDGE_MODEL_COLUMNS = ["URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS", "HasTitle", "DomainTitleMatchScore", "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive", "HasDescription", "NoOfPopup", "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef"];

// StandardScaler parameters exported from the trained scaler (mean_/scale_),
// applied only to the continuous_features subset, mirroring scaler.transform().
const EDGE_SCALER_PARAMS = { "continuous_features": ["URLLength", "DomainLength", "URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain", "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "DomainTitleMatchScore", "URLTitleMatchScore", "NoOfPopup", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef"], "mean": [34.57309527343667, 21.470395894739074, 78.430777963563, 0.8455082347139422, 0.2604229601543713, 0.05574724121675609, 2.7644564134099534, 1.1647575224241395, 0.024860578044487797, 0.00013842956805699866, 19.428919188277952, 0.5159455119913484, 1.8810110477321402, 0.02861616658538137, 0.06224050552386607, 0.029402659089463306, 0.02505566275790411, 2.340198053393838, 0.06330937466867406, 50.13142743413392, 52.12209777556208, 0.22176466846201148, 26.0756886278335, 6.333111389130389, 10.522305392395937, 65.07111261901228, 2.377628872537586, 49.262516168705865], "scale": [41.31406507181004, 9.150773819772796, 28.975993221260776, 0.21663196130255472, 0.25162770160817893, 0.010587053128888559, 0.5997373881945334, 0.6009680514933989, 1.876245267876546, 0.0038172021334155685, 29.09026814370105, 0.12331442039447021, 11.886670282883584, 0.07089715474783871, 0.9347021194386412, 0.19350499733234272, 0.8364457636259013, 3.527596014349466, 0.03239290509973864, 49.67687527788963, 49.600459215484676, 3.870531766137717, 79.4116463288522, 74.86613742665642, 22.312144689657547, 176.6871643175565, 17.641059965971774, 161.02708806604213] };

// ─── Feature Scaling (mirrors Python's StandardScaler) ───────────────────────
function scaleFeatures(rawFeatures, columnOrder, scalerParams) {
    const { continuous_features, mean, scale } = scalerParams;
    return columnOrder.map((col) => {
        const rawValue = rawFeatures[col];
        const contIdx = continuous_features.indexOf(col);
        if (contIdx === -1) {
            return rawValue; // binary/categorical feature, passed through unscaled
        }
        return (rawValue - mean[contIdx]) / scale[contIdx];
    });
}

// ─── Run the on-device edge model (fallback when the cloud API is unreachable) 
// Returns { phishingProbability, label } in the SAME shape as the cloud API's
// response fields (phishing_probability / label), so downstream code can treat
// both sources identically.
function runEdgeModel(allFeatures) {
    const scaledInput = scaleFeatures(allFeatures, EDGE_MODEL_COLUMNS, EDGE_SCALER_PARAMS);
    const output = score(scaledInput); // [prob_class0=phishing, prob_class1=legitimate]
    const phishingProbability = output[0];
    const label = phishingProbability >= 0.5 ? 'phishing' : 'legitimate';
    return { phishingProbability, label };
}

// ─── On-Device Random Forest (m2cgen-exported, validated bit-exact vs Python) 
// Compact edge model: 15 trees, max_depth=8 — trades a small amount of accuracy
// (99.99% vs the cloud model's 100.00% on the held-out test set) for the ability
// to run entirely offline, with zero network dependency.
function score(input) {
    var var0;
    if (input[21] <= 0.5) {
        var0 = [1.0, 0.0];
    } else {
        if (input[4] <= -0.15626885741949081) {
            if (input[45] <= -0.2779812812805176) {
                if (input[44] <= -0.1064351461827755) {
                    if (input[23] <= 0.7438451945781708) {
                        if (input[41] <= 0.0022291603963822126) {
                            if (input[40] <= -0.0828050896525383) {
                                if (input[39] <= 0.5) {
                                    var0 = [0.9983596551461051, 0.0016403448538948654];
                                } else {
                                    var0 = [0.93346911065852, 0.06653088934147998];
                                }
                            } else {
                                if (input[3] <= 0.1568361222743988) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[20] <= 1.1789811253547668) {
                                if (input[43] <= -0.34847530722618103) {
                                    var0 = [0.41379310344827586, 0.5862068965517241];
                                } else {
                                    var0 = [0.019230769230769232, 0.9807692307692307];
                                }
                            } else {
                                if (input[43] <= -0.2579197660088539) {
                                    var0 = [0.9642857142857143, 0.03571428571428571];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[13] <= 0.3856360614299774) {
                            if (input[3] <= 0.6105630546808243) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[42] <= -0.4491861164569855) {
                                    var0 = [0.16666666666666666, 0.8333333333333334];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[41] <= -0.0311637744307518) {
                                if (input[8] <= -1.1061445325613022) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.5714285714285714, 0.42857142857142855];
                                }
                            } else {
                                if (input[5] <= -0.9676447212696075) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[43] <= -0.35413502156734467) {
                        if (input[6] <= 0.5420522391796112) {
                            if (input[3] <= 0.46773459762334824) {
                                var0 = [1.0, 0.0];
                            } else {
                                var0 = [0.0, 1.0];
                            }
                        } else {
                            if (input[42] <= -0.4491861164569855) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[3] <= 0.23263804614543915) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[20] <= 1.8735777735710144) {
                            if (input[3] <= 0.009358614683151245) {
                                var0 = [1.0, 0.0];
                            } else {
                                var0 = [0.0, 1.0];
                            }
                        } else {
                            if (input[25] <= 0.5) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    }
                }
            } else {
                if (input[3] <= 0.3027139827609062) {
                    var0 = [1.0, 0.0];
                } else {
                    var0 = [0.0, 1.0];
                }
            }
        } else {
            if (input[0] <= 0.1676645651459694) {
                if (input[23] <= -0.5897732973098755) {
                    if (input[40] <= -0.30947208404541016) {
                        if (input[45] <= -0.2717711478471756) {
                            if (input[3] <= 0.6804569065570831) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[20] <= 0.5615620017051697) {
                                    var0 = [0.05263157894736842, 0.9473684210526315];
                                } else {
                                    var0 = [1.0, 0.0];
                                }
                            }
                        } else {
                            if (input[13] <= 1.014110803604126) {
                                if (input[42] <= -0.3595488369464874) {
                                    var0 = [0.3333333333333333, 0.6666666666666666];
                                } else {
                                    var0 = [0.02564102564102564, 0.9743589743589743];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[43] <= -0.35413502156734467) {
                            if (input[24] <= -1.016078531742096) {
                                if (input[3] <= 0.6659103035926819) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        } else {
                            if (input[20] <= 0.5306910574436188) {
                                if (input[13] <= 1.4479610919952393) {
                                    var0 = [0.0010672358591248667, 0.9989327641408752];
                                } else {
                                    var0 = [1.0, 0.0];
                                }
                            } else {
                                if (input[27] <= 0.5) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.5925925925925926, 0.4074074074074074];
                                }
                            }
                        }
                    }
                } else {
                    if (input[45] <= -0.2966116815805435) {
                        if (input[3] <= 0.6535632014274597) {
                            var0 = [1.0, 0.0];
                        } else {
                            if (input[13] <= 1.30604749917984) {
                                if (input[15] <= 1.2537009716033936) {
                                    var0 = [0.004357298474945534, 0.9956427015250545];
                                } else {
                                    var0 = [0.42857142857142855, 0.5714285714285714];
                                }
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[43] <= -0.35413502156734467) {
                            if (input[8] <= -1.1061445325613022) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[20] <= 0.0830621812492609) {
                                    var0 = [0.0017048683462777042, 0.9982951316537223];
                                } else {
                                    var0 = [0.08888888888888889, 0.9111111111111111];
                                }
                            }
                        } else {
                            if (input[13] <= 1.5979841351509094) {
                                if (input[15] <= 1.0632843375205994) {
                                    var0 = [0.00016045276853958806, 0.9998395472314604];
                                } else {
                                    var0 = [0.005504587155963303, 0.9944954128440368];
                                }
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    }
                }
            } else {
                if (input[6] <= 0.5320398509502411) {
                    if (input[45] <= -0.2407204732298851) {
                        if (input[3] <= 0.2802075892686844) {
                            var0 = [1.0, 0.0];
                        } else {
                            var0 = [0.0, 1.0];
                        }
                    } else {
                        if (input[12] <= 0.655582845211029) {
                            if (input[15] <= -0.26257988438010216) {
                                if (input[4] <= 0.18889334797859192) {
                                    var0 = [0.8571428571428571, 0.14285714285714285];
                                } else {
                                    var0 = [0.07563025210084033, 0.9243697478991597];
                                }
                            } else {
                                if (input[3] <= -0.32176560163497925) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var0 = [1.0, 0.0];
                        }
                    }
                } else {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[35] <= 0.5) {
                            if (input[42] <= -0.4491861164569855) {
                                if (input[45] <= -0.10409749299287796) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            } else {
                                if (input[42] <= -0.13545561954379082) {
                                    var0 = [0.05555555555555555, 0.9444444444444444];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[12] <= 0.5524555742740631) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[3] <= 0.047502756118774414) {
                            var0 = [1.0, 0.0];
                        } else {
                            var0 = [0.0, 1.0];
                        }
                    }
                }
            }
        }
    }
    var var1;
    if (input[40] <= -0.29687948524951935) {
        if (input[32] <= 0.5) {
            if (input[23] <= -0.17039640480652452) {
                if (input[40] <= -0.32206469774246216) {
                    if (input[28] <= 0.5) {
                        if (input[45] <= -0.1227279007434845) {
                            if (input[3] <= 0.6827416121959686) {
                                var1 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.2966116815805435) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var1 = [0.0, 1.0];
                        }
                    } else {
                        if (input[13] <= -0.4536818414926529) {
                            if (input[3] <= 0.4420396685600281) {
                                var1 = [1.0, 0.0];
                            } else {
                                if (input[33] <= 0.5) {
                                    var1 = [0.05263157894736842, 0.9473684210526315];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[15] <= -0.29784223437309265) {
                                if (input[34] <= 0.5) {
                                    var1 = [0.9542857142857143, 0.045714285714285714];
                                } else {
                                    var1 = [0.75, 0.25];
                                }
                            } else {
                                var1 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[42] <= -0.13545561954379082) {
                        if (input[43] <= -0.3088572472333908) {
                            if (input[3] <= 0.6659103035926819) {
                                var1 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.2966116815805435) {
                                    var1 = [0.3333333333333333, 0.6666666666666666];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[12] <= 0.4149525314569473) {
                                if (input[41] <= -0.024485187605023384) {
                                    var1 = [0.0, 1.0];
                                } else {
                                    var1 = [0.024390243902439025, 0.975609756097561];
                                }
                            } else {
                                var1 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[19] <= 0.045300520956516266) {
                            if (input[8] <= -1.1061445325613022) {
                                var1 = [1.0, 0.0];
                            } else {
                                if (input[7] <= -0.44095367193222046) {
                                    var1 = [0.0070921985815602835, 0.9929078014184397];
                                } else {
                                    var1 = [0.07692307692307693, 0.9230769230769231];
                                }
                            }
                        } else {
                            if (input[8] <= 2.2218194007873535) {
                                if (input[15] <= -0.34720951318740845) {
                                    var1 = [0.9125, 0.0875];
                                } else {
                                    var1 = [1.0, 0.0];
                                }
                            } else {
                                if (input[45] <= -0.28419141471385956) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[3] <= 0.6693269312381744) {
                    var1 = [1.0, 0.0];
                } else {
                    if (input[43] <= -0.3597947508096695) {
                        if (input[42] <= -0.4491861164569855) {
                            if (input[39] <= 0.5) {
                                if (input[40] <= -0.30947208404541016) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            } else {
                                if (input[1] <= 0.16715571284294128) {
                                    var1 = [0.14705882352941177, 0.8529411764705882];
                                } else {
                                    var1 = [1.0, 0.0];
                                }
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var1 = [1.0, 0.0];
                            } else {
                                if (input[25] <= 0.5) {
                                    var1 = [0.022727272727272728, 0.9772727272727273];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[0] <= -0.4132513999938965) {
                            if (input[1] <= -1.1988489627838135) {
                                var1 = [0.0, 1.0];
                            } else {
                                var1 = [1.0, 0.0];
                            }
                        } else {
                            var1 = [0.0, 1.0];
                        }
                    }
                }
            }
        } else {
            if (input[42] <= -0.40436747670173645) {
                if (input[45] <= -0.29040154814720154) {
                    if (input[23] <= -0.148029625415802) {
                        var1 = [1.0, 0.0];
                    } else {
                        if (input[0] <= -0.09858858585357666) {
                            if (input[41] <= -0.07791388034820557) {
                                var1 = [1.0, 0.0];
                            } else {
                                var1 = [0.0, 1.0];
                            }
                        } else {
                            var1 = [1.0, 0.0];
                        }
                    }
                } else {
                    if (input[19] <= 0.045300520956516266) {
                        if (input[1] <= 0.22179590165615082) {
                            if (input[5] <= -1.0342583656311035) {
                                if (input[3] <= 0.2833777442574501) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            } else {
                                if (input[21] <= 0.5) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[40] <= -0.30947208404541016) {
                                if (input[27] <= 0.5) {
                                    var1 = [0.0, 1.0];
                                } else {
                                    var1 = [1.0, 0.0];
                                }
                            } else {
                                var1 = [0.0, 1.0];
                            }
                        }
                    } else {
                        var1 = [1.0, 0.0];
                    }
                }
            } else {
                if (input[24] <= 0.9622800946235657) {
                    if (input[5] <= 0.16298678517341614) {
                        if (input[29] <= 0.7177916467189789) {
                            if (input[45] <= -0.28419141471385956) {
                                if (input[21] <= 0.5) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.4, 0.6];
                                }
                            } else {
                                if (input[13] <= 1.0465482473373413) {
                                    var1 = [0.005434782608695652, 0.9945652173913043];
                                } else {
                                    var1 = [0.45454545454545453, 0.5454545454545454];
                                }
                            }
                        } else {
                            if (input[1] <= -0.4338863641023636) {
                                if (input[0] <= -0.3527393192052841) {
                                    var1 = [0.0, 1.0];
                                } else {
                                    var1 = [0.5, 0.5];
                                }
                            } else {
                                var1 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[3] <= 0.5347404479980469) {
                            var1 = [1.0, 0.0];
                        } else {
                            var1 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[17] <= 2.431964725255966) {
                        if (input[8] <= -1.1061445325613022) {
                            var1 = [1.0, 0.0];
                        } else {
                            if (input[33] <= 0.5) {
                                if (input[12] <= -0.20381109416484833) {
                                    var1 = [0.0, 1.0];
                                } else {
                                    var1 = [0.01892744479495268, 0.9810725552050473];
                                }
                            } else {
                                if (input[20] <= 0.5615620017051697) {
                                    var1 = [0.0, 1.0];
                                } else {
                                    var1 = [0.2857142857142857, 0.7142857142857143];
                                }
                            }
                        }
                    } else {
                        var1 = [1.0, 0.0];
                    }
                }
            }
        }
    } else {
        if (input[32] <= 0.5) {
            if (input[23] <= -0.6206747591495514) {
                if (input[15] <= -0.35426197946071625) {
                    if (input[19] <= 0.045300520956516266) {
                        if (input[45] <= -0.28419141471385956) {
                            if (input[7] <= -0.44095367193222046) {
                                if (input[3] <= 0.4734257906675339) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            } else {
                                if (input[35] <= 0.5) {
                                    var1 = [0.5273972602739726, 0.4726027397260274];
                                } else {
                                    var1 = [0.8048780487804879, 0.1951219512195122];
                                }
                            }
                        } else {
                            if (input[13] <= 1.2979381084442139) {
                                if (input[3] <= 0.34292419254779816) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            } else {
                                if (input[43] <= -0.34847529232501984) {
                                    var1 = [1.0, 0.0];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[43] <= -0.3428155779838562) {
                            if (input[13] <= 0.35319866240024567) {
                                if (input[4] <= -0.8708109557628632) {
                                    var1 = [0.2835820895522388, 0.7164179104477612];
                                } else {
                                    var1 = [1.0, 0.0];
                                }
                            } else {
                                if (input[12] <= 0.174322247505188) {
                                    var1 = [0.9626168224299065, 0.037383177570093455];
                                } else {
                                    var1 = [0.9990662931839402, 0.0009337068160597573];
                                }
                            }
                        } else {
                            if (input[19] <= 0.895738035440445) {
                                if (input[13] <= 0.9249079525470734) {
                                    var1 = [0.002173913043478261, 0.9978260869565218];
                                } else {
                                    var1 = [0.5384615384615384, 0.46153846153846156];
                                }
                            } else {
                                if (input[12] <= 0.655582845211029) {
                                    var1 = [0.8, 0.2];
                                } else {
                                    var1 = [1.0, 0.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[39] <= 0.5) {
                            var1 = [1.0, 0.0];
                        } else {
                            var1 = [0.0, 1.0];
                        }
                    } else {
                        if (input[12] <= -0.2725626081228256) {
                            if (input[3] <= 0.3877654420211911) {
                                var1 = [1.0, 0.0];
                            } else {
                                if (input[12] <= -0.37568987905979156) {
                                    var1 = [0.125, 0.875];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[3] <= 0.5721969455480576) {
                                var1 = [1.0, 0.0];
                            } else {
                                var1 = [0.0, 1.0];
                            }
                        }
                    }
                }
            } else {
                if (input[8] <= -1.1061445325613022) {
                    var1 = [1.0, 0.0];
                } else {
                    if (input[3] <= 0.6420855820178986) {
                        var1 = [1.0, 0.0];
                    } else {
                        if (input[13] <= 0.8721971809864044) {
                            if (input[45] <= -0.2966116815805435) {
                                if (input[4] <= -1.097787857055664) {
                                    var1 = [0.02, 0.98];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            } else {
                                if (input[40] <= -0.27169427275657654) {
                                    var1 = [0.001563721657544957, 0.9984362783424551];
                                } else {
                                    var1 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[0] <= -0.025974091608077288) {
                                var1 = [1.0, 0.0];
                            } else {
                                var1 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        } else {
            if (input[0] <= 0.40971288084983826) {
                if (input[42] <= -0.4491861164569855) {
                    if (input[7] <= 1.226442813873291) {
                        if (input[39] <= 0.5) {
                            if (input[14] <= -0.03205363964661956) {
                                if (input[12] <= 0.19151011481881142) {
                                    var1 = [0.08843537414965986, 0.9115646258503401];
                                } else {
                                    var1 = [1.0, 0.0];
                                }
                            } else {
                                var1 = [1.0, 0.0];
                            }
                        } else {
                            if (input[33] <= 0.5) {
                                var1 = [0.0, 1.0];
                            } else {
                                if (input[13] <= 0.03693394549190998) {
                                    var1 = [0.0, 1.0];
                                } else {
                                    var1 = [0.030303030303030304, 0.9696969696969697];
                                }
                            }
                        }
                    } else {
                        if (input[3] <= 0.38206135388463736) {
                            var1 = [1.0, 0.0];
                        } else {
                            var1 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[8] <= -1.1061445325613022) {
                        var1 = [1.0, 0.0];
                    } else {
                        if (input[21] <= 0.5) {
                            var1 = [1.0, 0.0];
                        } else {
                            if (input[3] <= 0.35492668487131596) {
                                var1 = [1.0, 0.0];
                            } else {
                                var1 = [0.0, 1.0];
                            }
                        }
                    }
                }
            } else {
                if (input[3] <= -0.17604869604110718) {
                    var1 = [1.0, 0.0];
                } else {
                    var1 = [0.0, 1.0];
                }
            }
        }
    }
    var var2;
    if (input[40] <= -0.29687948524951935) {
        if (input[24] <= 0.9637757241725922) {
            if (input[25] <= 0.5) {
                if (input[37] <= 0.5) {
                    if (input[42] <= -0.0906369760632515) {
                        if (input[41] <= -0.05119953490793705) {
                            if (input[4] <= -3.8158740997314453) {
                                var2 = [0.0, 1.0];
                            } else {
                                if (input[28] <= 0.5) {
                                    var2 = [0.9991233586797019, 0.0008766413202980581];
                                } else {
                                    var2 = [0.9268482490272374, 0.07315175097276265];
                                }
                            }
                        } else {
                            if (input[3] <= 0.5684983134269714) {
                                var2 = [1.0, 0.0];
                            } else {
                                var2 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[3] <= 0.20206163823604584) {
                            var2 = [1.0, 0.0];
                        } else {
                            var2 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[42] <= -0.04581833723932505) {
                        if (input[0] <= -0.02597408974543214) {
                            if (input[3] <= 0.6112809628248215) {
                                var2 = [1.0, 0.0];
                            } else {
                                if (input[22] <= 0.5) {
                                    var2 = [0.5, 0.5];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[35] <= 0.5) {
                                if (input[30] <= 2.5) {
                                    var2 = [0.9860664523043944, 0.013933547695605574];
                                } else {
                                    var2 = [0.6153846153846154, 0.38461538461538464];
                                }
                            } else {
                                if (input[3] <= 0.1573905199766159) {
                                    var2 = [1.0, 0.0];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[3] <= 0.057890474796295166) {
                            var2 = [1.0, 0.0];
                        } else {
                            var2 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[43] <= -0.29753781855106354) {
                    if (input[39] <= 0.5) {
                        if (input[3] <= 0.6535632014274597) {
                            var2 = [1.0, 0.0];
                        } else {
                            var2 = [0.0, 1.0];
                        }
                    } else {
                        if (input[3] <= 0.48722052574157715) {
                            var2 = [1.0, 0.0];
                        } else {
                            var2 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[0] <= 0.1555621474981308) {
                        if (input[24] <= 0.8458555638790131) {
                            if (input[21] <= 0.5) {
                                var2 = [1.0, 0.0];
                            } else {
                                var2 = [0.0, 1.0];
                            }
                        } else {
                            var2 = [1.0, 0.0];
                        }
                    } else {
                        if (input[14] <= -0.11618148908019066) {
                            var2 = [0.0, 1.0];
                        } else {
                            var2 = [1.0, 0.0];
                        }
                    }
                }
            }
        } else {
            if (input[3] <= 0.6659103035926819) {
                var2 = [1.0, 0.0];
            } else {
                if (input[45] <= -0.2966116815805435) {
                    if (input[41] <= -0.07791388034820557) {
                        if (input[39] <= 0.5) {
                            if (input[42] <= -0.4491861164569855) {
                                var2 = [1.0, 0.0];
                            } else {
                                if (input[5] <= 0.16298678517341614) {
                                    var2 = [0.0, 1.0];
                                } else {
                                    var2 = [0.7, 0.3];
                                }
                            }
                        } else {
                            if (input[0] <= -0.29222723841667175) {
                                var2 = [0.0, 1.0];
                            } else {
                                if (input[27] <= 0.5) {
                                    var2 = [0.5, 0.5];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[12] <= 0.0540070915594697) {
                            if (input[26] <= 0.5) {
                                var2 = [0.0, 1.0];
                            } else {
                                if (input[43] <= -0.3597947508096695) {
                                    var2 = [0.2857142857142857, 0.7142857142857143];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var2 = [1.0, 0.0];
                        }
                    }
                } else {
                    if (input[42] <= -0.4491861164569855) {
                        if (input[39] <= 0.5) {
                            if (input[12] <= -0.13505957648158073) {
                                if (input[12] <= -0.20381109416484833) {
                                    var2 = [0.08571428571428572, 0.9142857142857143];
                                } else {
                                    var2 = [0.2857142857142857, 0.7142857142857143];
                                }
                            } else {
                                if (input[6] <= 0.6609805226325989) {
                                    var2 = [0.0, 1.0];
                                } else {
                                    var2 = [1.0, 0.0];
                                }
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var2 = [1.0, 0.0];
                            } else {
                                var2 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[0] <= -0.4132513999938965) {
                            if (input[44] <= -0.1064351461827755) {
                                if (input[32] <= 0.5) {
                                    var2 = [0.10526315789473684, 0.8947368421052632];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            } else {
                                var2 = [0.0, 1.0];
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var2 = [1.0, 0.0];
                            } else {
                                var2 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[12] <= 0.27744950354099274) {
            if (input[45] <= -0.2779812812805176) {
                if (input[42] <= -0.04581833723932505) {
                    if (input[36] <= 0.5) {
                        if (input[20] <= 0.5306910574436188) {
                            if (input[3] <= 0.6420855820178986) {
                                var2 = [1.0, 0.0];
                            } else {
                                if (input[21] <= 0.5) {
                                    var2 = [1.0, 0.0];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[12] <= -0.30693836510181427) {
                                if (input[19] <= 0.045300520956516266) {
                                    var2 = [0.08264462809917356, 0.9173553719008265];
                                } else {
                                    var2 = [0.696969696969697, 0.30303030303030304];
                                }
                            } else {
                                if (input[4] <= -1.435542643070221) {
                                    var2 = [0.823170731707317, 0.17682926829268292];
                                } else {
                                    var2 = [0.991578947368421, 0.008421052631578947];
                                }
                            }
                        }
                    } else {
                        if (input[12] <= -0.23818685114383698) {
                            if (input[15] <= 1.8531608581542969) {
                                if (input[14] <= -0.07411756506189704) {
                                    var2 = [0.06779661016949153, 0.9322033898305084];
                                } else {
                                    var2 = [0.5, 0.5];
                                }
                            } else {
                                var2 = [1.0, 0.0];
                            }
                        } else {
                            if (input[28] <= 0.5) {
                                if (input[6] <= 0.9608131945133209) {
                                    var2 = [0.9920508744038156, 0.00794912559618442];
                                } else {
                                    var2 = [0.38461538461538464, 0.6153846153846154];
                                }
                            } else {
                                if (input[3] <= 0.580042839050293) {
                                    var2 = [1.0, 0.0];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[19] <= 0.3287796899676323) {
                        if (input[16] <= 0.4683411903679371) {
                            if (input[18] <= 1.1655799122527242) {
                                if (input[3] <= 0.3295828849077225) {
                                    var2 = [1.0, 0.0];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            } else {
                                var2 = [1.0, 0.0];
                            }
                        } else {
                            var2 = [1.0, 0.0];
                        }
                    } else {
                        if (input[43] <= 0.016576677560806274) {
                            var2 = [1.0, 0.0];
                        } else {
                            var2 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[21] <= 0.5) {
                    var2 = [1.0, 0.0];
                } else {
                    if (input[24] <= -0.8780295550823212) {
                        if (input[42] <= -0.4491861164569855) {
                            if (input[19] <= -0.23817865550518036) {
                                var2 = [0.0, 1.0];
                            } else {
                                if (input[14] <= -0.11618148908019066) {
                                    var2 = [0.6352941176470588, 0.36470588235294116];
                                } else {
                                    var2 = [1.0, 0.0];
                                }
                            }
                        } else {
                            if (input[3] <= 0.3865265743806958) {
                                var2 = [1.0, 0.0];
                            } else {
                                var2 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[13] <= 1.33443021774292) {
                            if (input[8] <= -1.1061445325613022) {
                                var2 = [1.0, 0.0];
                            } else {
                                if (input[5] <= -1.034765899181366) {
                                    var2 = [0.0032258064516129032, 0.9967741935483871];
                                } else {
                                    var2 = [0.000054124269322364146, 0.9999458757306776];
                                }
                            }
                        } else {
                            var2 = [1.0, 0.0];
                        }
                    }
                }
            }
        } else {
            if (input[43] <= -0.3088572472333908) {
                if (input[42] <= 0.0886375829577446) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[6] <= -0.14167646318674088) {
                            var2 = [1.0, 0.0];
                        } else {
                            if (input[35] <= 0.5) {
                                if (input[21] <= 0.5) {
                                    var2 = [1.0, 0.0];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            } else {
                                var2 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[0] <= 0.2886887192726135) {
                            if (input[29] <= 0.2010667733848095) {
                                if (input[4] <= -2.106372833251953) {
                                    var2 = [0.9, 0.1];
                                } else {
                                    var2 = [1.0, 0.0];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var2 = [1.0, 0.0];
                                } else {
                                    var2 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var2 = [1.0, 0.0];
                        }
                    }
                } else {
                    if (input[0] <= 0.22817663848400116) {
                        var2 = [0.0, 1.0];
                    } else {
                        var2 = [1.0, 0.0];
                    }
                }
            } else {
                if (input[3] <= -0.3775026798248291) {
                    var2 = [1.0, 0.0];
                } else {
                    var2 = [0.0, 1.0];
                }
            }
        }
    }
    var var3;
    if (input[40] <= -0.30947208404541016) {
        if (input[42] <= -0.3147301971912384) {
            if (input[28] <= 0.5) {
                if (input[39] <= 0.5) {
                    if (input[29] <= 0.33024799078702927) {
                        if (input[21] <= 0.5) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[23] <= 0.9293030202388763) {
                                if (input[45] <= -0.19103939086198807) {
                                    var3 = [0.9998963265024018, 0.00010367349759823064];
                                } else {
                                    var3 = [0.9230769230769231, 0.07692307692307693];
                                }
                            } else {
                                if (input[3] <= 0.6208643913269043) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.14772727272727273, 0.8522727272727273];
                                }
                            }
                        }
                    } else {
                        if (input[3] <= 0.42729123681783676) {
                            var3 = [1.0, 0.0];
                        } else {
                            var3 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[43] <= -0.34847529232501984) {
                        if (input[24] <= 0.2740335091948509) {
                            if (input[45] <= 2.9792346321046352) {
                                if (input[0] <= -0.2196127474308014) {
                                    var3 = [0.9411764705882353, 0.058823529411764705];
                                } else {
                                    var3 = [1.0, 0.0];
                                }
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        } else {
                            if (input[3] <= 0.6335906386375427) {
                                var3 = [1.0, 0.0];
                            } else {
                                if (input[13] <= 0.7586662471294403) {
                                    var3 = [0.05555555555555555, 0.9444444444444444];
                                } else {
                                    var3 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[45] <= -0.2779812812805176) {
                            if (input[13] <= -0.20634660869836807) {
                                var3 = [0.0, 1.0];
                            } else {
                                if (input[40] <= -0.32206469774246216) {
                                    var3 = [0.8333333333333334, 0.16666666666666666];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[7] <= -0.44095367193222046) {
                                var3 = [0.0, 1.0];
                            } else {
                                if (input[19] <= 0.18704010546207428) {
                                    var3 = [0.015384615384615385, 0.9846153846153847];
                                } else {
                                    var3 = [1.0, 0.0];
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[15] <= -0.29784223437309265) {
                    if (input[41] <= -0.07791388034820557) {
                        if (input[21] <= 0.5) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[20] <= -0.08672808855772018) {
                                var3 = [0.0, 1.0];
                            } else {
                                if (input[39] <= 0.5) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[19] <= -0.23817865550518036) {
                            if (input[13] <= 0.7586662471294403) {
                                if (input[44] <= -0.1064351461827755) {
                                    var3 = [0.20054945054945056, 0.7994505494505495];
                                } else {
                                    var3 = [0.013888888888888888, 0.9861111111111112];
                                }
                            } else {
                                if (input[43] <= -0.28338851779699326) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[34] <= 0.5) {
                                if (input[43] <= -0.32017670571804047) {
                                    var3 = [0.868421052631579, 0.13157894736842105];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            } else {
                                if (input[45] <= -0.2624559551477432) {
                                    var3 = [0.9210526315789473, 0.07894736842105263];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[29] <= 0.07188555970788002) {
                        if (input[3] <= 0.598741739988327) {
                            var3 = [1.0, 0.0];
                        } else {
                            var3 = [0.0, 1.0];
                        }
                    } else {
                        var3 = [0.0, 1.0];
                    }
                }
            }
        } else {
            if (input[4] <= -0.13925611972808838) {
                if (input[3] <= 0.5226951539516449) {
                    var3 = [1.0, 0.0];
                } else {
                    var3 = [0.0, 1.0];
                }
            } else {
                if (input[3] <= 0.6178409606218338) {
                    var3 = [1.0, 0.0];
                } else {
                    if (input[39] <= 0.5) {
                        if (input[13] <= 1.2736101150512695) {
                            if (input[6] <= -1.901286542415619) {
                                var3 = [1.0, 0.0];
                            } else {
                                if (input[41] <= -0.07791388034820557) {
                                    var3 = [0.14285714285714285, 0.8571428571428571];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[45] <= -0.2748762145638466) {
                                var3 = [1.0, 0.0];
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        }
                    } else {
                        var3 = [0.0, 1.0];
                    }
                }
            }
        }
    } else {
        if (input[6] <= -0.6978091299533844) {
            if (input[0] <= -0.12279341742396355) {
                if (input[41] <= -0.07791388034820557) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[42] <= -0.4491861164569855) {
                            if (input[43] <= -0.2239614501595497) {
                                if (input[6] <= -0.8196671605110168) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.5, 0.5];
                                }
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var3 = [1.0, 0.0];
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[1] <= -0.7617274820804596) {
                            if (input[40] <= 0.12497299909591675) {
                                var3 = [1.0, 0.0];
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        } else {
                            if (input[3] <= 0.31933489441871643) {
                                var3 = [1.0, 0.0];
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[14] <= 0.30445775389671326) {
                        if (input[21] <= 0.5) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[44] <= -0.1064351461827755) {
                                if (input[13] <= 0.30859722197055817) {
                                    var3 = [0.013015991074748977, 0.9869840089252511];
                                } else {
                                    var3 = [1.0, 0.0];
                                }
                            } else {
                                if (input[42] <= -0.4491861164569855) {
                                    var3 = [0.6, 0.4];
                                } else {
                                    var3 = [0.001017293997965412, 0.9989827060020345];
                                }
                            }
                        }
                    } else {
                        if (input[21] <= 0.5) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[23] <= -0.0026456117630004883) {
                                var3 = [1.0, 0.0];
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        }
                    }
                }
            } else {
                if (input[32] <= 0.5) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[13] <= -2.1566456258296967) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[0] <= -0.05017892271280289) {
                                var3 = [0.0, 1.0];
                            } else {
                                if (input[20] <= -1.0437277555465698) {
                                    var3 = [0.0, 1.0];
                                } else {
                                    var3 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[26] <= 0.5) {
                            if (input[3] <= 0.5721969455480576) {
                                var3 = [1.0, 0.0];
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        } else {
                            if (input[15] <= -0.32605210691690445) {
                                if (input[40] <= -0.28428688645362854) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.08333333333333333, 0.9166666666666666];
                                }
                            } else {
                                var3 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[19] <= 0.6122588664293289) {
                        if (input[3] <= -0.10480928421020508) {
                            var3 = [1.0, 0.0];
                        } else {
                            var3 = [0.0, 1.0];
                        }
                    } else {
                        var3 = [1.0, 0.0];
                    }
                }
            }
        } else {
            if (input[19] <= 0.3287796899676323) {
                if (input[42] <= -0.4491861164569855) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[45] <= -0.2779812812805176) {
                            if (input[33] <= 0.5) {
                                if (input[28] <= 0.5) {
                                    var3 = [0.88, 0.12];
                                } else {
                                    var3 = [0.21428571428571427, 0.7857142857142857];
                                }
                            } else {
                                if (input[3] <= 0.40830789878964424) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[43] <= -0.36545446515083313) {
                                if (input[5] <= -1.0164686441421509) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.1452991452991453, 0.8547008547008547];
                                }
                            } else {
                                if (input[28] <= 0.5) {
                                    var3 = [0.04597701149425287, 0.9540229885057471];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[40] <= -0.1331755369901657) {
                            if (input[45] <= -0.153778575360775) {
                                if (input[20] <= 1.5339971780776978) {
                                    var3 = [0.88996138996139, 0.11003861003861004];
                                } else {
                                    var3 = [0.5652173913043478, 0.43478260869565216];
                                }
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        } else {
                            if (input[33] <= 0.5) {
                                var3 = [0.0, 1.0];
                            } else {
                                if (input[19] <= 0.045300520956516266) {
                                    var3 = [0.0, 1.0];
                                } else {
                                    var3 = [0.2857142857142857, 0.7142857142857143];
                                }
                            }
                        }
                    }
                } else {
                    if (input[19] <= 0.045300520956516266) {
                        if (input[3] <= 0.6659103035926819) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[43] <= -0.3258364200592041) {
                                if (input[21] <= 0.5) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            } else {
                                var3 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[22] <= 0.5) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[12] <= 0.13994648680090904) {
                                if (input[41] <= -0.07791388034820557) {
                                    var3 = [0.7746478873239436, 0.22535211267605634];
                                } else {
                                    var3 = [0.045335658238884045, 0.9546643417611159];
                                }
                            } else {
                                if (input[1] <= 1.0960389375686646) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.36065573770491804, 0.639344262295082];
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[28] <= 0.5) {
                    if (input[23] <= -0.04122832417488098) {
                        if (input[45] <= -0.0916772224009037) {
                            if (input[0] <= -0.0017692586407065392) {
                                if (input[3] <= 0.4070594161748886) {
                                    var3 = [1.0, 0.0];
                                } else {
                                    var3 = [0.0, 1.0];
                                }
                            } else {
                                if (input[40] <= 0.2823806405067444) {
                                    var3 = [0.9931934203062961, 0.006806579693703914];
                                } else {
                                    var3 = [0.625, 0.375];
                                }
                            }
                        } else {
                            if (input[15] <= -0.31899964064359665) {
                                if (input[19] <= 0.895738035440445) {
                                    var3 = [0.024390243902439025, 0.975609756097561];
                                } else {
                                    var3 = [1.0, 0.0];
                                }
                            } else {
                                if (input[0] <= -0.0259740948677063) {
                                    var3 = [0.0, 1.0];
                                } else {
                                    var3 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[21] <= 0.5) {
                            var3 = [1.0, 0.0];
                        } else {
                            if (input[4] <= -1.4300472736358643) {
                                var3 = [0.0, 1.0];
                            } else {
                                var3 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[3] <= 0.3941721599549055) {
                        var3 = [1.0, 0.0];
                    } else {
                        var3 = [0.0, 1.0];
                    }
                }
            }
        }
    }
    var var4;
    if (input[45] <= -0.2779812812805176) {
        if (input[40] <= -0.27169427275657654) {
            if (input[3] <= 0.6904508173465729) {
                var4 = [1.0, 0.0];
            } else {
                if (input[28] <= 0.5) {
                    if (input[41] <= -0.07791388034820557) {
                        if (input[36] <= 0.5) {
                            if (input[21] <= 0.5) {
                                var4 = [1.0, 0.0];
                            } else {
                                if (input[26] <= 0.5) {
                                    var4 = [0.6153846153846154, 0.38461538461538464];
                                } else {
                                    var4 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[39] <= 0.5) {
                                var4 = [1.0, 0.0];
                            } else {
                                var4 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[45] <= -0.3028218150138855) {
                            if (input[39] <= 0.5) {
                                if (input[37] <= 0.5) {
                                    var4 = [0.6666666666666666, 0.3333333333333333];
                                } else {
                                    var4 = [0.0, 1.0];
                                }
                            } else {
                                var4 = [0.0, 1.0];
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var4 = [1.0, 0.0];
                            } else {
                                if (input[6] <= 0.8287954330444336) {
                                    var4 = [0.0, 1.0];
                                } else {
                                    var4 = [0.020618556701030927, 0.979381443298969];
                                }
                            }
                        }
                    }
                } else {
                    if (input[21] <= 0.5) {
                        var4 = [1.0, 0.0];
                    } else {
                        var4 = [0.0, 1.0];
                    }
                }
            }
        } else {
            if (input[28] <= 0.5) {
                if (input[14] <= -0.11618148908019066) {
                    if (input[3] <= 0.5441315770149231) {
                        var4 = [1.0, 0.0];
                    } else {
                        var4 = [0.0, 1.0];
                    }
                } else {
                    if (input[23] <= 0.8920250535011292) {
                        if (input[19] <= -0.23817865550518036) {
                            if (input[23] <= -0.11447948217391968) {
                                var4 = [0.0, 1.0];
                            } else {
                                var4 = [1.0, 0.0];
                            }
                        } else {
                            var4 = [1.0, 0.0];
                        }
                    } else {
                        if (input[44] <= -0.1064351461827755) {
                            if (input[41] <= -0.03784236125648022) {
                                var4 = [1.0, 0.0];
                            } else {
                                var4 = [0.0, 1.0];
                            }
                        } else {
                            var4 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[14] <= 0.2203299105167389) {
                    if (input[0] <= 0.3007911294698715) {
                        if (input[21] <= 0.5) {
                            var4 = [1.0, 0.0];
                        } else {
                            if (input[24] <= 0.8462648391723633) {
                                if (input[20] <= 3.2473353147506714) {
                                    var4 = [0.0093603744149766, 0.9906396255850234];
                                } else {
                                    var4 = [1.0, 0.0];
                                }
                            } else {
                                if (input[42] <= -0.4491861164569855) {
                                    var4 = [0.01282051282051282, 0.9871794871794872];
                                } else {
                                    var4 = [0.000513083632632119, 0.9994869163673679];
                                }
                            }
                        }
                    } else {
                        var4 = [1.0, 0.0];
                    }
                } else {
                    if (input[3] <= 0.03434684872627258) {
                        var4 = [1.0, 0.0];
                    } else {
                        var4 = [0.0, 1.0];
                    }
                }
            }
        }
    } else {
        if (input[3] <= 0.6362443566322327) {
            var4 = [1.0, 0.0];
        } else {
            if (input[42] <= -0.4491861164569855) {
                if (input[40] <= -0.32206469774246216) {
                    if (input[20] <= -0.9356794059276581) {
                        var4 = [1.0, 0.0];
                    } else {
                        if (input[23] <= -0.0026456117630004883) {
                            var4 = [0.0, 1.0];
                        } else {
                            if (input[21] <= 0.5) {
                                var4 = [1.0, 0.0];
                            } else {
                                var4 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[39] <= 0.5) {
                        if (input[28] <= 0.5) {
                            if (input[43] <= -0.36545446515083313) {
                                if (input[32] <= 0.5) {
                                    var4 = [1.0, 0.0];
                                } else {
                                    var4 = [0.0, 1.0];
                                }
                            } else {
                                var4 = [0.0, 1.0];
                            }
                        } else {
                            var4 = [0.0, 1.0];
                        }
                    } else {
                        if (input[12] <= -0.37568987905979156) {
                            if (input[5] <= -0.7997196018695831) {
                                if (input[4] <= 0.13613763451576233) {
                                    var4 = [0.05555555555555555, 0.9444444444444444];
                                } else {
                                    var4 = [0.0, 1.0];
                                }
                            } else {
                                var4 = [0.0, 1.0];
                            }
                        } else {
                            var4 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[21] <= 0.5) {
                    var4 = [1.0, 0.0];
                } else {
                    var4 = [0.0, 1.0];
                }
            }
        }
    }
    var var5;
    if (input[28] <= 0.5) {
        if (input[19] <= -0.23817865550518036) {
            if (input[42] <= -0.40436747670173645) {
                if (input[45] <= -0.2779812812805176) {
                    if (input[21] <= 0.5) {
                        var5 = [1.0, 0.0];
                    } else {
                        if (input[22] <= 0.5) {
                            var5 = [1.0, 0.0];
                        } else {
                            if (input[39] <= 0.5) {
                                if (input[40] <= -0.29687948524951935) {
                                    var5 = [0.9864130434782609, 0.01358695652173913];
                                } else {
                                    var5 = [0.1, 0.9];
                                }
                            } else {
                                if (input[7] <= -0.44095367193222046) {
                                    var5 = [0.0, 1.0];
                                } else {
                                    var5 = [0.03896103896103896, 0.961038961038961];
                                }
                            }
                        }
                    }
                } else {
                    if (input[21] <= 0.5) {
                        var5 = [1.0, 0.0];
                    } else {
                        if (input[0] <= -0.44955864548683167) {
                            var5 = [1.0, 0.0];
                        } else {
                            if (input[13] <= 1.2371180057525635) {
                                if (input[6] <= -0.12848100438714027) {
                                    var5 = [0.01818181818181818, 0.9818181818181818];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            } else {
                                if (input[32] <= 0.5) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[6] <= -1.44103342294693) {
                    if (input[12] <= -0.4100656360387802) {
                        if (input[45] <= -0.2779812812805176) {
                            if (input[3] <= 0.552652895450592) {
                                var5 = [1.0, 0.0];
                            } else {
                                var5 = [0.0, 1.0];
                            }
                        } else {
                            if (input[20] <= -0.17934095859527588) {
                                if (input[21] <= 0.5) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0053475935828877, 0.9946524064171123];
                                }
                            } else {
                                if (input[21] <= 0.5) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[32] <= 0.5) {
                            if (input[40] <= -0.24650904536247253) {
                                if (input[42] <= -0.13545561954379082) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            } else {
                                var5 = [0.0, 1.0];
                            }
                        } else {
                            if (input[38] <= 0.5) {
                                var5 = [0.0, 1.0];
                            } else {
                                var5 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[22] <= 0.5) {
                        if (input[40] <= -0.27169427275657654) {
                            if (input[41] <= 0.1090865470468998) {
                                var5 = [1.0, 0.0];
                            } else {
                                var5 = [0.0, 1.0];
                            }
                        } else {
                            var5 = [0.0, 1.0];
                        }
                    } else {
                        if (input[45] <= -0.3028218150138855) {
                            if (input[3] <= 0.6659103035926819) {
                                var5 = [1.0, 0.0];
                            } else {
                                if (input[12] <= 0.0540070915594697) {
                                    var5 = [0.0, 1.0];
                                } else {
                                    var5 = [0.3333333333333333, 0.6666666666666666];
                                }
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var5 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.2779812812805176) {
                                    var5 = [0.028047464940668825, 0.9719525350593312];
                                } else {
                                    var5 = [0.0003073455588566745, 0.9996926544411433];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[44] <= -0.1064351461827755) {
                if (input[45] <= -0.25314074754714966) {
                    if (input[40] <= -0.0828050896525383) {
                        if (input[42] <= -0.04581833723932505) {
                            if (input[3] <= 0.6926922798156738) {
                                var5 = [1.0, 0.0];
                            } else {
                                if (input[43] <= -0.3597947508096695) {
                                    var5 = [0.7345132743362832, 0.26548672566371684];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[3] <= 0.29948145151138306) {
                                var5 = [1.0, 0.0];
                            } else {
                                var5 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[31] <= 0.5) {
                            if (input[13] <= 1.2411726713180542) {
                                if (input[20] <= 1.5957390666007996) {
                                    var5 = [0.0, 1.0];
                                } else {
                                    var5 = [0.6, 0.4];
                                }
                            } else {
                                var5 = [1.0, 0.0];
                            }
                        } else {
                            if (input[19] <= 0.3287796974182129) {
                                var5 = [0.0, 1.0];
                            } else {
                                var5 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[8] <= -1.1061445325613022) {
                        var5 = [1.0, 0.0];
                    } else {
                        if (input[3] <= 0.5684983134269714) {
                            var5 = [1.0, 0.0];
                        } else {
                            var5 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[32] <= 0.5) {
                    if (input[4] <= 0.3201136142015457) {
                        if (input[45] <= -0.2655610144138336) {
                            if (input[42] <= 0.0886375829577446) {
                                if (input[12] <= -0.3413141220808029) {
                                    var5 = [0.14285714285714285, 0.8571428571428571];
                                } else {
                                    var5 = [0.966786355475763, 0.03321364452423698];
                                }
                            } else {
                                if (input[19] <= 0.3287796899676323) {
                                    var5 = [0.037037037037037035, 0.9629629629629629];
                                } else {
                                    var5 = [1.0, 0.0];
                                }
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var5 = [1.0, 0.0];
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0121580547112462, 0.9878419452887538];
                                }
                            }
                        }
                    } else {
                        if (input[3] <= 0.5270889550447464) {
                            var5 = [1.0, 0.0];
                        } else {
                            var5 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[12] <= 0.449328288435936) {
                        if (input[1] <= -1.0895686149597168) {
                            if (input[30] <= 0.5) {
                                var5 = [0.0, 1.0];
                            } else {
                                var5 = [1.0, 0.0];
                            }
                        } else {
                            if (input[42] <= -0.4491861164569855) {
                                if (input[4] <= -1.155875325202942) {
                                    var5 = [0.5, 0.5];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            } else {
                                if (input[21] <= 0.5) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[4] <= -1.034864366054535) {
                            if (input[12] <= 1.033716157078743) {
                                var5 = [0.0, 1.0];
                            } else {
                                var5 = [1.0, 0.0];
                            }
                        } else {
                            var5 = [1.0, 0.0];
                        }
                    }
                }
            }
        }
    } else {
        if (input[41] <= -0.07791388034820557) {
            if (input[40] <= -0.29687948524951935) {
                if (input[45] <= -0.2655610144138336) {
                    if (input[23] <= 0.6124404668807983) {
                        if (input[20] <= -0.8739375174045563) {
                            if (input[34] <= 0.5) {
                                var5 = [1.0, 0.0];
                            } else {
                                var5 = [0.0, 1.0];
                            }
                        } else {
                            if (input[13] <= -0.07254230137914419) {
                                if (input[44] <= -0.1064351461827755) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            } else {
                                var5 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[3] <= 0.6105630546808243) {
                            var5 = [1.0, 0.0];
                        } else {
                            if (input[32] <= 0.5) {
                                if (input[0] <= -0.29222723841667175) {
                                    var5 = [0.46153846153846156, 0.5384615384615384];
                                } else {
                                    var5 = [0.15151515151515152, 0.8484848484848485];
                                }
                            } else {
                                var5 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[12] <= -0.16943533718585968) {
                        var5 = [0.0, 1.0];
                    } else {
                        if (input[13] <= 1.050602912902832) {
                            if (input[15] <= 0.05478121340274811) {
                                if (input[21] <= 0.5) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            } else {
                                var5 = [1.0, 0.0];
                            }
                        } else {
                            var5 = [1.0, 0.0];
                        }
                    }
                }
            } else {
                if (input[45] <= -0.28419141471385956) {
                    if (input[4] <= -1.0680063962936401) {
                        if (input[1] <= 0.3310762792825699) {
                            var5 = [0.0, 1.0];
                        } else {
                            var5 = [1.0, 0.0];
                        }
                    } else {
                        if (input[4] <= 0.4416164308786392) {
                            var5 = [0.0, 1.0];
                        } else {
                            if (input[21] <= 0.5) {
                                var5 = [1.0, 0.0];
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.011111111111111112, 0.9888888888888889];
                                }
                            }
                        }
                    }
                } else {
                    if (input[4] <= -2.2446857690811157) {
                        var5 = [1.0, 0.0];
                    } else {
                        if (input[8] <= -1.1061445325613022) {
                            var5 = [1.0, 0.0];
                        } else {
                            if (input[22] <= 0.5) {
                                if (input[15] <= 0.21698801219463348) {
                                    var5 = [0.0, 1.0];
                                } else {
                                    var5 = [1.0, 0.0];
                                }
                            } else {
                                if (input[1] <= -0.6524471044540405) {
                                    var5 = [0.004048582995951417, 0.9959514170040485];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[3] <= 0.580042839050293) {
                var5 = [1.0, 0.0];
            } else {
                if (input[12] <= -0.23818685114383698) {
                    if (input[23] <= 0.6195571720600128) {
                        if (input[7] <= -0.44095367193222046) {
                            var5 = [0.0, 1.0];
                        } else {
                            if (input[6] <= -1.3090821504592896) {
                                if (input[21] <= 0.5) {
                                    var5 = [1.0, 0.0];
                                } else {
                                    var5 = [0.0, 1.0];
                                }
                            } else {
                                if (input[13] <= -0.2104012742638588) {
                                    var5 = [0.0, 1.0];
                                } else {
                                    var5 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[43] <= -0.36545446515083313) {
                            if (input[40] <= -0.30947208404541016) {
                                if (input[5] <= -0.7997196018695831) {
                                    var5 = [0.0, 1.0];
                                } else {
                                    var5 = [0.05128205128205128, 0.9487179487179487];
                                }
                            } else {
                                var5 = [0.0, 1.0];
                            }
                        } else {
                            var5 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[44] <= 4.96695613861084) {
                        var5 = [0.0, 1.0];
                    } else {
                        if (input[44] <= 5.108670949935913) {
                            var5 = [1.0, 0.0];
                        } else {
                            var5 = [0.0, 1.0];
                        }
                    }
                }
            }
        }
    }
    var var6;
    if (input[21] <= 0.5) {
        var6 = [1.0, 0.0];
    } else {
        if (input[43] <= -0.34847529232501984) {
            if (input[23] <= -0.17039640480652452) {
                if (input[42] <= -0.000999697484076023) {
                    if (input[3] <= 0.6659103035926819) {
                        var6 = [1.0, 0.0];
                    } else {
                        if (input[12] <= 0.43214040994644165) {
                            if (input[41] <= -0.07791388034820557) {
                                if (input[28] <= 0.5) {
                                    var6 = [0.2682926829268293, 0.7317073170731707];
                                } else {
                                    var6 = [0.0, 1.0];
                                }
                            } else {
                                var6 = [0.0, 1.0];
                            }
                        } else {
                            var6 = [1.0, 0.0];
                        }
                    }
                } else {
                    if (input[16] <= 0.4683411903679371) {
                        if (input[3] <= 0.4847452864050865) {
                            var6 = [1.0, 0.0];
                        } else {
                            var6 = [0.0, 1.0];
                        }
                    } else {
                        var6 = [1.0, 0.0];
                    }
                }
            } else {
                if (input[13] <= 0.7343381941318512) {
                    if (input[3] <= 0.6535632014274597) {
                        var6 = [1.0, 0.0];
                    } else {
                        if (input[40] <= -0.32206469774246216) {
                            if (input[28] <= 0.5) {
                                if (input[45] <= -0.2966116815805435) {
                                    var6 = [0.5, 0.5];
                                } else {
                                    var6 = [0.0, 1.0];
                                }
                            } else {
                                var6 = [0.0, 1.0];
                            }
                        } else {
                            var6 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[41] <= -0.03784236125648022) {
                        if (input[3] <= 0.5492892265319824) {
                            var6 = [1.0, 0.0];
                        } else {
                            if (input[39] <= 0.5) {
                                if (input[20] <= -0.9511148929595947) {
                                    var6 = [0.0, 1.0];
                                } else {
                                    var6 = [1.0, 0.0];
                                }
                            } else {
                                var6 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[20] <= -0.2565183490514755) {
                            if (input[3] <= -0.08686265349388123) {
                                var6 = [1.0, 0.0];
                            } else {
                                var6 = [0.0, 1.0];
                            }
                        } else {
                            if (input[0] <= 0.15556214191019535) {
                                var6 = [1.0, 0.0];
                            } else {
                                var6 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        } else {
            if (input[8] <= -1.1061445325613022) {
                var6 = [1.0, 0.0];
            } else {
                if (input[12] <= 0.6212070882320404) {
                    if (input[14] <= 0.2203299105167389) {
                        if (input[43] <= -0.33715586364269257) {
                            if (input[20] <= 1.7964003086090088) {
                                if (input[19] <= 0.3287796899676323) {
                                    var6 = [0.016276041666666668, 0.9837239583333334];
                                } else {
                                    var6 = [0.7692307692307693, 0.23076923076923078];
                                }
                            } else {
                                if (input[31] <= 0.5) {
                                    var6 = [0.9743589743589743, 0.02564102564102564];
                                } else {
                                    var6 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[23] <= -0.9084997177124023) {
                                if (input[43] <= -0.31451697647571564) {
                                    var6 = [0.05465587044534413, 0.9453441295546559];
                                } else {
                                    var6 = [0.00042162071000927565, 0.9995783792899907];
                                }
                            } else {
                                if (input[19] <= -0.23817865550518036) {
                                    var6 = [0.000014700909986328154, 0.9999852990900137];
                                } else {
                                    var6 = [0.00018556318426424197, 0.9998144368157358];
                                }
                            }
                        }
                    } else {
                        if (input[39] <= 0.5) {
                            if (input[40] <= -0.22132381796836853) {
                                var6 = [1.0, 0.0];
                            } else {
                                var6 = [0.0, 1.0];
                            }
                        } else {
                            if (input[4] <= -0.6317702233791351) {
                                var6 = [1.0, 0.0];
                            } else {
                                var6 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[24] <= -0.024782836437225342) {
                        var6 = [1.0, 0.0];
                    } else {
                        var6 = [0.0, 1.0];
                    }
                }
            }
        }
    }
    var var7;
    if (input[45] <= -0.2779812812805176) {
        if (input[28] <= 0.5) {
            if (input[3] <= 0.6926922798156738) {
                var7 = [1.0, 0.0];
            } else {
                if (input[40] <= -0.32206469774246216) {
                    if (input[32] <= 0.5) {
                        if (input[39] <= 0.5) {
                            if (input[41] <= -0.05787811987102032) {
                                if (input[42] <= -0.2923208698630333) {
                                    var7 = [1.0, 0.0];
                                } else {
                                    var7 = [0.0, 1.0];
                                }
                            } else {
                                var7 = [0.0, 1.0];
                            }
                        } else {
                            if (input[26] <= 0.5) {
                                if (input[43] <= -0.3597947508096695) {
                                    var7 = [0.25, 0.75];
                                } else {
                                    var7 = [0.0, 1.0];
                                }
                            } else {
                                if (input[12] <= 0.002443456556648016) {
                                    var7 = [0.0, 1.0];
                                } else {
                                    var7 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        var7 = [0.0, 1.0];
                    }
                } else {
                    if (input[0] <= -0.43745623528957367) {
                        var7 = [1.0, 0.0];
                    } else {
                        if (input[0] <= 0.21607422828674316) {
                            if (input[45] <= -0.3028218150138855) {
                                if (input[21] <= 0.5) {
                                    var7 = [1.0, 0.0];
                                } else {
                                    var7 = [0.02666666666666667, 0.9733333333333334];
                                }
                            } else {
                                if (input[22] <= 0.5) {
                                    var7 = [0.3333333333333333, 0.6666666666666666];
                                } else {
                                    var7 = [0.005477308294209703, 0.9945226917057903];
                                }
                            }
                        } else {
                            if (input[7] <= -0.44095367193222046) {
                                var7 = [0.0, 1.0];
                            } else {
                                var7 = [1.0, 0.0];
                            }
                        }
                    }
                }
            }
        } else {
            if (input[21] <= 0.5) {
                var7 = [1.0, 0.0];
            } else {
                if (input[40] <= -0.29687948524951935) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[3] <= 0.5110007375478745) {
                            var7 = [1.0, 0.0];
                        } else {
                            var7 = [0.0, 1.0];
                        }
                    } else {
                        if (input[43] <= -0.32017670571804047) {
                            if (input[3] <= 0.4847452864050865) {
                                var7 = [1.0, 0.0];
                            } else {
                                var7 = [0.0, 1.0];
                            }
                        } else {
                            if (input[5] <= -1.0307052731513977) {
                                if (input[40] <= -0.32206469774246216) {
                                    var7 = [0.0, 1.0];
                                } else {
                                    var7 = [1.0, 0.0];
                                }
                            } else {
                                if (input[3] <= -0.40871644020080566) {
                                    var7 = [1.0, 0.0];
                                } else {
                                    var7 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[8] <= -1.1061445325613022) {
                        var7 = [1.0, 0.0];
                    } else {
                        if (input[0] <= 0.1676645651459694) {
                            if (input[3] <= 0.3858481468632817) {
                                var7 = [1.0, 0.0];
                            } else {
                                var7 = [0.0, 1.0];
                            }
                        } else {
                            if (input[42] <= -0.000999697484076023) {
                                if (input[43] <= -0.28055865317583084) {
                                    var7 = [0.9629629629629629, 0.037037037037037035];
                                } else {
                                    var7 = [0.0, 1.0];
                                }
                            } else {
                                var7 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[13] <= 1.358758270740509) {
            if (input[21] <= 0.5) {
                var7 = [1.0, 0.0];
            } else {
                if (input[14] <= 0.2203299105167389) {
                    if (input[19] <= 0.6122588664293289) {
                        if (input[1] <= -1.4174097776412964) {
                            var7 = [1.0, 0.0];
                        } else {
                            if (input[43] <= -0.36545446515083313) {
                                if (input[13] <= 0.9208532571792603) {
                                    var7 = [0.040437823046518696, 0.9595621769534813];
                                } else {
                                    var7 = [0.6170212765957447, 0.3829787234042553];
                                }
                            } else {
                                if (input[19] <= 0.3287796899676323) {
                                    var7 = [0.0008164252668501103, 0.9991835747331499];
                                } else {
                                    var7 = [0.08296943231441048, 0.9170305676855895];
                                }
                            }
                        }
                    } else {
                        if (input[13] <= 0.6613540053367615) {
                            if (input[40] <= -0.1268792301416397) {
                                if (input[30] <= 3.0) {
                                    var7 = [1.0, 0.0];
                                } else {
                                    var7 = [0.0, 1.0];
                                }
                            } else {
                                var7 = [0.0, 1.0];
                            }
                        } else {
                            if (input[23] <= -0.0026456117630004883) {
                                if (input[40] <= 0.20052867010235786) {
                                    var7 = [0.9495798319327731, 0.05042016806722689];
                                } else {
                                    var7 = [0.0, 1.0];
                                }
                            } else {
                                var7 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[3] <= 0.3865265743806958) {
                        var7 = [1.0, 0.0];
                    } else {
                        var7 = [0.0, 1.0];
                    }
                }
            }
        } else {
            if (input[1] <= 1.4238800406455994) {
                var7 = [1.0, 0.0];
            } else {
                if (input[20] <= -1.1826470494270325) {
                    var7 = [0.0, 1.0];
                } else {
                    if (input[43] <= -0.30319754779338837) {
                        if (input[1] <= 1.9702818989753723) {
                            if (input[3] <= 0.031295210123062134) {
                                var7 = [1.0, 0.0];
                            } else {
                                var7 = [0.0, 1.0];
                            }
                        } else {
                            var7 = [1.0, 0.0];
                        }
                    } else {
                        if (input[14] <= -0.032053641974925995) {
                            var7 = [0.0, 1.0];
                        } else {
                            var7 = [1.0, 0.0];
                        }
                    }
                }
            }
        }
    }
    var var8;
    if (input[40] <= -0.30947208404541016) {
        if (input[3] <= 0.6926922798156738) {
            var8 = [1.0, 0.0];
        } else {
            if (input[28] <= 0.5) {
                if (input[41] <= -0.07791388034820557) {
                    if (input[33] <= 0.5) {
                        if (input[39] <= 0.5) {
                            if (input[25] <= 0.5) {
                                if (input[37] <= 0.5) {
                                    var8 = [0.9981981981981982, 0.0018018018018018018];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            } else {
                                if (input[13] <= -0.16579985618591309) {
                                    var8 = [1.0, 0.0];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[7] <= -0.44095367193222046) {
                                var8 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.2966116815805435) {
                                    var8 = [1.0, 0.0];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        var8 = [0.0, 1.0];
                    }
                } else {
                    if (input[6] <= 1.87261563539505) {
                        if (input[21] <= 0.5) {
                            var8 = [1.0, 0.0];
                        } else {
                            if (input[42] <= -0.3595488369464874) {
                                if (input[43] <= -0.3597947508096695) {
                                    var8 = [0.15, 0.85];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var8 = [1.0, 0.0];
                                } else {
                                    var8 = [0.0015625, 0.9984375];
                                }
                            }
                        }
                    } else {
                        if (input[33] <= 0.5) {
                            if (input[12] <= -0.42725351452827454) {
                                var8 = [0.0, 1.0];
                            } else {
                                var8 = [1.0, 0.0];
                            }
                        } else {
                            var8 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[43] <= -0.36545446515083313) {
                    if (input[41] <= -0.07791388034820557) {
                        if (input[30] <= 0.5) {
                            if (input[1] <= -0.8710078597068787) {
                                var8 = [1.0, 0.0];
                            } else {
                                if (input[26] <= 0.5) {
                                    var8 = [0.7142857142857143, 0.2857142857142857];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var8 = [0.0, 1.0];
                        }
                    } else {
                        var8 = [0.0, 1.0];
                    }
                } else {
                    if (input[1] <= -1.0895686149597168) {
                        if (input[30] <= 0.5) {
                            if (input[45] <= -0.1848292574286461) {
                                var8 = [0.0, 1.0];
                            } else {
                                if (input[44] <= -0.1064351461827755) {
                                    var8 = [0.3333333333333333, 0.6666666666666666];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var8 = [0.0, 1.0];
                        }
                    } else {
                        if (input[19] <= -0.23817865550518036) {
                            var8 = [0.0, 1.0];
                        } else {
                            if (input[1] <= 0.11251552402973175) {
                                var8 = [0.0, 1.0];
                            } else {
                                if (input[41] <= -0.07791388034820557) {
                                    var8 = [0.25, 0.75];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[16] <= 0.4683411903679371) {
            if (input[0] <= 0.09505007043480873) {
                if (input[21] <= 0.5) {
                    var8 = [1.0, 0.0];
                } else {
                    if (input[15] <= 1.7826361656188965) {
                        if (input[3] <= 0.6659103035926819) {
                            var8 = [1.0, 0.0];
                        } else {
                            var8 = [0.0, 1.0];
                        }
                    } else {
                        if (input[0] <= -0.14699824899435043) {
                            if (input[24] <= 0.8877286314964294) {
                                if (input[14] <= 0.30445775389671326) {
                                    var8 = [0.16666666666666666, 0.8333333333333334];
                                } else {
                                    var8 = [1.0, 0.0];
                                }
                            } else {
                                if (input[20] <= 1.0709328055381775) {
                                    var8 = [0.012658227848101266, 0.9873417721518988];
                                } else {
                                    var8 = [1.0, 0.0];
                                }
                            }
                        } else {
                            if (input[14] <= 0.30445775389671326) {
                                if (input[41] <= -0.031163774197921157) {
                                    var8 = [1.0, 0.0];
                                } else {
                                    var8 = [0.0, 1.0];
                                }
                            } else {
                                var8 = [1.0, 0.0];
                            }
                        }
                    }
                }
            } else {
                if (input[3] <= 0.5361550152301788) {
                    var8 = [1.0, 0.0];
                } else {
                    var8 = [0.0, 1.0];
                }
            }
        } else {
            var8 = [1.0, 0.0];
        }
    }
    var var9;
    if (input[39] <= 0.5) {
        if (input[25] <= 0.5) {
            if (input[23] <= -0.24228958785533905) {
                if (input[29] <= 0.07188555970788002) {
                    if (input[3] <= 0.6904508173465729) {
                        var9 = [1.0, 0.0];
                    } else {
                        if (input[40] <= -0.32206469774246216) {
                            if (input[45] <= -0.29350660741329193) {
                                var9 = [1.0, 0.0];
                            } else {
                                var9 = [0.0, 1.0];
                            }
                        } else {
                            if (input[1] <= 1.5331604480743408) {
                                if (input[21] <= 0.5) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                if (input[37] <= 0.5) {
                                    var9 = [0.3333333333333333, 0.6666666666666666];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[41] <= -0.07791388034820557) {
                        var9 = [1.0, 0.0];
                    } else {
                        if (input[3] <= 0.31483445316553116) {
                            var9 = [1.0, 0.0];
                        } else {
                            var9 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[3] <= 0.6659103035926819) {
                    var9 = [1.0, 0.0];
                } else {
                    if (input[45] <= -0.2966116815805435) {
                        if (input[0] <= -0.26802240312099457) {
                            if (input[36] <= 0.5) {
                                if (input[33] <= 0.5) {
                                    var9 = [0.891213389121339, 0.1087866108786611];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                var9 = [0.0, 1.0];
                            }
                        } else {
                            if (input[27] <= 0.5) {
                                if (input[40] <= -0.32206469774246216) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                if (input[21] <= 0.5) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.03424657534246575, 0.9657534246575342];
                                }
                            }
                        }
                    } else {
                        if (input[45] <= -0.28419141471385956) {
                            if (input[23] <= 0.2489804751239717) {
                                var9 = [1.0, 0.0];
                            } else {
                                if (input[40] <= -0.30947208404541016) {
                                    var9 = [0.1794871794871795, 0.8205128205128205];
                                } else {
                                    var9 = [0.006779661016949152, 0.9932203389830508];
                                }
                            }
                        } else {
                            if (input[41] <= -0.07791388034820557) {
                                if (input[30] <= 0.5) {
                                    var9 = [0.0058823529411764705, 0.9941176470588236];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                if (input[42] <= -0.4491861164569855) {
                                    var9 = [0.011235955056179775, 0.9887640449438202];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[15] <= -0.32605210691690445) {
                if (input[40] <= -0.29687948524951935) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[33] <= 0.5) {
                            if (input[3] <= 0.5950546413660049) {
                                var9 = [1.0, 0.0];
                            } else {
                                if (input[28] <= 0.5) {
                                    var9 = [0.02666666666666667, 0.9733333333333334];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[3] <= 0.5072638541460037) {
                                var9 = [1.0, 0.0];
                            } else {
                                var9 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[42] <= 0.22309350967407227) {
                            if (input[42] <= -0.3595488369464874) {
                                if (input[41] <= -0.06455670669674873) {
                                    var9 = [0.9912390488110138, 0.008760951188986232];
                                } else {
                                    var9 = [0.9428571428571428, 0.05714285714285714];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.6739130434782609, 0.32608695652173914];
                                }
                            }
                        } else {
                            var9 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[32] <= 0.5) {
                        if (input[43] <= -0.36545446515083313) {
                            if (input[45] <= -0.14135830849409103) {
                                if (input[3] <= 0.5749632567167282) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                var9 = [0.0, 1.0];
                            }
                        } else {
                            if (input[43] <= -0.34847529232501984) {
                                if (input[45] <= -0.25935088098049164) {
                                    var9 = [0.3953488372093023, 0.6046511627906976];
                                } else {
                                    var9 = [0.10416666666666667, 0.8958333333333334];
                                }
                            } else {
                                if (input[20] <= 1.827271282672882) {
                                    var9 = [0.010769230769230769, 0.9892307692307692];
                                } else {
                                    var9 = [0.35, 0.65];
                                }
                            }
                        }
                    } else {
                        if (input[42] <= -0.3147301971912384) {
                            if (input[0] <= 0.5065322071313858) {
                                if (input[3] <= 0.6174811571836472) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                var9 = [1.0, 0.0];
                            }
                        } else {
                            var9 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[45] <= -0.24693060666322708) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[40] <= -0.29687948524951935) {
                            if (input[43] <= -0.3597947508096695) {
                                if (input[3] <= 0.5334799438714981) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                var9 = [1.0, 0.0];
                            }
                        } else {
                            if (input[15] <= 0.1182534508407116) {
                                if (input[37] <= 0.5) {
                                    var9 = [0.0, 1.0];
                                } else {
                                    var9 = [1.0, 0.0];
                                }
                            } else {
                                if (input[21] <= 0.5) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[6] <= 0.8903712928295135) {
                            if (input[23] <= 0.16510513424873352) {
                                if (input[45] <= -0.25935088098049164) {
                                    var9 = [0.9997118985883031, 0.0002881014116969173];
                                } else {
                                    var9 = [0.9876543209876543, 0.012345679012345678];
                                }
                            } else {
                                if (input[43] <= -0.2720690667629242) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[1] <= -0.4885265529155731) {
                                var9 = [0.0, 1.0];
                            } else {
                                var9 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[23] <= -0.3381471484899521) {
                        if (input[40] <= -0.038730951957404613) {
                            if (input[14] <= 0.052074207458645105) {
                                if (input[3] <= 0.18093225359916687) {
                                    var9 = [1.0, 0.0];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            } else {
                                if (input[19] <= -0.23817865550518036) {
                                    var9 = [0.0, 1.0];
                                } else {
                                    var9 = [0.971830985915493, 0.028169014084507043];
                                }
                            }
                        } else {
                            if (input[5] <= 0.06083625555038452) {
                                if (input[24] <= -0.04278385639190674) {
                                    var9 = [0.0, 1.0];
                                } else {
                                    var9 = [1.0, 0.0];
                                }
                            } else {
                                if (input[45] <= -0.23140526562929153) {
                                    var9 = [0.0, 1.0];
                                } else {
                                    var9 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[5] <= -1.03425794839859) {
                            if (input[0] <= -0.29222724586725235) {
                                var9 = [0.0, 1.0];
                            } else {
                                var9 = [1.0, 0.0];
                            }
                        } else {
                            if (input[0] <= -0.24381757527589798) {
                                var9 = [0.0, 1.0];
                            } else {
                                if (input[32] <= 0.5) {
                                    var9 = [0.14285714285714285, 0.8571428571428571];
                                } else {
                                    var9 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[14] <= 0.2203299105167389) {
            if (input[21] <= 0.5) {
                var9 = [1.0, 0.0];
            } else {
                if (input[3] <= 0.5913675278425217) {
                    var9 = [1.0, 0.0];
                } else {
                    if (input[22] <= 0.5) {
                        if (input[45] <= -0.2966116815805435) {
                            if (input[43] <= -0.36545446515083313) {
                                var9 = [1.0, 0.0];
                            } else {
                                var9 = [0.0, 1.0];
                            }
                        } else {
                            var9 = [0.0, 1.0];
                        }
                    } else {
                        var9 = [0.0, 1.0];
                    }
                }
            }
        } else {
            if (input[3] <= 0.34198704175651073) {
                var9 = [1.0, 0.0];
            } else {
                var9 = [0.0, 1.0];
            }
        }
    }
    var var10;
    if (input[3] <= 0.6926922798156738) {
        var10 = [1.0, 0.0];
    } else {
        if (input[1] <= -1.4174097776412964) {
            if (input[14] <= -0.11618148908019066) {
                var10 = [1.0, 0.0];
            } else {
                var10 = [0.0, 1.0];
            }
        } else {
            if (input[43] <= -0.3597947508096695) {
                if (input[39] <= 0.5) {
                    if (input[40] <= -0.32206469774246216) {
                        if (input[30] <= 0.5) {
                            if (input[25] <= 0.5) {
                                if (input[0] <= -0.31643207371234894) {
                                    var10 = [1.0, 0.0];
                                } else {
                                    var10 = [0.9767441860465116, 0.023255813953488372];
                                }
                            } else {
                                if (input[26] <= 0.5) {
                                    var10 = [0.1, 0.9];
                                } else {
                                    var10 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[23] <= -0.0026456117630004883) {
                                var10 = [1.0, 0.0];
                            } else {
                                if (input[1] <= -0.10604522936046124) {
                                    var10 = [0.0, 1.0];
                                } else {
                                    var10 = [0.5, 0.5];
                                }
                            }
                        }
                    } else {
                        if (input[33] <= 0.5) {
                            if (input[42] <= -0.4491861164569855) {
                                if (input[41] <= -0.07791388034820557) {
                                    var10 = [0.5625, 0.4375];
                                } else {
                                    var10 = [0.05128205128205128, 0.9487179487179487];
                                }
                            } else {
                                if (input[5] <= 0.16298678517341614) {
                                    var10 = [0.0, 1.0];
                                } else {
                                    var10 = [0.0136986301369863, 0.9863013698630136];
                                }
                            }
                        } else {
                            var10 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[42] <= -0.4491861164569855) {
                        if (input[13] <= 0.7789396345615387) {
                            if (input[21] <= 0.5) {
                                var10 = [1.0, 0.0];
                            } else {
                                var10 = [0.0, 1.0];
                            }
                        } else {
                            if (input[27] <= 0.5) {
                                var10 = [0.0, 1.0];
                            } else {
                                var10 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[21] <= 0.5) {
                            var10 = [1.0, 0.0];
                        } else {
                            if (input[40] <= -0.30947208404541016) {
                                if (input[22] <= 0.5) {
                                    var10 = [0.5, 0.5];
                                } else {
                                    var10 = [0.0, 1.0];
                                }
                            } else {
                                var10 = [0.0, 1.0];
                            }
                        }
                    }
                }
            } else {
                if (input[30] <= 0.5) {
                    if (input[41] <= -0.07791388034820557) {
                        if (input[12] <= -0.4100656360387802) {
                            if (input[6] <= 3.1054047346115112) {
                                if (input[5] <= -1.0335662961006165) {
                                    var10 = [0.2222222222222222, 0.7777777777777778];
                                } else {
                                    var10 = [0.0, 1.0];
                                }
                            } else {
                                var10 = [1.0, 0.0];
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var10 = [1.0, 0.0];
                            } else {
                                var10 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[44] <= -0.1064351461827755) {
                            if (input[1] <= -0.21532560884952545) {
                                if (input[42] <= -0.4491861164569855) {
                                    var10 = [0.014084507042253521, 0.9859154929577465];
                                } else {
                                    var10 = [0.00016178611875101117, 0.9998382138812489];
                                }
                            } else {
                                var10 = [0.0, 1.0];
                            }
                        } else {
                            var10 = [0.0, 1.0];
                        }
                    }
                } else {
                    var10 = [0.0, 1.0];
                }
            }
        }
    }
    var var11;
    if (input[34] <= 0.5) {
        if (input[20] <= -0.30282479524612427) {
            if (input[41] <= -0.07791388034820557) {
                if (input[43] <= -0.3597947508096695) {
                    if (input[39] <= 0.5) {
                        if (input[25] <= 0.5) {
                            if (input[23] <= 0.6321384608745575) {
                                var11 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.2779812812805176) {
                                    var11 = [0.9987855234393976, 0.0012144765606023804];
                                } else {
                                    var11 = [0.2413793103448276, 0.7586206896551724];
                                }
                            }
                        } else {
                            if (input[40] <= -0.28428688645362854) {
                                if (input[1] <= 0.44035664200782776) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.8823529411764706, 0.11764705882352941];
                                }
                            } else {
                                if (input[40] <= -0.26539796590805054) {
                                    var11 = [0.5, 0.5];
                                } else {
                                    var11 = [0.034482758620689655, 0.9655172413793104];
                                }
                            }
                        }
                    } else {
                        if (input[14] <= 0.052074207458645105) {
                            if (input[28] <= 0.5) {
                                if (input[37] <= 0.5) {
                                    var11 = [0.6022727272727273, 0.3977272727272727];
                                } else {
                                    var11 = [0.3076923076923077, 0.6923076923076923];
                                }
                            } else {
                                if (input[0] <= -0.24381757527589798) {
                                    var11 = [0.2037037037037037, 0.7962962962962963];
                                } else {
                                    var11 = [0.02830188679245283, 0.9716981132075472];
                                }
                            }
                        } else {
                            var11 = [1.0, 0.0];
                        }
                    }
                } else {
                    if (input[3] <= 0.6116466224193573) {
                        var11 = [1.0, 0.0];
                    } else {
                        if (input[32] <= 0.5) {
                            if (input[40] <= -0.30947208404541016) {
                                if (input[45] <= -0.28419141471385956) {
                                    var11 = [0.6666666666666666, 0.3333333333333333];
                                } else {
                                    var11 = [0.0, 1.0];
                                }
                            } else {
                                var11 = [0.0, 1.0];
                            }
                        } else {
                            var11 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[8] <= -1.1061445325613022) {
                    var11 = [1.0, 0.0];
                } else {
                    if (input[12] <= 0.4837040454149246) {
                        if (input[43] <= -0.34847529232501984) {
                            if (input[21] <= 0.5) {
                                var11 = [1.0, 0.0];
                            } else {
                                if (input[42] <= -0.40436747670173645) {
                                    var11 = [0.16058394160583941, 0.8394160583941606];
                                } else {
                                    var11 = [0.005575266092245312, 0.9944247339077547];
                                }
                            }
                        } else {
                            if (input[25] <= 0.5) {
                                if (input[21] <= 0.5) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.00008972633467922836, 0.9999102736653208];
                                }
                            } else {
                                if (input[15] <= 2.8898738622665405) {
                                    var11 = [0.0005368326871458396, 0.9994631673128541];
                                } else {
                                    var11 = [0.875, 0.125];
                                }
                            }
                        }
                    } else {
                        if (input[20] <= -1.244388997554779) {
                            if (input[12] <= 7.977618992328644) {
                                var11 = [0.0, 1.0];
                            } else {
                                var11 = [1.0, 0.0];
                            }
                        } else {
                            var11 = [1.0, 0.0];
                        }
                    }
                }
            }
        } else {
            if (input[3] <= 0.6926922798156738) {
                var11 = [1.0, 0.0];
            } else {
                if (input[22] <= 0.5) {
                    if (input[39] <= 0.5) {
                        if (input[45] <= -0.25935086607933044) {
                            var11 = [1.0, 0.0];
                        } else {
                            var11 = [0.0, 1.0];
                        }
                    } else {
                        if (input[33] <= 0.5) {
                            var11 = [0.0, 1.0];
                        } else {
                            if (input[42] <= -0.24750222638249397) {
                                var11 = [1.0, 0.0];
                            } else {
                                var11 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[44] <= -0.1064351461827755) {
                        if (input[25] <= 0.5) {
                            if (input[39] <= 0.5) {
                                if (input[21] <= 0.5) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.0031007751937984496, 0.9968992248062015];
                                }
                            } else {
                                if (input[8] <= 0.5578374415636063) {
                                    var11 = [0.014796547472256474, 0.9852034525277436];
                                } else {
                                    var11 = [0.0010504201680672268, 0.9989495798319328];
                                }
                            }
                        } else {
                            if (input[6] <= -1.5321118831634521) {
                                if (input[26] <= 0.5) {
                                    var11 = [0.0, 1.0];
                                } else {
                                    var11 = [0.02702702702702703, 0.972972972972973];
                                }
                            } else {
                                var11 = [0.0, 1.0];
                            }
                        }
                    } else {
                        var11 = [0.0, 1.0];
                    }
                }
            }
        }
    } else {
        if (input[12] <= 0.3118252605199814) {
            if (input[15] <= -0.24847494810819626) {
                if (input[42] <= -0.4491861164569855) {
                    if (input[13] <= 0.30859722197055817) {
                        if (input[43] <= -0.35413502156734467) {
                            if (input[28] <= 0.5) {
                                if (input[40] <= -0.29687948524951935) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.07142857142857142, 0.9285714285714286];
                                }
                            } else {
                                if (input[41] <= -0.05119953490793705) {
                                    var11 = [0.13043478260869565, 0.8695652173913043];
                                } else {
                                    var11 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[8] <= -1.1061445325613022) {
                                var11 = [1.0, 0.0];
                            } else {
                                var11 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[3] <= 0.6129107773303986) {
                            var11 = [1.0, 0.0];
                        } else {
                            var11 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[41] <= -0.05119953490793705) {
                        if (input[3] <= 0.5839933156967163) {
                            var11 = [1.0, 0.0];
                        } else {
                            if (input[1] <= -0.21532560884952545) {
                                if (input[21] <= 0.5) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.0, 1.0];
                                }
                            } else {
                                if (input[43] <= -0.36545446515083313) {
                                    var11 = [0.015873015873015872, 0.9841269841269841];
                                } else {
                                    var11 = [0.00022655188038060717, 0.9997734481196194];
                                }
                            }
                        }
                    } else {
                        if (input[40] <= -0.28428688645362854) {
                            if (input[19] <= 0.3287796899676323) {
                                if (input[13] <= 1.0222201347351074) {
                                    var11 = [0.012858555885262116, 0.9871414441147379];
                                } else {
                                    var11 = [0.515625, 0.484375];
                                }
                            } else {
                                if (input[42] <= 0.2679121494293213) {
                                    var11 = [0.8461538461538461, 0.15384615384615385];
                                } else {
                                    var11 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[8] <= -1.1061445325613022) {
                                var11 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.2779812812805176) {
                                    var11 = [0.008174386920980926, 0.9918256130790191];
                                } else {
                                    var11 = [0.00027449324324324326, 0.9997255067567568];
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[32] <= 0.5) {
                    if (input[3] <= 0.6223016083240509) {
                        var11 = [1.0, 0.0];
                    } else {
                        var11 = [0.0, 1.0];
                    }
                } else {
                    if (input[40] <= -0.30947208404541016) {
                        if (input[45] <= -0.06994174793362617) {
                            if (input[19] <= -0.23817865550518036) {
                                var11 = [0.0, 1.0];
                            } else {
                                if (input[44] <= 0.006936721503734589) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var11 = [0.0, 1.0];
                        }
                    } else {
                        if (input[8] <= -1.1061445325613022) {
                            var11 = [1.0, 0.0];
                        } else {
                            if (input[12] <= -0.5475686490535736) {
                                if (input[21] <= 0.5) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.0, 1.0];
                                }
                            } else {
                                var11 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        } else {
            if (input[40] <= -0.19613859057426453) {
                if (input[12] <= 0.5524555742740631) {
                    if (input[14] <= -0.11618148908019066) {
                        if (input[20] <= -0.580663412809372) {
                            if (input[40] <= -0.30947208404541016) {
                                if (input[13] <= 1.8899207711219788) {
                                    var11 = [0.25, 0.75];
                                } else {
                                    var11 = [1.0, 0.0];
                                }
                            } else {
                                if (input[21] <= 0.5) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.034482758620689655, 0.9655172413793104];
                                }
                            }
                        } else {
                            if (input[13] <= 1.1357510685920715) {
                                if (input[1] <= 0.9867585152387619) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.0, 1.0];
                                }
                            } else {
                                if (input[42] <= 0.9177824258804321) {
                                    var11 = [1.0, 0.0];
                                } else {
                                    var11 = [0.6666666666666666, 0.3333333333333333];
                                }
                            }
                        }
                    } else {
                        var11 = [1.0, 0.0];
                    }
                } else {
                    if (input[3] <= 0.007900625467300415) {
                        var11 = [1.0, 0.0];
                    } else {
                        var11 = [0.0, 1.0];
                    }
                }
            } else {
                if (input[3] <= -0.17604869604110718) {
                    var11 = [1.0, 0.0];
                } else {
                    var11 = [0.0, 1.0];
                }
            }
        }
    }
    var var12;
    if (input[45] <= -0.2779812812805176) {
        if (input[39] <= 0.5) {
            if (input[44] <= -0.1064351461827755) {
                if (input[43] <= -0.3428155779838562) {
                    if (input[23] <= 0.05327127268537879) {
                        if (input[30] <= 0.5) {
                            if (input[42] <= 0.17827486991882324) {
                                if (input[0] <= -0.05017892271280289) {
                                    var12 = [0.9983667855761379, 0.0016332144238621223];
                                } else {
                                    var12 = [0.9998872826668921, 0.00011271733310789867];
                                }
                            } else {
                                if (input[7] <= -0.44095367193222046) {
                                    var12 = [0.0, 1.0];
                                } else {
                                    var12 = [0.8, 0.2];
                                }
                            }
                        } else {
                            if (input[3] <= 0.6427658200263977) {
                                var12 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.3028218150138855) {
                                    var12 = [0.8125, 0.1875];
                                } else {
                                    var12 = [0.18181818181818182, 0.8181818181818182];
                                }
                            }
                        }
                    } else {
                        if (input[33] <= 0.5) {
                            if (input[3] <= 0.6693269312381744) {
                                var12 = [1.0, 0.0];
                            } else {
                                if (input[40] <= -0.32206469774246216) {
                                    var12 = [0.9825870646766169, 0.017412935323383085];
                                } else {
                                    var12 = [0.044534412955465584, 0.9554655870445344];
                                }
                            }
                        } else {
                            if (input[21] <= 0.5) {
                                var12 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.29040154814720154) {
                                    var12 = [0.7103448275862069, 0.2896551724137931];
                                } else {
                                    var12 = [0.26153846153846155, 0.7384615384615385];
                                }
                            }
                        }
                    }
                } else {
                    if (input[34] <= 0.5) {
                        if (input[8] <= -1.1061445325613022) {
                            var12 = [1.0, 0.0];
                        } else {
                            if (input[43] <= -0.3088572472333908) {
                                if (input[3] <= 0.5238934606313705) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                if (input[12] <= 0.1915101259946823) {
                                    var12 = [0.015625, 0.984375];
                                } else {
                                    var12 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[14] <= 0.4306495189666748) {
                            if (input[13] <= 2.1494200229644775) {
                                if (input[3] <= 0.34788286685943604) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                var12 = [1.0, 0.0];
                            }
                        } else {
                            var12 = [1.0, 0.0];
                        }
                    }
                }
            } else {
                if (input[12] <= 0.0711949709802866) {
                    if (input[41] <= -0.06455670669674873) {
                        if (input[42] <= -0.40436747670173645) {
                            if (input[28] <= 0.5) {
                                var12 = [1.0, 0.0];
                            } else {
                                if (input[45] <= -0.2997167557477951) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[24] <= -0.7148206532001495) {
                                if (input[43] <= -0.3060273975133896) {
                                    var12 = [0.7368421052631579, 0.2631578947368421];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.08823529411764706, 0.9117647058823529];
                                }
                            }
                        }
                    } else {
                        if (input[42] <= -0.40436747670173645) {
                            if (input[3] <= 0.46773459762334824) {
                                var12 = [1.0, 0.0];
                            } else {
                                var12 = [0.0, 1.0];
                            }
                        } else {
                            if (input[3] <= 0.552652895450592) {
                                var12 = [1.0, 0.0];
                            } else {
                                if (input[23] <= 0.6477111279964447) {
                                    var12 = [0.013333333333333334, 0.9866666666666667];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[30] <= 2.5) {
                        if (input[23] <= -0.04445430636405945) {
                            if (input[5] <= 0.16298678517341614) {
                                var12 = [1.0, 0.0];
                            } else {
                                if (input[3] <= 0.16378015279769897) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[45] <= -0.29040154814720154) {
                                if (input[3] <= 0.3209019564092159) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                var12 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[42] <= -0.1354556204751134) {
                            var12 = [1.0, 0.0];
                        } else {
                            if (input[6] <= -0.33413413166999817) {
                                var12 = [1.0, 0.0];
                            } else {
                                var12 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        } else {
            if (input[15] <= -0.36836692690849304) {
                if (input[13] <= 0.9208532571792603) {
                    if (input[3] <= 0.6420855820178986) {
                        var12 = [1.0, 0.0];
                    } else {
                        if (input[27] <= 0.5) {
                            if (input[1] <= -0.8710078597068787) {
                                if (input[5] <= -1.033695101737976) {
                                    var12 = [0.5, 0.5];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                if (input[34] <= 0.5) {
                                    var12 = [0.007739938080495356, 0.9922600619195047];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[42] <= -0.4491861164569855) {
                                if (input[19] <= -0.23817865550518036) {
                                    var12 = [0.038461538461538464, 0.9615384615384616];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                if (input[42] <= -0.3595488369464874) {
                                    var12 = [0.004424778761061947, 0.995575221238938];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[12] <= 0.08838284946978092) {
                            var12 = [1.0, 0.0];
                        } else {
                            if (input[5] <= -1.0242624878883362) {
                                var12 = [1.0, 0.0];
                            } else {
                                if (input[24] <= -0.04278385639190674) {
                                    var12 = [0.0, 1.0];
                                } else {
                                    var12 = [0.08450704225352113, 0.9154929577464789];
                                }
                            }
                        }
                    } else {
                        if (input[43] <= -0.16453437507152557) {
                            if (input[26] <= 0.5) {
                                if (input[20] <= -0.549792468547821) {
                                    var12 = [0.6666666666666666, 0.3333333333333333];
                                } else {
                                    var12 = [1.0, 0.0];
                                }
                            } else {
                                if (input[13] <= 1.0222201347351074) {
                                    var12 = [0.9090909090909091, 0.09090909090909091];
                                } else {
                                    var12 = [1.0, 0.0];
                                }
                            }
                        } else {
                            var12 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[43] <= -0.3088572472333908) {
                    if (input[12] <= -0.20381109416484833) {
                        if (input[23] <= -0.3381471484899521) {
                            if (input[3] <= 0.24261170625686646) {
                                var12 = [1.0, 0.0];
                            } else {
                                var12 = [0.0, 1.0];
                            }
                        } else {
                            if (input[20] <= -0.7195827066898346) {
                                if (input[21] <= 0.5) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                if (input[40] <= -0.26539796590805054) {
                                    var12 = [0.8507462686567164, 0.14925373134328357];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[8] <= 0.5578374415636063) {
                            if (input[19] <= -0.23817865550518036) {
                                if (input[42] <= -0.2923208698630333) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            } else {
                                var12 = [1.0, 0.0];
                            }
                        } else {
                            var12 = [1.0, 0.0];
                        }
                    }
                } else {
                    if (input[21] <= 0.5) {
                        var12 = [1.0, 0.0];
                    } else {
                        if (input[3] <= -0.623989999294281) {
                            var12 = [1.0, 0.0];
                        } else {
                            var12 = [0.0, 1.0];
                        }
                    }
                }
            }
        }
    } else {
        if (input[0] <= 0.26448388397693634) {
            if (input[19] <= 0.3287796899676323) {
                if (input[13] <= 1.5939294695854187) {
                    if (input[21] <= 0.5) {
                        var12 = [1.0, 0.0];
                    } else {
                        if (input[3] <= 0.5492892265319824) {
                            var12 = [1.0, 0.0];
                        } else {
                            var12 = [0.0, 1.0];
                        }
                    }
                } else {
                    var12 = [1.0, 0.0];
                }
            } else {
                if (input[42] <= -0.3147301971912384) {
                    if (input[40] <= -0.1331755369901657) {
                        if (input[15] <= -0.24142247810959816) {
                            if (input[8] <= 2.2218194007873535) {
                                if (input[26] <= 0.5) {
                                    var12 = [0.9879518072289156, 0.012048192771084338];
                                } else {
                                    var12 = [0.5714285714285714, 0.42857142857142855];
                                }
                            } else {
                                if (input[41] <= -0.02448518737219274) {
                                    var12 = [1.0, 0.0];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var12 = [1.0, 0.0];
                        }
                    } else {
                        if (input[6] <= -0.6058638319373131) {
                            var12 = [1.0, 0.0];
                        } else {
                            var12 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[14] <= -0.11618148908019066) {
                        if (input[5] <= -1.0269772410392761) {
                            if (input[43] <= -0.20132256299257278) {
                                var12 = [1.0, 0.0];
                            } else {
                                var12 = [0.0, 1.0];
                            }
                        } else {
                            if (input[40] <= -0.29687948524951935) {
                                if (input[13] <= 0.9776187539100647) {
                                    var12 = [0.0, 1.0];
                                } else {
                                    var12 = [1.0, 0.0];
                                }
                            } else {
                                if (input[45] <= -0.25935088098049164) {
                                    var12 = [0.5, 0.5];
                                } else {
                                    var12 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[43] <= -0.2607496380805969) {
                            var12 = [1.0, 0.0];
                        } else {
                            if (input[1] <= -0.21532560884952545) {
                                var12 = [1.0, 0.0];
                            } else {
                                var12 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        } else {
            if (input[19] <= -0.23817865550518036) {
                var12 = [0.0, 1.0];
            } else {
                if (input[3] <= 0.2916056737303734) {
                    var12 = [1.0, 0.0];
                } else {
                    var12 = [0.0, 1.0];
                }
            }
        }
    }
    var var13;
    if (input[21] <= 0.5) {
        var13 = [1.0, 0.0];
    } else {
        if (input[45] <= -0.2779812812805176) {
            if (input[28] <= 0.5) {
                if (input[19] <= -0.23817865550518036) {
                    if (input[41] <= -0.07791388034820557) {
                        if (input[25] <= 0.5) {
                            if (input[39] <= 0.5) {
                                if (input[37] <= 0.5) {
                                    var13 = [0.99836867862969, 0.0016313213703099511];
                                } else {
                                    var13 = [0.9090909090909091, 0.09090909090909091];
                                }
                            } else {
                                if (input[40] <= -0.32206469774246216) {
                                    var13 = [1.0, 0.0];
                                } else {
                                    var13 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[43] <= -0.36545446515083313) {
                                if (input[39] <= 0.5) {
                                    var13 = [1.0, 0.0];
                                } else {
                                    var13 = [0.0, 1.0];
                                }
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[3] <= 0.5922664552927017) {
                            var13 = [1.0, 0.0];
                        } else {
                            if (input[12] <= -0.031932301353663206) {
                                if (input[39] <= 0.5) {
                                    var13 = [0.006802721088435374, 0.9931972789115646];
                                } else {
                                    var13 = [0.0, 1.0];
                                }
                            } else {
                                if (input[41] <= -0.06455670669674873) {
                                    var13 = [0.17647058823529413, 0.8235294117647058];
                                } else {
                                    var13 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[43] <= -0.29753781855106354) {
                        if (input[39] <= 0.5) {
                            if (input[40] <= -0.013545729278121144) {
                                if (input[6] <= 1.4970347881317139) {
                                    var13 = [0.9993532835557646, 0.0006467164442354047];
                                } else {
                                    var13 = [0.9285714285714286, 0.07142857142857142];
                                }
                            } else {
                                if (input[14] <= -0.11618148908019066) {
                                    var13 = [0.2727272727272727, 0.7272727272727273];
                                } else {
                                    var13 = [1.0, 0.0];
                                }
                            }
                        } else {
                            if (input[0] <= -0.07438375428318977) {
                                if (input[3] <= 0.5684983134269714) {
                                    var13 = [1.0, 0.0];
                                } else {
                                    var13 = [0.012345679012345678, 0.9876543209876543];
                                }
                            } else {
                                if (input[20] <= 0.3145943582057953) {
                                    var13 = [0.9888888888888889, 0.011111111111111112];
                                } else {
                                    var13 = [0.9992464204973625, 0.0007535795026375283];
                                }
                            }
                        }
                    } else {
                        if (input[12] <= 0.2430737540125847) {
                            if (input[7] <= 1.226442813873291) {
                                if (input[4] <= 0.5356099009513855) {
                                    var13 = [0.0, 1.0];
                                } else {
                                    var13 = [1.0, 0.0];
                                }
                            } else {
                                var13 = [1.0, 0.0];
                            }
                        } else {
                            var13 = [1.0, 0.0];
                        }
                    }
                }
            } else {
                if (input[0] <= 0.04664040543138981) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[3] <= 0.5110007375478745) {
                            var13 = [1.0, 0.0];
                        } else {
                            var13 = [0.0, 1.0];
                        }
                    } else {
                        if (input[40] <= -0.32206469774246216) {
                            if (input[39] <= 0.5) {
                                var13 = [1.0, 0.0];
                            } else {
                                if (input[30] <= 0.5) {
                                    var13 = [0.7945205479452054, 0.2054794520547945];
                                } else {
                                    var13 = [0.3333333333333333, 0.6666666666666666];
                                }
                            }
                        } else {
                            if (input[3] <= 0.580042839050293) {
                                var13 = [1.0, 0.0];
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[3] <= 0.29462411999702454) {
                        var13 = [1.0, 0.0];
                    } else {
                        var13 = [0.0, 1.0];
                    }
                }
            }
        } else {
            if (input[13] <= 1.4155237674713135) {
                if (input[12] <= 0.3805767744779587) {
                    if (input[45] <= -0.25314074754714966) {
                        if (input[13] <= 1.0830403566360474) {
                            if (input[41] <= -0.07791388034820557) {
                                if (input[6] <= -0.48574022948741913) {
                                    var13 = [0.8243243243243243, 0.17567567567567569];
                                } else {
                                    var13 = [0.07894736842105263, 0.9210526315789473];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var13 = [1.0, 0.0];
                                } else {
                                    var13 = [0.009971721982437863, 0.9900282780175621];
                                }
                            }
                        } else {
                            if (input[23] <= -0.05422118306159973) {
                                if (input[43] <= -0.2409406080842018) {
                                    var13 = [1.0, 0.0];
                                } else {
                                    var13 = [0.0, 1.0];
                                }
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[0] <= 0.19186940044164658) {
                            if (input[3] <= 0.5226951539516449) {
                                var13 = [1.0, 0.0];
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        } else {
                            if (input[3] <= 0.13798262178897858) {
                                var13 = [1.0, 0.0];
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    if (input[40] <= -0.19613859057426453) {
                        if (input[42] <= 0.24550282955169678) {
                            if (input[43] <= -0.17868367582559586) {
                                var13 = [1.0, 0.0];
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        } else {
                            if (input[3] <= -0.4572798013687134) {
                                var13 = [1.0, 0.0];
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        }
                    } else {
                        if (input[3] <= -0.34654760360717773) {
                            var13 = [1.0, 0.0];
                        } else {
                            var13 = [0.0, 1.0];
                        }
                    }
                }
            } else {
                if (input[32] <= 0.5) {
                    if (input[13] <= 1.614202857017517) {
                        if (input[3] <= 0.1789589524269104) {
                            var13 = [1.0, 0.0];
                        } else {
                            var13 = [0.0, 1.0];
                        }
                    } else {
                        if (input[40] <= -0.0953977033495903) {
                            var13 = [1.0, 0.0];
                        } else {
                            if (input[17] <= 2.431964725255966) {
                                var13 = [0.0, 1.0];
                            } else {
                                var13 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[1] <= 1.5331604480743408) {
                        var13 = [1.0, 0.0];
                    } else {
                        if (input[5] <= -1.0322433114051819) {
                            var13 = [1.0, 0.0];
                        } else {
                            if (input[34] <= 0.5) {
                                if (input[24] <= -0.10038700699806213) {
                                    var13 = [0.0967741935483871, 0.9032258064516129];
                                } else {
                                    var13 = [0.0, 1.0];
                                }
                            } else {
                                var13 = [0.0, 1.0];
                            }
                        }
                    }
                }
            }
        }
    }
    var var14;
    if (input[43] <= -0.34847529232501984) {
        if (input[45] <= -0.2779812812805176) {
            if (input[32] <= 0.5) {
                if (input[4] <= 0.6410261988639832) {
                    if (input[28] <= 0.5) {
                        if (input[42] <= 0.13345622643828392) {
                            if (input[3] <= 0.6904508173465729) {
                                var14 = [1.0, 0.0];
                            } else {
                                if (input[37] <= 0.5) {
                                    var14 = [0.68, 0.32];
                                } else {
                                    var14 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[8] <= -1.1061445325613022) {
                                var14 = [1.0, 0.0];
                            } else {
                                if (input[19] <= 0.3287796899676323) {
                                    var14 = [0.0625, 0.9375];
                                } else {
                                    var14 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[29] <= 0.07188555970788002) {
                            if (input[20] <= 0.3454653173685074) {
                                if (input[1] <= 0.44035665690898895) {
                                    var14 = [0.6494845360824743, 0.35051546391752575];
                                } else {
                                    var14 = [0.9696969696969697, 0.030303030303030304];
                                }
                            } else {
                                if (input[7] <= -0.44095367193222046) {
                                    var14 = [0.8511326860841424, 0.1488673139158576];
                                } else {
                                    var14 = [0.9864636209813875, 0.01353637901861252];
                                }
                            }
                        } else {
                            if (input[15] <= 2.8334540724754333) {
                                var14 = [0.0, 1.0];
                            } else {
                                var14 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[42] <= -0.3595488369464874) {
                        if (input[25] <= 0.5) {
                            if (input[40] <= -0.23391643166542053) {
                                if (input[43] <= -0.35413502156734467) {
                                    var14 = [0.9961947941545561, 0.003805205845443873];
                                } else {
                                    var14 = [0.7631578947368421, 0.23684210526315788];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var14 = [1.0, 0.0];
                                } else {
                                    var14 = [0.09183673469387756, 0.9081632653061225];
                                }
                            }
                        } else {
                            if (input[20] <= -0.30282479524612427) {
                                if (input[27] <= 0.5) {
                                    var14 = [0.33980582524271846, 0.6601941747572816];
                                } else {
                                    var14 = [0.7821229050279329, 0.21787709497206703];
                                }
                            } else {
                                if (input[6] <= -3.5316139459609985) {
                                    var14 = [0.0, 1.0];
                                } else {
                                    var14 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[23] <= -0.17039640247821808) {
                            if (input[43] <= -0.36545446515083313) {
                                if (input[0] <= -0.05017892271280289) {
                                    var14 = [0.9142857142857143, 0.08571428571428572];
                                } else {
                                    var14 = [0.9985401459854014, 0.00145985401459854];
                                }
                            } else {
                                if (input[0] <= -0.025974091608077288) {
                                    var14 = [0.4883720930232558, 0.5116279069767442];
                                } else {
                                    var14 = [0.9857142857142858, 0.014285714285714285];
                                }
                            }
                        } else {
                            if (input[40] <= -0.25910165905952454) {
                                if (input[41] <= -0.03784236125648022) {
                                    var14 = [0.7586912065439673, 0.24130879345603273];
                                } else {
                                    var14 = [0.22018348623853212, 0.7798165137614679];
                                }
                            } else {
                                if (input[8] <= -1.1061445325613022) {
                                    var14 = [1.0, 0.0];
                                } else {
                                    var14 = [0.03672316384180791, 0.963276836158192];
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[14] <= -0.11618148908019066) {
                    if (input[19] <= -0.23817865550518036) {
                        if (input[45] <= -0.2966116815805435) {
                            if (input[25] <= 0.5) {
                                if (input[3] <= 0.4843149930238724) {
                                    var14 = [1.0, 0.0];
                                } else {
                                    var14 = [0.0, 1.0];
                                }
                            } else {
                                var14 = [0.0, 1.0];
                            }
                        } else {
                            if (input[41] <= 0.028943507000803947) {
                                if (input[21] <= 0.5) {
                                    var14 = [1.0, 0.0];
                                } else {
                                    var14 = [0.0, 1.0];
                                }
                            } else {
                                if (input[30] <= 0.5) {
                                    var14 = [0.17391304347826086, 0.8260869565217391];
                                } else {
                                    var14 = [0.0, 1.0];
                                }
                            }
                        }
                    } else {
                        if (input[41] <= -0.03784236125648022) {
                            if (input[8] <= -1.1061445325613022) {
                                var14 = [1.0, 0.0];
                            } else {
                                if (input[5] <= -0.9703989624977112) {
                                    var14 = [0.8695652173913043, 0.13043478260869565];
                                } else {
                                    var14 = [0.21739130434782608, 0.782608695652174];
                                }
                            }
                        } else {
                            var14 = [0.0, 1.0];
                        }
                    }
                } else {
                    if (input[0] <= -0.17120308429002762) {
                        if (input[3] <= 0.18093225359916687) {
                            var14 = [1.0, 0.0];
                        } else {
                            var14 = [0.0, 1.0];
                        }
                    } else {
                        var14 = [1.0, 0.0];
                    }
                }
            }
        } else {
            if (input[3] <= 0.6535632014274597) {
                var14 = [1.0, 0.0];
            } else {
                if (input[15] <= 0.36508987843990326) {
                    if (input[45] <= -0.25314074754714966) {
                        if (input[21] <= 0.5) {
                            var14 = [1.0, 0.0];
                        } else {
                            var14 = [0.0, 1.0];
                        }
                    } else {
                        var14 = [0.0, 1.0];
                    }
                } else {
                    if (input[42] <= -0.4491861164569855) {
                        if (input[27] <= 0.5) {
                            var14 = [0.0, 1.0];
                        } else {
                            var14 = [1.0, 0.0];
                        }
                    } else {
                        var14 = [0.0, 1.0];
                    }
                }
            }
        }
    } else {
        if (input[20] <= 1.7655293941497803) {
            if (input[1] <= -1.4174097776412964) {
                var14 = [1.0, 0.0];
            } else {
                if (input[0] <= 0.3370983898639679) {
                    if (input[42] <= -0.40436747670173645) {
                        if (input[14] <= 0.052074207458645105) {
                            if (input[3] <= 0.6116466224193573) {
                                var14 = [1.0, 0.0];
                            } else {
                                if (input[21] <= 0.5) {
                                    var14 = [1.0, 0.0];
                                } else {
                                    var14 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[45] <= -0.28108634054660797) {
                                var14 = [1.0, 0.0];
                            } else {
                                if (input[37] <= 0.5) {
                                    var14 = [0.0, 1.0];
                                } else {
                                    var14 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        if (input[8] <= -1.1061445325613022) {
                            var14 = [1.0, 0.0];
                        } else {
                            if (input[42] <= -0.3147301971912384) {
                                if (input[21] <= 0.5) {
                                    var14 = [1.0, 0.0];
                                } else {
                                    var14 = [0.0054806070826306915, 0.9945193929173693];
                                }
                            } else {
                                if (input[16] <= 0.4683411903679371) {
                                    var14 = [0.0008138374061374191, 0.9991861625938626];
                                } else {
                                    var14 = [1.0, 0.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[20] <= -1.2752599716186523) {
                        var14 = [0.0, 1.0];
                    } else {
                        if (input[8] <= 2.2218194007873535) {
                            if (input[19] <= 0.045300520956516266) {
                                if (input[33] <= 0.5) {
                                    var14 = [1.0, 0.0];
                                } else {
                                    var14 = [0.0, 1.0];
                                }
                            } else {
                                if (input[43] <= 0.05336486827582121) {
                                    var14 = [0.9914163090128756, 0.008583690987124463];
                                } else {
                                    var14 = [0.0, 1.0];
                                }
                            }
                        } else {
                            var14 = [0.0, 1.0];
                        }
                    }
                }
            }
        } else {
            if (input[13] <= 0.0004418622702360153) {
                if (input[40] <= -0.32206469774246216) {
                    if (input[14] <= -0.11618148908019066) {
                        var14 = [0.0, 1.0];
                    } else {
                        var14 = [1.0, 0.0];
                    }
                } else {
                    if (input[14] <= 0.1782659813761711) {
                        var14 = [0.0, 1.0];
                    } else {
                        var14 = [1.0, 0.0];
                    }
                }
            } else {
                if (input[8] <= 2.2218194007873535) {
                    if (input[28] <= 0.5) {
                        var14 = [1.0, 0.0];
                    } else {
                        if (input[39] <= 0.5) {
                            if (input[3] <= -0.14224010705947876) {
                                var14 = [1.0, 0.0];
                            } else {
                                var14 = [0.0, 1.0];
                            }
                        } else {
                            if (input[3] <= -0.4381391406059265) {
                                var14 = [1.0, 0.0];
                            } else {
                                var14 = [0.0, 1.0];
                            }
                        }
                    }
                } else {
                    var14 = [0.0, 1.0];
                }
            }
        }
    }
    return mulVectorNumber(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(addVectors(var0, var1), var2), var3), var4), var5), var6), var7), var8), var9), var10), var11), var12), var13), var14), 0.06666666666666667);
}
function addVectors(v1, v2) {
    var result = new Array(v1.length);
    for (var i = 0; i < v1.length; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}
function mulVectorNumber(v1, num) {
    var result = new Array(v1.length);
    for (var i = 0; i < v1.length; i++) {
        result[i] = v1[i] * num;
    }
    return result;
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
            notifyTabIfHighRisk(tabId, cached.result);
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
                    mlSource: 'cloud', // Feature: hybrid edge-cloud inference — source tag (not shown in UI)
                    timestamp: new Date().toISOString()
                };

                tabResults[tabId] = analysisResult;
                tabStatus[tabId] = 'complete';
                updateBadge(tabId, finalThreatLevel);

                // #4.4: Cache the result
                urlCache[url] = { result: analysisResult, timestamp: Date.now() };
                saveToHistory(analysisResult);
                notifyTabIfHighRisk(tabId, analysisResult);

                sendResponse({ status: 'ok' });
            })
            .catch(apiError => {
                clearTimeout(timeoutId);
                console.warn('PhishGuard: Cloud ML API unavailable, attempting on-device edge model.', apiError);

                // ─── Hybrid Edge-Cloud Inference: fall back to on-device model ──────
                let edgeResult = null;
                try {
                    edgeResult = runEdgeModel(allFeatures);
                    console.log('[PhishGuard] Cloud API unreachable — using on-device edge model. Phishing probability:', edgeResult.phishingProbability.toFixed(4));
                } catch (edgeError) {
                    console.error('[PhishGuard] Edge model also failed — falling back to heuristic-only.', edgeError);
                }

                const phishingProbability = edgeResult ? edgeResult.phishingProbability : 0;
                const mlLabel = edgeResult ? edgeResult.label : 'unknown';
                const mlSource = edgeResult ? 'edge' : 'none';

                const finalThreatLevel = computeFinalThreatLevel(
                    heuristicResult.heuristicLevel,
                    heuristicResult.score,
                    phishingProbability
                );

                const analysisResult = {
                    url: url,
                    finalThreatLevel: finalThreatLevel,
                    heuristicScore: heuristicResult.score,
                    heuristicLevel: heuristicResult.heuristicLevel,
                    triggeredRules: heuristicResult.triggeredRules,
                    mlPhishingProbability: phishingProbability,
                    mlLabel: mlLabel,
                    mlSource: mlSource, // 'edge' if on-device model ran, 'none' if that also failed
                    timestamp: new Date().toISOString()
                };

                tabResults[tabId] = analysisResult;
                tabStatus[tabId] = 'complete';
                updateBadge(tabId, finalThreatLevel);

                // Cache even edge/heuristic-only results to avoid repeated failed API calls
                urlCache[url] = { result: analysisResult, timestamp: Date.now() };
                saveToHistory(analysisResult);
                notifyTabIfHighRisk(tabId, analysisResult);

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
            mlSource: 'none',
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

    // ── Feature B: Popup requests for scan history ─────────────────────────
    if (message.type === 'GET_HISTORY') {
        chrome.storage.local.get(['scanHistory'], (data) => {
            sendResponse({ history: data.scanHistory || [] });
        });
        return true;
    }

    if (message.type === 'CLEAR_HISTORY') {
        chrome.storage.local.set({ scanHistory: [] }, () => {
            sendResponse({ status: 'ok' });
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