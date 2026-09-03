/**
 * Maji Alert — Client-Side Application
 * =====================================
 * Pure vanilla JavaScript. Zero frameworks.
 * Features:
 * - Real-time Socket.io communication with client-side timeout safeguards
 * - Multi-language localization (English, Afaan Oromoo, Amharic)
 * - HTML5 Geolocation & Borana pastoral cluster presets
 * - Accessible visual drought gauge and pump operation advisory
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. Centralized Localization Dictionary (EN, OM, AM)
  // =========================================================================
  const translations = {
    en: {
      app_title: 'Maji Alert',
      app_subtitle: 'Borana Zone Solar Pump Drought Warning',
      status_connected: 'Connected to network',
      status_reconnecting: 'Reconnecting to network...',
      status_offline: 'Offline / Connection lost',
      section_location: 'Target Location & Coordinates',
      location_desc: 'Detect your GPS location, select a Borana well cluster, or enter coordinates manually.',
      btn_use_location: 'Use My Location',
      quick_select: 'Borana Well Clusters:',
      label_latitude: 'Latitude (°N/S)',
      label_longitude: 'Longitude (°E/W)',
      label_year: 'Forecast Year',
      hint_lat: 'Range: -90.0 to 90.0',
      hint_lon: 'Range: -180.0 to 180.0',
      hint_year: 'Solar projection cycle: 2025–2035',
      btn_check_drought: 'Check Drought Forecast',
      btn_checking: 'Analyzing Climate Data...',
      section_result: 'Drought Risk Assessment',
      risk_level: 'Risk Level',
      severity_normal: 'NORMAL / WET',
      severity_moderate: 'MODERATE DROUGHT',
      severity_severe: 'SEVERE DROUGHT',
      advisory_normal: 'Normal groundwater recharge expected. Solar water pumping operational at standard capacity.',
      advisory_moderate: 'Moderate water stress forecast. Implement rotational solar pumping schedules and conserve reservoir storage.',
      advisory_severe: 'CRITICAL: Severe drought phase projected. Enforce emergency water rationing for pastoral herds and prioritize human consumption.',
      confidence_breakdown: 'Model Confidence Probabilities',
      class_normal: 'Normal / Wet',
      class_moderate: 'Moderate Drought',
      class_severe: 'Severe Drought',
      meta_grid_cell: 'SPEI Grid Cell:',
      meta_distance: 'Grid Distance:',
      meta_year: 'Evaluation Year:',
      meta_latency: 'Engine Latency:',
      loading_analyzing: 'Analyzing Regional Climate Data...',
      loading_text: 'Analyzing Schwabe solar cycles & tree-ring memory...',
      footer_text: 'Maji Alert · Humanitarian Teleconnection Forecasting for Borana Zone, Oromia, Ethiopia · Low-Bandwidth Mode',
      err_geo_denied: 'Location permission was denied. Please enter coordinates manually.',
      err_geo_unavailable: 'GPS location is currently unavailable. Please enter coordinates manually.',
      err_geo_timeout: 'GPS location timed out. Please enter coordinates manually.',
      err_geo_unsupported: 'Geolocation is not supported by your browser.',
      err_invalid_lat: 'Latitude must be between -90.0 and 90.0',
      err_invalid_lon: 'Longitude must be between -180.0 and 180.0',
      err_invalid_year: 'Year must be between 1700 and 2100',
      err_timeout: 'Request timed out. Please check connection and try again.',
      err_generic: 'An unexpected error occurred. Please try again.',
    },
    om: {
      app_title: 'Maji Alert',
      app_subtitle: 'Akeekkachiisa Hongee Pompii Aduu Godina Booranaa',
      status_connected: 'Netwoorkii waliin walqunnamenteera',
      status_reconnecting: 'Irra deebi\'ee walqunnamaa jira...',
      status_offline: 'Netwoorkiin citeera',
      section_location: 'Iddoo fi Qubsuma Bakkaa',
      location_desc: 'GPS keessan fayyadamaa, madda bishaanii filadhaa, yookiin koordineetii galchaa.',
      btn_use_location: 'Bakkan Jiru Fayyadami',
      quick_select: 'Maddoota Bishaanii Booranaa:',
      label_latitude: 'Laatituudii (°N/S)',
      label_longitude: 'Loongitiudii (°E/W)',
      label_year: 'Waggaa Raagame',
      hint_lat: 'Gidduu: -90.0 hanga 90.0',
      hint_lon: 'Gidduu: -180.0 hanga 180.0',
      hint_year: 'Marsaa Aduu: 2025–2035',
      btn_check_drought: 'Raaga Hongee Ilaali',
      btn_checking: 'Qorannoon gaggeeffamaa jira...',
      section_result: 'Sadarkaa Sodaa Hongee',
      risk_level: 'Sadarkaa Balaa',
      severity_normal: 'NAGAYA / JIDDU-GALEESSA',
      severity_moderate: 'HONGEE GIDDU-GALEESSA',
      severity_severe: 'HONGEE JAWA / HAMAAN',
      advisory_normal: 'Bishaan lafa jalaa haala gaariin jira. Pompiin aduu bifa idileetiin ni hojjeta.',
      advisory_moderate: 'Bishaan hir\'achuu danda\'a. Sagantaa dabaree fayyadamaa bishaan qusadhaa.',
      advisory_severe: 'AKEERRA: Hongee cimaan ni dhufa! Bishaan beeyladaaf qusadhaa, dhala namaaf dursa kennaa.',
      confidence_breakdown: 'Hammamtaa Rakkinaa (Probabilities)',
      class_normal: 'Nagaya / Rooba',
      class_moderate: 'Hongee Giddu-galeessa',
      class_severe: 'Hongee Hamaa',
      meta_grid_cell: 'Qubsuma SPEI:',
      meta_distance: 'Fageenya Qubsumaa:',
      meta_year: 'Waggaa Qoratame:',
      meta_latency: 'Yeroo Qorannoo:',
      loading_analyzing: 'Oodeeffannoo Qilleensa Naannoo Xiinxalaa Jira...',
      loading_text: 'Marsaa aduu fi mallattoo mukkeeniin ragaa qindeessaa jira...',
      footer_text: 'Maji Alert · Tajaajila Raaga Hongee Godina Booranaatiif Hojjetame · Haafeeraa Xiqqaa',
      err_geo_denied: 'Eeyyamni bakkaa hin kennamne. Koordineetii harkaan galchaa.',
      err_geo_unavailable: 'Bakki GPS hin argamne. Koordineetii harkaan galchaa.',
      err_geo_timeout: 'GPS argachuun yeroo fudhateera. Harkaan galchaa.',
      err_geo_unsupported: 'Browseriin keessan GPS hin deeggaru.',
      err_invalid_lat: 'Laatituudiin -90.0 hanga 90.0 ta\'uu qaba.',
      err_invalid_lon: 'Loongitiudiin -180.0 hanga 180.0 ta\'uu qaba.',
      err_invalid_year: 'Waggaan 1700 hanga 2100 gidduu ta\'uu qaba.',
      err_timeout: 'Yeroon xumurameera. Maaloo irra deebi\'aa yaalaa.',
      err_generic: 'Dogoggorri uumameera. Irra deebi\'aa yaalaa.',
    },
    am: {
      app_title: 'ማጂ አለርት',
      app_subtitle: 'የቦረና ዞን የፀሐይ ኃይል ውኃ ፓምፕ ድርቅ ቅድመ ማስጠንቀቂያ',
      status_connected: 'ከመረብ ጋር ተገናኝቷል',
      status_reconnecting: 'እንደገና በመገናኘት ላይ...',
      status_offline: 'ግንኙነት ተቋርጧል',
      section_location: 'የቦታ መረጃ እና መጋጠሚያዎች',
      location_desc: 'የጂፒኤስ መገኛ ይጠቀሙ፣ የቦረና የውኃ ማዕከል ይምረጡ፣ ወይም መጋጠሚያዎችን ያስገቡ።',
      btn_use_location: 'መገኛዬን ተጠቀም',
      quick_select: 'የቦረና የውኃ ማዕከላት:',
      label_latitude: 'ላቲቲውድ (°N/S)',
      label_longitude: 'ሎንጊቲውድ (°E/W)',
      label_year: 'ትንበያ ዓመት',
      hint_lat: 'ወሰን: -90.0 እስከ 90.0',
      hint_lon: 'ወሰን: -180.0 እስከ 180.0',
      hint_year: 'የፀሐይ ዑደት ትንበያ: 2025–2035',
      btn_check_drought: 'የድርቅ ትንበያውን መርምር',
      btn_checking: 'ትንታኔ እየተከናወነ ነው...',
      section_result: 'የድርቅ አደጋ ግምገማ',
      risk_level: 'የስጋት ደረጃ',
      severity_normal: 'መደበኛ / እርጥበት',
      severity_moderate: 'መካከለኛ ድርቅ',
      severity_severe: 'ከባድ ድርቅ',
      advisory_normal: 'መደበኛ የከርሰ ምድር ውኃ ክምችት ይጠበቃል። የፀሐይ ውኃ ፓምፖች በመደበኛ መጠን ይሠራሉ።',
      advisory_moderate: 'መካከለኛ የውኃ እጥረት ይጠበቃል። የፈረቃ ውኃ አቅርቦት ይተግብሩ እና ክምችት ይቆጥቡ።',
      advisory_severe: 'አስቸኳይ ማስጠንቀቂያ: ከፍተኛ ድርቅ ይጠበቃል! የአስቸኳይ ጊዜ ውኃ ቁጠባ ይተግብሩ እና ቅድሚያ ለሰው ልጅ ይስጡ።',
      confidence_breakdown: 'የሞዴል ትንበያ ዕድሎች',
      class_normal: 'መደበኛ / እርጥብ',
      class_moderate: 'መካከለኛ ድርቅ',
      class_severe: 'ከባድ ድርቅ',
      meta_grid_cell: 'የ SPEI ሴል:',
      meta_distance: 'የሴል ርቀት:',
      meta_year: 'የተገመገመበት ዓመት:',
      meta_latency: 'የሞዴሉ የፍጥነት ጊዜ:',
      loading_analyzing: 'የአካባቢ የአየር ንብረት መረጃን በመተንተን ላይ...',
      loading_text: 'የፀሐይ ዑደቶችን እና የዛፍ ቀለበቶችን ታሪክ በመተንተን ላይ...',
      footer_text: 'ማጂ አለርት · ለቦረና ዞን የድርቅ ቅድመ ማስጠንቀቂያ ሥርዓት · አነስተኛ ባንድዊድዝ',
      err_geo_denied: 'የቦታ መረጃ ፈቃድ ተከልክሏል። እባክዎ መጋጠሚያዎችን በእጅ ያስገቡ።',
      err_geo_unavailable: 'የጂፒኤስ መገኛ ማግኘት አልተቻለም። በእጅ ያስገቡ።',
      err_geo_timeout: 'የጂፒኤስ መረጃ ዘግይቷል። እባክዎ በእጅ ያስገቡ።',
      err_geo_unsupported: 'የእርስዎ አሳሽ ጂፒኤስ አይደግፍም።',
      err_invalid_lat: 'ላቲቲውድ በ -90.0 እና 90.0 መካከል መሆን አለበት።',
      err_invalid_lon: 'ሎንጊቲውድ በ -180.0 እና 180.0 መካከል መሆን አለበት።',
      err_invalid_year: 'ዓመት በ 1700 እና 2100 መካከል መሆን አለበት።',
      err_timeout: 'ጥያቄው ጊዜ ወስዷል። እባክዎ ግንኙነትዎን ፈትሸው እንደገና ይሞክሩ።',
      err_generic: 'ያልተጠበቀ ስህተት ተከስቷል። እባክዎ ደግመው ይሞክሩ።',
    },
  };

  let currentLang = 'en';

  // =========================================================================
  // 2. DOM Elements Selection
  // =========================================================================
  const el = {
    langButtons: document.querySelectorAll('.lang-btn'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    btnGeolocation: document.getElementById('btnGeolocation'),
    presetButtons: document.querySelectorAll('.btn-preset'),
    form: document.getElementById('predictionForm'),
    inputLatitude: document.getElementById('inputLatitude'),
    inputLongitude: document.getElementById('inputLongitude'),
    inputYear: document.getElementById('inputYear'),
    latError: document.getElementById('latError'),
    lonError: document.getElementById('lonError'),
    yearError: document.getElementById('yearError'),
    generalError: document.getElementById('generalError'),
    btnSubmit: document.getElementById('btnSubmit'),
    btnSubmitText: document.getElementById('btnSubmitText'),
    btnSubmitSpinner: document.getElementById('btnSubmitSpinner'),
    processingCard: document.getElementById('processingCard'),
    processingMessage: document.getElementById('processingMessage'),
    resultSection: document.getElementById('resultSection'),
    gaugeContainer: document.getElementById('gaugeContainer'),
    gaugeIcon: document.getElementById('gaugeIcon'),
    severityLabel: document.getElementById('severityLabel'),
    pumpAdvisory: document.getElementById('pumpAdvisory'),
    probNormalText: document.getElementById('probNormalText'),
    probModerateText: document.getElementById('probModerateText'),
    probSevereText: document.getElementById('probSevereText'),
    barNormal: document.getElementById('barNormal'),
    barModerate: document.getElementById('barModerate'),
    barSevere: document.getElementById('barSevere'),
    progNormal: document.getElementById('progNormal'),
    progModerate: document.getElementById('progModerate'),
    progSevere: document.getElementById('progSevere'),
    metaGridCell: document.getElementById('metaGridCell'),
    metaDistance: document.getElementById('metaDistance'),
    metaYear: document.getElementById('metaYear'),
    metaLatency: document.getElementById('metaLatency'),
  };

  // State management
  let lastPredictionResult = null;
  let clientTimeoutTimer = null;
  const CLIENT_TIMEOUT_MS = 45000;

  // =========================================================================
  // 3. Localization Management
  // =========================================================================
  function setLanguage(lang) {
    if (!translations[lang]) return;
    currentLang = lang;

    // Update active button state
    el.langButtons.forEach((btn) => {
      const isActive = btn.getAttribute('data-lang') === lang;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });

    // Update static translated elements
    document.querySelectorAll('[data-i18n]').forEach((elem) => {
      const key = elem.getAttribute('data-i18n');
      if (translations[lang][key]) {
        elem.textContent = translations[lang][key];
      }
    });

    // Update dynamic results if present
    if (lastPredictionResult) {
      renderDroughtResult(lastPredictionResult);
    }
  }

  el.langButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const lang = btn.getAttribute('data-lang');
      setLanguage(lang);
    });
  });

  // =========================================================================
  // 4. Geolocation & Quick Presets
  // =========================================================================
  el.btnGeolocation.addEventListener('click', () => {
    clearErrors();
    if (!navigator.geolocation) {
      showGeneralError(translations[currentLang].err_geo_unsupported);
      return;
    }

    el.btnGeolocation.disabled = true;
    el.btnGeolocation.style.opacity = '0.7';

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        el.btnGeolocation.disabled = false;
        el.btnGeolocation.style.opacity = '1';
        el.inputLatitude.value = pos.coords.latitude.toFixed(4);
        el.inputLongitude.value = pos.coords.longitude.toFixed(4);
      },
      (err) => {
        el.btnGeolocation.disabled = false;
        el.btnGeolocation.style.opacity = '1';
        let msg = translations[currentLang].err_geo_unavailable;
        if (err.code === err.PERMISSION_DENIED) {
          msg = translations[currentLang].err_geo_denied;
        } else if (err.code === err.TIMEOUT) {
          msg = translations[currentLang].err_geo_timeout;
        }
        showGeneralError(msg);
      },
      { timeout: 10000, enableHighAccuracy: true }
    );
  });

  el.presetButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      clearErrors();
      el.inputLatitude.value = btn.getAttribute('data-lat');
      el.inputLongitude.value = btn.getAttribute('data-lon');
    });
  });

  // =========================================================================
  // 5. Form Validation & Submission
  // =========================================================================
  function clearErrors() {
    el.latError.textContent = '';
    el.lonError.textContent = '';
    el.yearError.textContent = '';
    el.inputLatitude.classList.remove('invalid');
    el.inputLongitude.classList.remove('invalid');
    el.inputYear.classList.remove('invalid');
    el.generalError.classList.add('hidden');
    el.generalError.textContent = '';
  }

  function showGeneralError(msg) {
    el.generalError.textContent = msg;
    el.generalError.classList.remove('hidden');
  }

  function validateForm() {
    clearErrors();
    let isValid = true;

    const lat = parseFloat(el.inputLatitude.value);
    const lon = parseFloat(el.inputLongitude.value);
    const yr = parseInt(el.inputYear.value, 10);

    if (isNaN(lat) || lat < -90 || lat > 90) {
      el.latError.textContent = translations[currentLang].err_invalid_lat;
      el.inputLatitude.classList.add('invalid');
      isValid = false;
    }

    if (isNaN(lon) || lon < -180 || lon > 180) {
      el.lonError.textContent = translations[currentLang].err_invalid_lon;
      el.inputLongitude.classList.add('invalid');
      isValid = false;
    }

    if (isNaN(yr) || yr < 1700 || yr > 2100) {
      el.yearError.textContent = translations[currentLang].err_invalid_year;
      el.inputYear.classList.add('invalid');
      isValid = false;
    }

    return isValid ? { latitude: lat, longitude: lon, year: yr } : null;
  }

  let isSubmitting = false;

  function setLoading(isLoading) {
    if (isLoading) {
      isSubmitting = true;
      el.btnSubmit.disabled = true;
      el.btnSubmitText.textContent = translations[currentLang].btn_checking;
      el.btnSubmitSpinner.classList.remove('hidden');
      el.processingMessage.textContent = translations[currentLang].loading_analyzing;
      el.processingCard.classList.remove('hidden');
      el.resultSection.classList.add('hidden');
      clearErrors();

      // Start client safety timer
      if (clientTimeoutTimer) clearTimeout(clientTimeoutTimer);
      clientTimeoutTimer = setTimeout(() => {
        setLoading(false);
        showGeneralError(translations[currentLang].err_timeout);
      }, CLIENT_TIMEOUT_MS);
    } else {
      isSubmitting = false;
      if (clientTimeoutTimer) {
        clearTimeout(clientTimeoutTimer);
        clientTimeoutTimer = null;
      }
      el.btnSubmit.disabled = false;
      el.btnSubmitText.textContent = translations[currentLang].btn_check_drought;
      el.btnSubmitSpinner.classList.add('hidden');
      el.processingCard.classList.add('hidden');
    }
  }

  const SERVER_ORIGIN =
    window.location.protocol === 'file:' || !window.location.host
      ? 'http://localhost:3000'
      : window.location.origin;

  if (window.location.protocol === 'file:') {
    const fileBanner = document.getElementById('fileProtocolWarning');
    if (fileBanner) fileBanner.classList.remove('hidden');
  }

  // =========================================================================
  // 6. Socket.io Client Connection & Event Listeners
  // =========================================================================
  let socket = null;

  function initSocket() {
    if (typeof io !== 'undefined') {
      socket = io(SERVER_ORIGIN, {
        reconnection: true,
        reconnectionAttempts: 10,
        reconnectionDelay: 1000,
        timeout: 60000,
      });

      socket.on('connect', () => {
        el.statusDot.className = 'status-dot online';
        el.statusText.textContent = translations[currentLang].status_connected;
      });

      socket.on('disconnect', () => {
        el.statusDot.className = 'status-dot offline';
        el.statusText.textContent = translations[currentLang].status_offline;
        setLoading(false);
      });

      socket.on('reconnecting', () => {
        el.statusDot.className = 'status-dot reconnecting';
        el.statusText.textContent = translations[currentLang].status_reconnecting;
      });

      socket.on('connect_error', () => {
        el.statusDot.className = 'status-dot offline';
        el.statusText.textContent = translations[currentLang].status_offline;
      });

      socket.on('drought:status', (data) => {
        if (data && data.message) {
          el.processingMessage.textContent = data.message;
        }
      });

      socket.on('drought:prediction_result', (data) => {
        setLoading(false);
        lastPredictionResult = data;
        renderDroughtResult(data);
      });

      socket.on('drought:prediction_error', (data) => {
        setLoading(false);
        const errMsg = data && data.message ? data.message : translations[currentLang].err_generic;
        showGeneralError(errMsg);
      });
    } else {
      console.warn('Socket.io client library not loaded. Falling back to HTTP REST API.');
    }
  }

  // If page was loaded directly as a local file, inject socket.io script from backend
  if (typeof io === 'undefined' && window.location.protocol === 'file:') {
    const socketScript = document.createElement('script');
    socketScript.src = `${SERVER_ORIGIN}/socket.io/socket.io.js`;
    socketScript.onload = () => initSocket();
    socketScript.onerror = () => {
      console.warn('Could not load remote socket.io script from server. Will use HTTP fallback.');
    };
    document.head.appendChild(socketScript);
  } else {
    initSocket();
  }

  // =========================================================================
  // 7. Prediction Submission Handler (Socket.io with HTTP Fallback)
  // =========================================================================
  el.form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isSubmitting) return; // Prevent duplicate rapid clicks

    const validData = validateForm();
    if (!validData) return;

    setLoading(true);

    if (socket && socket.connected) {
      socket.emit('drought:predict', validData);
    } else {
      // Graceful fallback to HTTP REST endpoint if WebSocket is offline
      try {
        const resp = await fetch(`${SERVER_ORIGIN}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(validData),
        });
        const data = await resp.json();
        setLoading(false);

        if (!resp.ok) {
          showGeneralError(data.error || translations[currentLang].err_generic);
        } else {
          lastPredictionResult = data;
          renderDroughtResult(data);
        }
      } catch (err) {
        setLoading(false);
        const isOffline =
          !navigator.onLine ||
          err.message?.includes('Failed to fetch') ||
          err.name === 'TypeError';
        const msg = isOffline
          ? `Cannot connect to server at ${SERVER_ORIGIN}. Please verify backend is running on port 3000.`
          : translations[currentLang].err_timeout;
        showGeneralError(msg);
      }
    }
  });

  // =========================================================================
  // 8. Result Renderer & Visual Drought Gauge
  // =========================================================================
  function renderDroughtResult(result) {
    const cls = result.predicted_drought_class;
    const probs = result.confidence_probabilities || { class_0: 0, class_1: 0, class_2: 0 };
    const grid = result.grid_cell || {};

    // 1. Reset gauge classes
    el.gaugeContainer.className = 'gauge-container';

    // 2. Set severity & advisory based on verified ML class
    if (cls === 2) {
      // Severe Drought (Class 2)
      el.gaugeContainer.classList.add('gauge-severe');
      el.gaugeIcon.textContent = '🔴';
      el.severityLabel.textContent = translations[currentLang].severity_severe;
      el.pumpAdvisory.textContent = translations[currentLang].advisory_severe;
    } else if (cls === 1) {
      // Moderate Drought (Class 1)
      el.gaugeContainer.classList.add('gauge-moderate');
      el.gaugeIcon.textContent = '🟡';
      el.severityLabel.textContent = translations[currentLang].severity_moderate;
      el.pumpAdvisory.textContent = translations[currentLang].advisory_moderate;
    } else {
      // Normal / Wet (Class 0)
      el.gaugeContainer.classList.add('gauge-normal');
      el.gaugeIcon.textContent = '🟢';
      el.severityLabel.textContent = translations[currentLang].severity_normal;
      el.pumpAdvisory.textContent = translations[currentLang].advisory_normal;
    }

    // 3. Confidence probabilities
    const p0 = Math.round(probs.class_0 * 100);
    const p1 = Math.round(probs.class_1 * 100);
    const p2 = Math.round(probs.class_2 * 100);

    el.probNormalText.textContent = `${p0}%`;
    el.probModerateText.textContent = `${p1}%`;
    el.probSevereText.textContent = `${p2}%`;

    el.barNormal.style.width = `${p0}%`;
    el.barModerate.style.width = `${p1}%`;
    el.barSevere.style.width = `${p2}%`;

    el.progNormal.setAttribute('aria-valuenow', p0);
    el.progModerate.setAttribute('aria-valuenow', p1);
    el.progSevere.setAttribute('aria-valuenow', p2);

    // 4. Metadata
    const selLat = grid.selected_lat !== undefined ? `${grid.selected_lat}°N` : '—';
    const selLon = grid.selected_lon !== undefined ? `${grid.selected_lon}°E` : '—';
    el.metaGridCell.textContent = `${selLat}, ${selLon}`;
    el.metaDistance.textContent = grid.distance_km !== undefined ? `${grid.distance_km} km` : '0 km';
    el.metaYear.textContent = String(result.year || '—');
    el.metaLatency.textContent = result.execution_duration_ms
      ? `${(result.execution_duration_ms / 1000).toFixed(2)}s`
      : '—';

    // 5. Display result section with smooth visibility
    el.resultSection.classList.remove('hidden');
    el.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
})();
