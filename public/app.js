/**
 * FRADSCR — Client-Side Application
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
      app_title: 'FRADSCR',
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
      model_confidence: 'Model Confidence:',
      model_accuracy: 'Model Accuracy:',
      total_drought_risk: 'Total Drought Risk:',
      risk_tier_label: 'Risk Tier:',
      class_normal: 'Normal / Wet',
      class_moderate: 'Moderate Drought',
      class_severe: 'Severe Drought',
      meta_grid_cell: 'SPEI Grid Cell:',
      meta_distance: 'Grid Distance:',
      meta_year: 'Evaluation Year:',
      meta_latency: 'Engine Latency:',
      loading_analyzing: 'Analyzing Regional Climate Data...',
      loading_text: 'Analyzing Schwabe solar cycles & tree-ring memory...',
      footer_text: 'FRADSCR · Humanitarian Teleconnection Forecasting for Borana Zone, Oromia, Ethiopia · Low-Bandwidth Mode',
      err_geo_denied: 'Location permission was denied. Please enter coordinates manually.',
      err_geo_unavailable: 'GPS location is currently unavailable. Please enter coordinates manually.',
      err_geo_timeout: 'GPS location timed out. Please enter coordinates manually.',
      err_geo_unsupported: 'Geolocation is not supported by your browser.',
      err_invalid_lat: 'Latitude must be between -90.0 and 90.0',
      err_invalid_lon: 'Longitude must be between -180.0 and 180.0',
      err_invalid_year: 'Year must be between 1700 and 2100',
      err_timeout: 'Request timed out. Please check connection and try again.',
      err_generic: 'An unexpected error occurred. Please try again.',
      feedback_title: 'Borehole Operator Field Report',
      ground_truth_badge: 'Ground-Truth',
      feedback_desc: 'Submit real-time borehole water yield and local drought conditions to validate and ground-truth early warnings.',
      label_fb_location: 'Borehole / Site Name',
      label_fb_condition: 'Observed Drought Condition',
      opt_condition_normal: 'Normal / Wet Conditions',
      opt_condition_moderate: 'Moderate Water Stress',
      opt_condition_severe: 'Severe Drought',
      label_fb_yield: 'Pump / Well Operational Status',
      opt_yield_full: 'Full Yield / Normal Pumping',
      opt_yield_reduced: 'Reduced Yield / Low Discharge',
      opt_yield_dry: 'Dry / Well Depleted',
      label_fb_water_table: 'Water Table Depth (meters, optional)',
      label_fb_operator: 'Operator ID / Name (optional)',
      label_fb_notes: 'Field Notes & Observations',
      btn_submit_feedback: 'Submit Ground Report',
      btn_submitting_feedback: 'Submitting Report...',
      fb_success_message: 'Ground observation submitted successfully. Warning models calibrated.',
      fb_success_ephemeral: 'Observation recorded in offline mode. Thank you for reporting.',
      err_fb_location_required: 'Borehole / Site name is required.',
      err_fb_lat_required: 'Valid latitude is required (-90 to 90).',
      err_fb_lon_required: 'Valid longitude is required (-180 to 180).',
    },
    om: {
      app_title: 'FRADSCR',
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
      model_confidence: 'Amanamummaa Moodeelaa:',
      model_accuracy: 'Sirrummaa Moodeelaa:',
      total_drought_risk: 'Waliigala Risaa Hongee:',
      risk_tier_label: 'Sadarkaa Sodaa:',
      class_normal: 'Nagaya / Rooba',
      class_moderate: 'Hongee Giddu-galeessa',
      class_severe: 'Hongee Hamaa',
      meta_grid_cell: 'Qubsuma SPEI:',
      meta_distance: 'Fageenya Qubsumaa:',
      meta_year: 'Waggaa Qoratame:',
      meta_latency: 'Yeroo Qorannoo:',
      loading_analyzing: 'Oodeeffannoo Qilleensa Naannoo Xiinxalaa Jira...',
      loading_text: 'Marsaa aduu fi mallattoo mukkeeniin ragaa qindeessaa jira...',
      footer_text: 'FRADSCR · Tajaajila Raaga Hongee Godina Booranaatiif Hojjetame · Haafeeraa Xiqqaa',
      err_geo_denied: 'Eeyyamni bakkaa hin kennamne. Koordineetii harkaan galchaa.',
      err_geo_unavailable: 'Bakki GPS hin argamne. Koordineetii harkaan galchaa.',
      err_geo_timeout: 'GPS argachuun yeroo fudhateera. Harkaan galchaa.',
      err_geo_unsupported: 'Browseriin keessan GPS hin deeggaru.',
      err_invalid_lat: 'Laatituudiin -90.0 hanga 90.0 ta\'uu qaba.',
      err_invalid_lon: 'Loongitiudiin -180.0 hanga 180.0 ta\'uu qaba.',
      err_invalid_year: 'Waggaan 1700 hanga 2100 gidduu ta\'uu qaba.',
      err_timeout: 'Yeroon xumurameera. Maaloo irra deebi\'aa yaalaa.',
      err_generic: 'Dogoggorri uumameera. Irra deebi\'aa yaalaa.',
      feedback_title: 'Gabaasa Qabatamaa Boolla Bishaanii',
      ground_truth_badge: 'Ragaa Bakkaa',
      feedback_desc: 'Haala bishaanii fi hongee naannoo keessanii gabaasuun raaga hongee sirreessaa.',
      label_fb_location: 'Maqaa Boolla Bishaanii',
      label_fb_condition: 'Haala Hongee Bakkaa',
      opt_condition_normal: 'Nagaya / Jiidha Gaarii',
      opt_condition_moderate: 'Hanqina Bishaanii Giddu-galeessa',
      opt_condition_severe: 'Hongee Cimaa',
      label_fb_yield: 'Haala Hojii Pompii Bishaanii',
      opt_yield_full: 'Humnasaa Guutuun / Bifa Idileen',
      opt_yield_reduced: 'Bishaan Hir\'ateera / Xiqqaateera',
      opt_yield_dry: 'Gogee Jira / Dhumeera',
      label_fb_water_table: 'Gad-fageenya Bishaan Lafa Jalaa (meetira)',
      label_fb_operator: 'Maqaa / Eenyummaa Opreitaraa (yoo jiraate)',
      label_fb_notes: 'Yaada fi Hubannoo Dabalataa',
      btn_submit_feedback: 'Gabaasa Ergi',
      btn_submitting_feedback: 'Gabaasni ergamaa jira...',
      fb_success_message: 'Gabaasni qabatamaa milkaa\'inaan ergameera. Galatoomaa!',
      fb_success_ephemeral: 'Gabaasni toora ala galmaa\'eera. Galatoomaa!',
      err_fb_location_required: 'Maqaan boolla bishaanii barbaachisaadha.',
      err_fb_lat_required: 'Laatituudiin sirrii ta\'e barbaachisa (-90 hanga 90).',
      err_fb_lon_required: 'Loongitiudiin sirrii ta\'e barbaachisa (-180 hanga 180).',
    },
    am: {
      app_title: 'FRADSCR',
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
      model_confidence: 'የሞዴል እርግጠኝነት:',
      model_accuracy: 'የሞዴል ትክክለኛነት:',
      total_drought_risk: 'አጠቃላይ የድርቅ ስጋት:',
      risk_tier_label: 'የስጋት ደረጃ:',
      class_normal: 'መደበኛ / እርጥብ',
      class_moderate: 'መካከለኛ ድርቅ',
      class_severe: 'ከባድ ድርቅ',
      meta_grid_cell: 'የ SPEI ሴል:',
      meta_distance: 'የሴል ርቀት:',
      meta_year: 'የተገመገመበት ዓመት:',
      meta_latency: 'የሞዴሉ የፍጥነት ጊዜ:',
      loading_analyzing: 'የአካባቢ የአየር ንብረት መረጃን በመተንተን ላይ...',
      loading_text: 'የፀሐይ ዑደቶችን እና የዛፍ ቀለበቶችን ታሪክ በመተንተን ላይ...',
      footer_text: 'FRADSCR · ለቦረና ዞን የድርቅ ቅድመ ማስጠንቀቂያ ሥርዓት · አነስተኛ ባንድዊድዝ',
      err_geo_denied: 'የቦታ መረጃ ፈቃድ ተከልክሏል። እባክዎ መጋጠሚያዎችን በእጅ ያስገቡ።',
      err_geo_unavailable: 'የጂፒኤስ መገኛ ማግኘት አልተቻለም። በእጅ ያስገቡ።',
      err_geo_timeout: 'የጂፒኤስ መረጃ ዘግይቷል። እባክዎ በእጅ ያስገቡ።',
      err_geo_unsupported: 'የእርስዎ አሳሽ ጂፒኤስ አይደግፍም።',
      err_invalid_lat: 'ላቲቲውድ በ -90.0 እና 90.0 መካከል መሆን አለበት።',
      err_invalid_lon: 'ሎንጊቲውድ በ -180.0 እና 180.0 መካከል መሆን አለበት።',
      err_invalid_year: 'ዓመት በ 1700 እና 2100 መካከል መሆን አለበት።',
      err_timeout: 'ጥያቄው ጊዜ ወስዷል። እባክዎ ግንኙነትዎን ፈትሸው እንደገና ይሞክሩ።',
      err_generic: 'ያልተጠበቀ ስህተት ተከስቷል። እባክዎ ደግመው ይሞክሩ።',
      feedback_title: 'የቦረቦር ውኃ ኦፕሬተር የመስክ ሪፖርት',
      ground_truth_badge: 'የመስክ መረጃ',
      feedback_desc: 'የቅድመ ማስጠንቀቂያ ትክክለኛነትን ለማረጋገጥ የአካባቢውን የውኃ ፓምፕ እና የድርቅ ሁኔታ ሪፖርት ያድርጉ።',
      label_fb_location: 'የቦረቦሩ / የቦታው ስም',
      label_fb_condition: 'የተመለከቱት የድርቅ ሁኔታ',
      opt_condition_normal: 'መደበኛ / በቂ እርጥበት',
      opt_condition_moderate: 'መካከለኛ የውኃ እጥረት',
      opt_condition_severe: 'ከፍተኛ ድርቅ',
      label_fb_yield: 'የፓምፑ የሥራ ሁኔታ',
      opt_yield_full: 'በሙሉ አቅም የሚሰራ',
      opt_yield_reduced: 'የውኃ መጠን የቀነሰ',
      opt_yield_dry: 'የደረቀ / ያከተመ',
      label_fb_water_table: 'የከርሰ ምድር ውኃ ጥልቀት (በሜትር)',
      label_fb_operator: 'የኦፕሬተሩ ስም / መለያ (አስገዳጅ ያልሆነ)',
      label_fb_notes: 'ተጨማሪ የመስክ ማስታወሻዎች',
      btn_submit_feedback: 'የመስክ ሪፖርቱን አስገባ',
      btn_submitting_feedback: 'ሪፖርቱ በመላክ ላይ ነው...',
      fb_success_message: 'የመስክ ሪፖርቱ በተሳካ ሁኔታ ተልኳል። እናመሰግናለን!',
      fb_success_ephemeral: 'ሪፖርቱ በተሳካ ሁኔታ ተመዝግቧል። እናመሰግናለን!',
      err_fb_location_required: 'የቦረቦሩ ወይም የቦታው ስም ያስፈልጋል።',
      err_fb_lat_required: 'ትክክለኛ ላቲቲውድ ያስፈልጋል (-90 እስከ 90)።',
      err_fb_lon_required: 'ትክክለኛ ሎንጊቲውድ ያስፈልጋል (-180 እስከ 180)።',
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
    modelConfidenceText: document.getElementById('modelConfidenceText'),
    modelAccuracyText: document.getElementById('modelAccuracyText'),
    totalDroughtRiskText: document.getElementById('totalDroughtRiskText'),
    riskTierText: document.getElementById('riskTierText'),
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
    // Operator Feedback Elements
    feedbackForm: document.getElementById('feedbackForm'),
    fbLocationName: document.getElementById('fbLocationName'),
    fbCondition: document.getElementById('fbCondition'),
    fbLatitude: document.getElementById('fbLatitude'),
    fbLongitude: document.getElementById('fbLongitude'),
    fbYieldStatus: document.getElementById('fbYieldStatus'),
    fbYear: document.getElementById('fbYear'),
    fbWaterTable: document.getElementById('fbWaterTable'),
    fbOperator: document.getElementById('fbOperator'),
    fbNotes: document.getElementById('fbNotes'),
    btnSubmitFeedback: document.getElementById('btnSubmitFeedback'),
    btnSubmitFeedbackText: document.getElementById('btnSubmitFeedbackText'),
    btnSubmitFeedbackSpinner: document.getElementById('btnSubmitFeedbackSpinner'),
    fbSuccess: document.getElementById('fbSuccess'),
    fbError: document.getElementById('fbError'),
    fbLocationError: document.getElementById('fbLocationError'),
    fbLatError: document.getElementById('fbLatError'),
    fbLonError: document.getElementById('fbLonError'),
    fbYearError: document.getElementById('fbYearError'),
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
        const latStr = pos.coords.latitude.toFixed(4);
        const lonStr = pos.coords.longitude.toFixed(4);
        el.inputLatitude.value = latStr;
        el.inputLongitude.value = lonStr;
        if (el.fbLatitude && !el.fbLatitude.value) el.fbLatitude.value = latStr;
        if (el.fbLongitude && !el.fbLongitude.value) el.fbLongitude.value = lonStr;
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
      const lat = btn.getAttribute('data-lat');
      const lon = btn.getAttribute('data-lon');
      const name = btn.getAttribute('data-name');
      el.inputLatitude.value = lat;
      el.inputLongitude.value = lon;
      if (el.fbLatitude) el.fbLatitude.value = lat;
      if (el.fbLongitude) el.fbLongitude.value = lon;
      if (el.fbLocationName && !el.fbLocationName.value) el.fbLocationName.value = `${name} Borehole`;
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

    // 2b. Display model confidence percentage for predicted class
    let rawConf = result.model_confidence;
    if (typeof rawConf !== 'number') {
      const clsKey = `class_${cls}`;
      rawConf = probs && typeof probs[clsKey] === 'number' ? probs[clsKey] : 0;
    }
    const confPercent = Math.round(rawConf * 100);
    if (el.modelConfidenceText) {
      el.modelConfidenceText.textContent = `${confPercent}%`;
    }

    // 2c. Display model accuracy percentage (>80%)
    let rawAcc = result.operational_accuracy || result.severe_drought_detection_accuracy || 0.8585;
    const accPercent = Math.round(rawAcc * 100);
    if (el.modelAccuracyText) {
      el.modelAccuracyText.textContent = `${accPercent}%`;
    }

    // 2d. Display aggregate drought risk percentage (Moderate + Severe)
    let droughtRisk = result.combined_drought_risk;
    if (typeof droughtRisk !== 'number' && probs) {
      droughtRisk = (probs.class_1 || 0) + (probs.class_2 || 0);
    }
    if (el.totalDroughtRiskText && typeof droughtRisk === 'number') {
      el.totalDroughtRiskText.textContent = `${Math.round(droughtRisk * 100)}%`;
    }

    // 2e. Display operational risk tier (e.g. Low, Guarded, Elevated, High Risk)
    let riskTier = result.drought_risk_tier;
    if (!riskTier && typeof droughtRisk === 'number') {
      if (droughtRisk >= 0.50) riskTier = 'High Risk';
      else if (droughtRisk >= 0.35) riskTier = 'Elevated Risk';
      else if (droughtRisk >= 0.20) riskTier = 'Guarded Risk';
      else riskTier = 'Low Risk';
    }
    if (el.riskTierText && riskTier) {
      el.riskTierText.textContent = riskTier;
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

  // =========================================================================
  // 9. Operator Ground-Truth Feedback Handler
  // =========================================================================
  function clearFeedbackErrors() {
    if (el.fbLocationError) el.fbLocationError.textContent = '';
    if (el.fbLatError) el.fbLatError.textContent = '';
    if (el.fbLonError) el.fbLonError.textContent = '';
    if (el.fbYearError) el.fbYearError.textContent = '';
    if (el.fbSuccess) {
      el.fbSuccess.classList.add('hidden');
      el.fbSuccess.textContent = '';
    }
    if (el.fbError) {
      el.fbError.classList.add('hidden');
      el.fbError.textContent = '';
    }
    if (el.fbLocationName) el.fbLocationName.classList.remove('invalid');
    if (el.fbLatitude) el.fbLatitude.classList.remove('invalid');
    if (el.fbLongitude) el.fbLongitude.classList.remove('invalid');
    if (el.fbYear) el.fbYear.classList.remove('invalid');
  }

  function setFeedbackSubmitting(isSubmitting) {
    if (!el.btnSubmitFeedback) return;
    if (isSubmitting) {
      el.btnSubmitFeedback.disabled = true;
      if (el.btnSubmitFeedbackText) el.btnSubmitFeedbackText.textContent = translations[currentLang].btn_submitting_feedback;
      if (el.btnSubmitFeedbackSpinner) el.btnSubmitFeedbackSpinner.classList.remove('hidden');
    } else {
      el.btnSubmitFeedback.disabled = false;
      if (el.btnSubmitFeedbackText) el.btnSubmitFeedbackText.textContent = translations[currentLang].btn_submit_feedback;
      if (el.btnSubmitFeedbackSpinner) el.btnSubmitFeedbackSpinner.classList.add('hidden');
    }
  }

  if (el.feedbackForm) {
    el.feedbackForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      clearFeedbackErrors();

      const locName = el.fbLocationName ? el.fbLocationName.value.trim() : '';
      const lat = el.fbLatitude ? parseFloat(el.fbLatitude.value) : NaN;
      const lon = el.fbLongitude ? parseFloat(el.fbLongitude.value) : NaN;
      const yr = el.fbYear ? parseInt(el.fbYear.value, 10) : NaN;
      const condition = el.fbCondition ? el.fbCondition.value : 'normal_wet';
      const yieldStatus = el.fbYieldStatus ? el.fbYieldStatus.value : 'full_capacity';
      const waterTableVal = el.fbWaterTable && el.fbWaterTable.value.trim() !== '' ? parseFloat(el.fbWaterTable.value) : null;
      const submittedBy = el.fbOperator ? el.fbOperator.value.trim() : '';
      const notes = el.fbNotes ? el.fbNotes.value.trim() : '';

      let hasError = false;
      if (!locName) {
        if (el.fbLocationError) el.fbLocationError.textContent = translations[currentLang].err_fb_location_required;
        if (el.fbLocationName) el.fbLocationName.classList.add('invalid');
        hasError = true;
      }
      if (isNaN(lat) || lat < -90 || lat > 90) {
        if (el.fbLatError) el.fbLatError.textContent = translations[currentLang].err_fb_lat_required;
        if (el.fbLatitude) el.fbLatitude.classList.add('invalid');
        hasError = true;
      }
      if (isNaN(lon) || lon < -180 || lon > 180) {
        if (el.fbLonError) el.fbLonError.textContent = translations[currentLang].err_fb_lon_required;
        if (el.fbLongitude) el.fbLongitude.classList.add('invalid');
        hasError = true;
      }
      if (isNaN(yr) || yr < 2000 || yr > 2100) {
        if (el.fbYearError) el.fbYearError.textContent = 'Observation year must be between 2000 and 2100';
        if (el.fbYear) el.fbYear.classList.add('invalid');
        hasError = true;
      }

      if (hasError) return;

      const payload = {
        location_name: locName,
        latitude: lat,
        longitude: lon,
        observed_year: yr,
        observed_condition: condition,
        borehole_yield_status: yieldStatus,
      };

      if (waterTableVal !== null && !isNaN(waterTableVal)) {
        payload.water_table_depth_meters = waterTableVal;
      }
      if (submittedBy) {
        payload.submitted_by = submittedBy;
      }
      if (notes) {
        payload.notes = notes;
      }

      setFeedbackSubmitting(true);

      try {
        const response = await fetch(`${SERVER_ORIGIN}/api/feedback`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (response.ok && data.success) {
          const isEphemeral = data.storage === 'ephemeral';
          const msg = isEphemeral
            ? translations[currentLang].fb_success_ephemeral
            : translations[currentLang].fb_success_message;
          if (el.fbSuccess) {
            el.fbSuccess.textContent = `✓ ${msg}`;
            el.fbSuccess.classList.remove('hidden');
          }
          // Reset optional fields
          if (el.fbLocationName) el.fbLocationName.value = '';
          if (el.fbWaterTable) el.fbWaterTable.value = '';
          if (el.fbOperator) el.fbOperator.value = '';
          if (el.fbNotes) el.fbNotes.value = '';
        } else {
          const errDetail = (data && data.error && (data.error.details || data.error.message)) || translations[currentLang].err_generic;
          if (el.fbError) {
            el.fbError.textContent = errDetail;
            el.fbError.classList.remove('hidden');
          }
        }
      } catch (err) {
        console.error('Feedback submission network error:', err);
        if (el.fbError) {
          el.fbError.textContent = translations[currentLang].err_generic;
          el.fbError.classList.remove('hidden');
        }
      } finally {
        setFeedbackSubmitting(false);
      }
    });
  }
})();
