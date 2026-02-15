// Museum Stream Runtime
// Displays drift narration with typewriter effect, bilingual subtitles, 10-minute pacing

(function() {
  // Read persona from current page filename
  const pageMap = {
    'museum_human.html': 'human',
    'museum_liminal.html': 'liminal',
    'museum_environment.html': 'environment'
  };
  const currentPage = window.location.pathname.split('/').pop();
  const PERSONA = pageMap[currentPage] || 'human';

  const TICK_INTERVAL = 3600; // 1 hour in seconds (from KD_TICK_INTERVAL)
  const NARRATION_DURATION = 600; // 10 minutes in seconds
  const TYPEWRITER_SPEED = 50; // milliseconds per character

  let currentDrift = {
    version: 0,
    drift_en: '',
    drift_l2: '',
    l2_lang: 'ar',
    l2_dir: 'rtl'
  };

  // Elements
  const subtitlePrimary = document.getElementById('subtitlePrimary');
  const subtitleSecondary = document.getElementById('subtitleSecondary');
  const qrCode = document.getElementById('qrCode');
  const tempLabel = document.getElementById('tempLabel');
  const bgImage = document.getElementById('bgImage');

  // Fetch language config
  async function fetchLangConfig() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/lang_config`);
      const data = await res.json();
      currentDrift.l2_lang = data.second_lang || 'ar';
      currentDrift.l2_dir = data.direction || 'rtl';

      // Apply RTL if needed
      if (currentDrift.l2_dir === 'rtl') {
        subtitleSecondary.classList.add('rtl');
      } else {
        subtitleSecondary.classList.remove('rtl');
      }
    } catch (e) {
      console.error('Failed to fetch lang config:', e);
    }
  }

  // Fetch latest drift for this persona
  async function fetchDrift() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/state?persona=${PERSONA}`);
      const data = await res.json();

      currentDrift.version = data.version || 0;
      currentDrift.drift_en = data.drift_en || '';

      // Second language field depends on active lang
      const l2Field = `drift_${currentDrift.l2_lang}`;
      currentDrift.drift_l2 = data[l2Field] || data.drift_ar || '';

      return data;
    } catch (e) {
      console.error('Failed to fetch drift:', e);
      return null;
    }
  }

  // Smooth fade-in effect for narration (TV caption style)
  function smoothFadeInEffect(text, element, duration = 2000) {
    return new Promise((resolve) => {
      // Set text content immediately but invisible
      element.textContent = text;
      element.style.opacity = '0';

      // Trigger fade-in after a brief moment (for CSS transition to work)
      setTimeout(() => {
        element.style.opacity = '1';

        // Resolve after fade completes
        setTimeout(resolve, duration);
      }, 50);
    });
  }

  // Split text into sentences for one-at-a-time display
  function splitIntoSentences(text) {
    if (!text || !text.trim()) return [];

    // Split on sentence boundaries: . ! ? followed by space or end
    // Regex captures sentence with its punctuation
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];

    // Clean up whitespace and filter empty
    return sentences
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  // Fetch version history for a persona
  async function fetchVersionHistory(persona) {
    try {
      const res = await fetch(`/versions?persona=${persona}`);
      const data = await res.json();
      return data.versions || []; // Returns [0, 1, 2, 3, ...]
    } catch (e) {
      console.error('Failed to fetch version history:', e);
      return [];
    }
  }

  // Fetch all drift versions for evolutionary analysis
  async function fetchAllDrifts(persona, versions) {
    const drifts = [];

    for (const version of versions) {
      try {
        const res = await fetch(`/state_at?persona=${persona}&version=${version}`);
        const data = await res.json();
        drifts.push(data);
      } catch (e) {
        console.error(`Failed to fetch version ${version}:`, e);
      }
    }

    return drifts;
  }

  // Generate evolutionary narration from drift history
  function generateEvolutionaryNarration(drifts) {
    const sentences = [];

    // Edge case: no drifts
    if (drifts.length === 0) {
      return ["This memory is waiting to drift."];
    }

    // Edge case: only drift 0 (no evolution yet)
    if (drifts.length === 1) {
      const drift0 = drifts[0];
      const sensoryAnchors = drift0.sensory_anchors || [];
      const anchorsText = sensoryAnchors.slice(0, 3).join(', ');

      return [`This memory holds ${anchorsText || 'a moment of stillness'}, waiting to drift.`];
    }

    // SENTENCE 1: Origin (Drift 0)
    const drift0 = drifts[0];
    const sensoryAnchors = drift0.sensory_anchors || [];
    const anchorsText = sensoryAnchors.slice(0, 3).join(', ');

    const originSentence = `This memory began: ${anchorsText || 'a moment preserved in time'}.`;
    sentences.push(originSentence);

    // SENTENCE 2: First Transformation (Drift 1)
    if (drifts.length >= 2) {
      const drift1 = drifts[1];
      const headlines1 = (drift1.curated_headlines && drift1.curated_headlines[0]) || 'the present moment';
      const justification1 = drift1.justification_en || '';

      // Extract first sentence from justification for brevity
      const firstJustificationSentence = justification1.split('.')[0] + '.';

      const firstTransformSentence = `When ${headlines1} emerged, ${firstJustificationSentence.toLowerCase()}`;
      sentences.push(firstTransformSentence);
    }

    // SENTENCE 3: Cumulative Pattern (Drift 2 through N-1)
    if (drifts.length >= 4) {
      const middleDrifts = drifts.slice(2, -1);
      const allImagery = middleDrifts.flatMap(d => {
        const imagery = d.infiltrating_imagery;
        return Array.isArray(imagery) ? imagery : [];
      });
      const uniqueImagery = [...new Set(allImagery)];
      const imageryList = uniqueImagery.slice(0, 3).join(', ');

      const driftCount = middleDrifts.length;
      if (imageryList) {
        const cumulativeSentence = `Over ${driftCount} drift${driftCount > 1 ? 's' : ''}, imagery accumulated: ${imageryList}.`;
        sentences.push(cumulativeSentence);
      }
    }

    // SENTENCE 4: Current State (Latest Drift)
    const currentDrift = drifts[drifts.length - 1];
    const currentRecap = currentDrift.recap_en || '';

    if (currentRecap) {
      // Make first letter lowercase if it starts with "This"
      const recap = currentRecap.startsWith('This')
        ? 'now this' + currentRecap.substring(4)
        : 'Now ' + currentRecap.charAt(0).toLowerCase() + currentRecap.substring(1);

      sentences.push(recap);
    } else {
      sentences.push("Now the memory stands transformed.");
    }

    return sentences;
  }

  // Generate second language evolutionary narration
  function generateEvolutionaryNarrationL2(drifts) {
    const sentences = [];

    if (drifts.length === 0) {
      return [""];
    }

    if (drifts.length === 1) {
      return [drifts[0].recap_ar || drifts[0].recap_el || drifts[0].recap_pt || ""];
    }

    // Mirror the structure of primary narration
    // SENTENCE 1: Origin
    const drift0 = drifts[0];
    sentences.push(drift0.recap_ar || drift0.recap_el || drift0.recap_pt || "");

    // SENTENCE 2: First transformation
    if (drifts.length >= 2) {
      const drift1 = drifts[1];
      sentences.push(drift1.justification_l2 || "");
    }

    // SENTENCE 3: Cumulative pattern
    if (drifts.length >= 4) {
      const drift2 = drifts[2];
      sentences.push(drift2.drift_l2 || "");
    }

    // SENTENCE 4: Current state
    const currentDrift = drifts[drifts.length - 1];
    sentences.push(currentDrift.recap_ar || currentDrift.recap_el || currentDrift.recap_pt || "");

    return sentences;
  }

  // Generate meta-narration about HOW the memory shifted
  function generateMetaNarration(drift) {
    // Extract data from drift object (now included in /state response)
    const summary = drift.recap_en || '';
    const justification = drift.justification_en || '';
    const driftDirection = drift.drift_direction || '';
    const infiltratingImagery = drift.infiltrating_imagery || [];

    // Construct meta-narration explaining the shift
    let narration = '';

    if (summary) {
      narration += summary + ' ';
    }

    if (justification) {
      narration += justification + ' ';
    }

    if (driftDirection) {
      narration += `This memory drifts ${driftDirection}. `;
    }

    if (infiltratingImagery.length > 0) {
      narration += `New imagery emerges: ${infiltratingImagery.slice(0, 3).join(', ')}. `;
    }

    // Fallback to drift_text if no meta-narration available
    if (!narration.trim()) {
      narration = drift.drift_en || '';
    }

    return narration.trim();
  }

  // Generate second language meta-narration
  function generateMetaNarrationL2(drift) {
    // Try to get second language recap (field name depends on active L2)
    const recapField = `recap_${currentDrift.l2_lang}`;
    const summary = drift[recapField] || drift.recap_ar || drift.recap_el || drift.recap_pt || '';
    const justification = drift.justification_l2 || '';

    let narration = '';

    if (summary) {
      narration += summary + ' ';
    }

    if (justification) {
      narration += justification + ' ';
    }

    // Fallback to translated drift_text
    if (!narration.trim()) {
      narration = currentDrift.drift_l2 || '';
    }

    return narration.trim();
  }

  // Background image cycling state
  let imageHistory = [];
  let currentImageIndex = 0;
  let imageCycleInterval = null;

  // Update background image with fade transition
  function updateBackgroundImage(imageUrl) {
    if (!imageUrl) return;

    // Fade out current image
    bgImage.style.opacity = '0';

    setTimeout(() => {
      bgImage.src = imageUrl;
      bgImage.style.opacity = '0.25'; // Fade back in at 25% opacity
    }, 3000); // 3 second fade
  }

  // Start cycling through cached background images
  function startImageCycling(drift) {
    // Clear previous cycle if exists
    if (imageCycleInterval) {
      clearInterval(imageCycleInterval);
    }

    // Build image history array (current + previous cached images)
    const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
    imageHistory = [];
    if (drift.image_url) imageHistory.push(apiBase + drift.image_url);
    if (drift.image_prev1) imageHistory.push(apiBase + drift.image_prev1);
    if (drift.image_prev2) imageHistory.push(apiBase + drift.image_prev2);

    // If no images at all, skip
    if (imageHistory.length === 0) {
      return;
    }

    // If only one image, show it without cycling
    if (imageHistory.length === 1) {
      updateBackgroundImage(imageHistory[0]);
      return;
    }

    // Set initial image
    currentImageIndex = 0;
    updateBackgroundImage(imageHistory[0]);

    // Cycle through images every 2 minutes during 10-min narration
    // 10 min / 3 images = ~3 min per image, but we'll do 2 min for variety
    imageCycleInterval = setInterval(() => {
      currentImageIndex = (currentImageIndex + 1) % imageHistory.length;
      updateBackgroundImage(imageHistory[currentImageIndex]);
    }, 120000); // 2 minutes
  }

  // Display drift with sentence-by-sentence narration
  async function displayDrift() {
    // Fetch version history for evolutionary narration
    const versions = await fetchVersionHistory(PERSONA);

    console.log(`Found ${versions.length} versions for ${PERSONA}`);

    // Fetch all drift versions
    const allDrifts = await fetchAllDrifts(PERSONA, versions);

    if (allDrifts.length === 0) {
      console.error('No drifts found');
      return;
    }

    // Use latest drift for background images
    const latestDrift = allDrifts[allDrifts.length - 1];
    startImageCycling(latestDrift);

    // Generate evolutionary narration from full history
    const primarySentences = generateEvolutionaryNarration(allDrifts);
    const secondarySentences = generateEvolutionaryNarrationL2(allDrifts);

    const totalSentences = Math.max(primarySentences.length, secondarySentences.length);

    if (totalSentences === 0) {
      await new Promise(resolve => setTimeout(resolve, NARRATION_DURATION * 1000));
      return;
    }

    // Display sentences one at a time
    for (let i = 0; i < totalSentences; i++) {
      const primarySentence = primarySentences[i] || '';
      const secondarySentence = secondarySentences[i] || '';

      subtitlePrimary.textContent = '';
      subtitleSecondary.textContent = '';
      subtitlePrimary.style.opacity = '0';
      subtitleSecondary.style.opacity = '0';

      // 1. SMOOTH FADE-IN PHASE - Both languages simultaneously (2.5 seconds)
      const primaryPromise = smoothFadeInEffect(primarySentence, subtitlePrimary, 2500);
      const secondaryPromise = smoothFadeInEffect(secondarySentence, subtitleSecondary, 2500);
      await Promise.all([primaryPromise, secondaryPromise]);

      // 2. HOLD PHASE - Keep sentence visible for reading
      const wordsCount = primarySentence.split(/\s+/).filter(w => w.length > 0).length;
      const holdDuration = Math.max(5000, Math.min(10000, wordsCount * 500));
      await new Promise(resolve => setTimeout(resolve, holdDuration));

      // 3. FADE PHASE - Fade out both languages together
      subtitlePrimary.style.opacity = '0';
      subtitleSecondary.style.opacity = '0';
      await new Promise(resolve => setTimeout(resolve, 1000));

      // 4. PAUSE PHASE - Blank screen contemplative pause
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  // Generate QR code using external API
  function generateQRCode() {
    const targetUrl = window.location.origin + '/chat?persona=' + PERSONA;
    const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(targetUrl)}`;
    qrCode.src = qrUrl;
  }

  // Initialize voice synthesis (voices load asynchronously)
  function initializeVoices() {
    return new Promise((resolve) => {
      // Voices might already be loaded
      if (window.speechSynthesis.getVoices().length > 0) {
        resolve();
        return;
      }

      // Wait for voices to load
      window.speechSynthesis.onvoiceschanged = () => {
        resolve();
      };

      // Fallback timeout in case voices never load
      setTimeout(resolve, 1000);
    });
  }

  // Main loop - refresh drift every TICK_INTERVAL
  async function startStream() {
    await fetchLangConfig();
    generateQRCode();

    console.log('Museum stream initialized');

    // Initial display
    await displayDrift();

    // Poll for new drifts every tick interval
    setInterval(async () => {
      await displayDrift();
    }, TICK_INTERVAL * 1000);
  }

  // Start on load
  document.addEventListener('DOMContentLoaded', startStream);
})();
