// Museum Stream Runtime v2
// Bilingual: L2 wipes in → English fades in after. EN-only: English wipes in. 6s pauses. Seamless loop.

(function() {
  let IS_BILINGUAL = false; // set from lang_config API after fetch

  const pageMap = {
    'museum_human.html': 'human',
    'museum_human': 'human',
    'museum1': 'human',
    'museum_liminal.html': 'liminal',
    'museum_liminal': 'liminal',
    'museum2': 'liminal',
    'museum_environment.html': 'environment',
    'museum_environment': 'environment',
    'museum3': 'environment',
    'museum_digital.html': 'digital',
    'museum_digital': 'digital',
    'museum_infrastructure.html': 'infrastructure',
    'museum_infrastructure': 'infrastructure',
    'museum_more_than_human.html': 'more_than_human',
    'museum_more_than_human': 'more_than_human'
  };
  const currentPage = window.location.pathname.split('/').pop();
  const PERSONA = pageMap[currentPage] || 'human';

  const TICK_INTERVAL = 3600;
  const WIPE_DURATION = 2000;
  const EN_FADE_IN_DURATION = 1200;
  const PAUSE_BETWEEN_SENTENCES = 6000;
  const FADE_OUT_DURATION = 2000;

  let config = { l2_lang: 'ar', l2_dir: 'rtl' };

  const subtitlePrimary   = document.getElementById('subtitlePrimary');
  const subtitleSecondary = document.getElementById('subtitleSecondary');
  const tempLabel         = document.getElementById('tempLabel');
  const bgImage           = document.getElementById('bgImage');

  // ─── API FETCHES ───────────────────────────────────────────────

  async function fetchLangConfig() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/lang_config`);
      const data = await res.json();
      IS_BILINGUAL   = data.enable_translation === true;
      config.l2_lang = data.second_lang || 'ar';
      config.l2_dir  = data.direction   || 'rtl';
      if (config.l2_dir === 'rtl') {
        subtitleSecondary.classList.add('rtl');
      } else {
        subtitleSecondary.classList.remove('rtl');
      }
      console.log('[Museum] lang_config — IS_BILINGUAL:', IS_BILINGUAL, '| l2:', config.l2_lang);
    } catch (e) {
      console.error('Failed to fetch lang config:', e);
    }
  }

  async function fetchState() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/state?persona=${PERSONA}`);
      return await res.json();
    } catch (e) {
      console.error('Failed to fetch state:', e);
      return null;
    }
  }

  async function fetchVersionHistory() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/versions?persona=${PERSONA}`);
      const data = await res.json();
      return data.versions || [];
    } catch (e) {
      console.error('Failed to fetch versions:', e);
      return [];
    }
  }

  async function fetchStateAt(version) {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/state_at?persona=${PERSONA}&version=${version}`);
      return await res.json();
    } catch (e) {
      console.error(`Failed to fetch state_at ${version}:`, e);
      return null;
    }
  }

  // ─── DIALOGUE BUILDER ─────────────────────────────────────────

  function buildDialogue(state, allDrifts) {
    const pairs = [];
    if (!state) return pairs;

    const l2Field      = `drift_${config.l2_lang}`;
    const recapL2Field = `recap_${config.l2_lang}`;

    if (state.keepsake_en) {
      pairs.push({ en: state.keepsake_en, l2: IS_BILINGUAL ? (state[`keepsake_${config.l2_lang}`] || state.keepsake_ar || '') : '' });
    }
    if (state.justification_en) {
      pairs.push({ en: state.justification_en, l2: IS_BILINGUAL ? (state.justification_l2 || '') : '' });
    }
    if (state.drift_direction) {
      pairs.push({ en: state.drift_direction, l2: '' });
    }
    if (state.drift_en) {
      pairs.push({ en: state.drift_en, l2: IS_BILINGUAL ? (state[l2Field] || state.drift_ar || '') : '' });
    }
    if (state.recap_en) {
      pairs.push({ en: state.recap_en, l2: IS_BILINGUAL ? (state[recapL2Field] || state.recap_ar || '') : '' });
    }

    if (allDrifts && allDrifts.length > 1) {
      const drift0  = allDrifts[0];
      const anchors = drift0.sensory_anchors || [];
      if (anchors.length > 0) {
        pairs.push({ en: `This memory began with ${anchors.slice(0, 3).join(', ')}.`, l2: IS_BILINGUAL ? (drift0[recapL2Field] || drift0.recap_ar || '') : '' });
      }
      if (allDrifts.length >= 4) {
        const allImagery = allDrifts.slice(1, -1).flatMap(d => Array.isArray(d.infiltrating_imagery) ? d.infiltrating_imagery : []);
        const unique = [...new Set(allImagery)].slice(0, 4);
        if (unique.length > 0) pairs.push({ en: `Over ${allDrifts.length - 1} drifts, imagery accumulated: ${unique.join(', ')}.`, l2: '' });
      }
      const headlines = allDrifts.slice(1).map(d => d.curated_headlines?.[0]).filter(Boolean);
      if (headlines.length > 0) pairs.push({ en: `The world brought: ${headlines.slice(-3).join(' — ')}.`, l2: '' });
    }

    pairs.push({ en: 'The memory continues to drift...', l2: '' });
    return pairs.filter(p => p.en || p.l2);
  }

  function splitIntoSentences(text) {
    if (!text || !text.trim()) return [''];
    const sentences = text.match(/[^.!?؟]+[.!?؟]+/g) || [text];
    return sentences.map(s => s.trim()).filter(s => s.length > 0);
  }

  // ─── ANIMATIONS ───────────────────────────────────────────────

  function wipeReveal(text, element, isRTL = false, duration = WIPE_DURATION) {
    return new Promise((resolve) => {
      // Cancel any active fill:forwards animation so opacity/display changes take effect
      element.getAnimations().forEach(a => a.cancel());

      if (!text || !text.trim()) {
        element.textContent = '';
        element.style.opacity = '1';
        resolve();
        return;
      }

      element.textContent = text;
      element.style.opacity = '1';
      element.style.display = 'inline-block';
      element.style.overflow = 'hidden';
      element.style.whiteSpace = 'nowrap';
      element.style.float = isRTL ? 'right' : 'none';
      element.style.maxWidth = 'none';
      const fullWidth = element.scrollWidth;
      element.style.maxWidth = '0px';

      const anim = element.animate(
        [{ maxWidth: '0px' }, { maxWidth: fullWidth + 'px' }],
        { duration, easing: 'ease-out', fill: 'forwards' }
      );
      anim.onfinish = () => {
        element.style.display = 'block';
        element.style.overflow = '';
        element.style.whiteSpace = '';
        element.style.maxWidth = '';
        element.style.float = '';
        anim.cancel();
        element._wipeResolve = null;
        resolve();
      };
      element._wipeResolve = resolve;
    });
  }

  function cancelWipe(element) {
    element.getAnimations().forEach(a => a.cancel());
    element.style.display   = 'block';
    element.style.overflow  = '';
    element.style.whiteSpace = '';
    element.style.maxWidth  = '';
    element.style.float     = '';
    if (element._wipeResolve) { element._wipeResolve(); element._wipeResolve = null; }
  }

  function fadeIn(element, duration = EN_FADE_IN_DURATION) {
    return new Promise((resolve) => {
      element.getAnimations().forEach(a => a.cancel());
      const anim = element.animate(
        [{ opacity: 0 }, { opacity: 1 }],
        { duration, fill: 'forwards' }
      );
      anim.onfinish = () => { element.style.opacity = '1'; anim.cancel(); resolve(); };
    });
  }

  function fadeOut(element, duration = FADE_OUT_DURATION) {
    return new Promise((resolve) => {
      element.getAnimations().forEach(a => a.cancel());
      const anim = element.animate(
        [{ opacity: 1 }, { opacity: 0 }],
        { duration, fill: 'forwards' }
      );
      anim.onfinish = () => { element.style.opacity = '0'; anim.cancel(); resolve(); };
    });
  }

  // ─── BACKGROUND IMAGE ─────────────────────────────────────────

  let imageHistory = [];
  let currentImageIndex = 0;
  let imageCycleInterval = null;

  function updateBackgroundImage(imageUrl) {
    if (!imageUrl) return;
    bgImage.style.opacity = '0';
    setTimeout(() => { bgImage.src = imageUrl; bgImage.style.opacity = '1'; }, 3000);
  }

  function startImageCycling(state) {
    if (imageCycleInterval) clearInterval(imageCycleInterval);
    const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
    imageHistory = [];
    if (state.image_url)   imageHistory.push(apiBase + state.image_url);
    if (state.image_prev1) imageHistory.push(apiBase + state.image_prev1);
    if (state.image_prev2) imageHistory.push(apiBase + state.image_prev2);
    if (imageHistory.length === 0) return;
    currentImageIndex = 0;
    updateBackgroundImage(imageHistory[0]);
    if (imageHistory.length > 1) {
      imageCycleInterval = setInterval(() => {
        currentImageIndex = (currentImageIndex + 1) % imageHistory.length;
        updateBackgroundImage(imageHistory[currentImageIndex]);
      }, 120000);
    }
  }

  // ─── MAIN DISPLAY LOOP ────────────────────────────────────────

  async function displayLoop() {
    const [state, versions] = await Promise.all([fetchState(), fetchVersionHistory()]);
    if (!state) {
      console.error('No state data');
      await new Promise(r => setTimeout(r, 30000));
      return;
    }

    const recentVersions = versions.slice(-5);
    const allDrifts = [];
    for (const v of recentVersions) {
      const d = await fetchStateAt(v);
      if (d) allDrifts.push(d);
    }

    startImageCycling(state);

    const dialoguePairs = buildDialogue(state, allDrifts);
    console.log(`[Museum] Built ${dialoguePairs.length} dialogue pairs for ${PERSONA}`);
    if (dialoguePairs.length === 0) {
      await new Promise(r => setTimeout(r, 30000));
      return;
    }

    const displaySequence = [];
    for (const pair of dialoguePairs) {
      const enSentences = splitIntoSentences(pair.en);
      const l2Sentences = splitIntoSentences(pair.l2);
      const maxLen = Math.max(enSentences.length, l2Sentences.length);
      for (let i = 0; i < maxLen; i++) {
        displaySequence.push({ en: enSentences[i] || '', l2: l2Sentences[i] || '' });
      }
    }

    while (true) {
      for (const { en, l2 } of displaySequence) {

        // ── Clear ──
        cancelWipe(subtitlePrimary);
        cancelWipe(subtitleSecondary);
        subtitlePrimary.textContent   = '';
        subtitleSecondary.textContent = '';
        subtitlePrimary.style.opacity   = '0';
        subtitleSecondary.style.opacity = '0';

        await new Promise(r => setTimeout(r, 300));

        if (IS_BILINGUAL && l2) {
          // 1. L2 wipes in (subtitleSecondary)
          await wipeReveal(l2, subtitleSecondary, config.l2_dir === 'rtl', WIPE_DURATION);

          // 2. English fades in after L2 completes (subtitlePrimary)
          if (en) {
            subtitlePrimary.textContent   = en;
            subtitlePrimary.style.opacity = '0';
            await fadeIn(subtitlePrimary, EN_FADE_IN_DURATION);
          }
        } else {
          // EN-only: English wipes into subtitlePrimary
          await wipeReveal(en, subtitlePrimary, false, WIPE_DURATION);
        }

        // 3. Hold
        const wordCount = (en + ' ' + l2).split(/\s+/).filter(w => w.length > 0).length;
        const readTime = Math.max(4000, Math.min(8000, wordCount * 350));
        await new Promise(r => setTimeout(r, readTime));

        // 4. Fade out both
        await Promise.all([fadeOut(subtitlePrimary), fadeOut(subtitleSecondary)]);

        // 5. 6s blank pause
        await new Promise(r => setTimeout(r, PAUSE_BETWEEN_SENTENCES));
      }

      console.log('[Museum] Dialogue loop complete, restarting...');
      await new Promise(r => setTimeout(r, 3000));
    }
  }

  // ─── INIT ─────────────────────────────────────────────────────

  async function startStream() {
    await fetchLangConfig();
    if (!IS_BILINGUAL) subtitleSecondary.style.display = 'none';
    console.log(`[Museum] Stream initialized for ${PERSONA} | IS_BILINGUAL: ${IS_BILINGUAL}`);
    displayLoop();
    setInterval(() => { window.location.reload(); }, TICK_INTERVAL * 1000);
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (window.KD_API_CONFIG && window.KD_API_CONFIG.API_BASE_URL != null) {
      startStream();
    } else {
      window.addEventListener('kd-config-ready', () => startStream(), { once: true });
      setTimeout(() => {
        if (!subtitlePrimary?.textContent) {
          console.warn('[Museum] kd-config-ready never fired, starting with fallback');
          startStream();
        }
      }, 6000);
    }
  });
})();
