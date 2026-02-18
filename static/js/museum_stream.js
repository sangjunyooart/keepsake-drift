// Museum Stream Runtime
// Displays bilingual drift dialogue with wipe-reveal animation, 6s pauses, seamless looping

(function() {
  // Read persona from current page filename
  const pageMap = {
    'museum_human.html': 'human',
    'museum_human': 'human',
    'museum_liminal.html': 'liminal',
    'museum_liminal': 'liminal',
    'museum_environment.html': 'environment',
    'museum_environment': 'environment'
  };
  const currentPage = window.location.pathname.split('/').pop();
  const PERSONA = pageMap[currentPage] || 'human';

  const TICK_INTERVAL = 3600; // 1 hour in seconds
  const WIPE_DURATION = 2000; // 2 seconds for wipe reveal
  const PAUSE_BETWEEN_SENTENCES = 6000; // 6 seconds pause between sentences
  const FADE_OUT_DURATION = 2000; // 2 seconds to fade out

  let config = {
    l2_lang: 'ar',
    l2_dir: 'rtl'
  };

  // Elements
  const subtitlePrimary = document.getElementById('subtitlePrimary');
  const subtitleSecondary = document.getElementById('subtitleSecondary');
  const tempLabel = document.getElementById('tempLabel');
  const bgImage = document.getElementById('bgImage');

  // ─── API FETCHES ───────────────────────────────────────────────

  async function fetchLangConfig() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/lang_config`);
      const data = await res.json();
      config.l2_lang = data.second_lang || 'ar';
      config.l2_dir = data.direction || 'rtl';

      if (config.l2_dir === 'rtl') {
        subtitleSecondary.classList.add('rtl');
      } else {
        subtitleSecondary.classList.remove('rtl');
      }
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
      console.error('Failed to fetch version history:', e);
      return [];
    }
  }

  async function fetchStateAt(version) {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const res = await fetch(`${apiBase}/state_at?persona=${PERSONA}&version=${version}`);
      return await res.json();
    } catch (e) {
      console.error(`Failed to fetch version ${version}:`, e);
      return null;
    }
  }

  // ─── DIALOGUE BUILDER ─────────────────────────────────────────
  // Builds a looping dialogue from state data — the temporality AI
  // persona speaks about its own memory drift process.

  function buildDialogue(state, allDrifts) {
    const pairs = []; // [{en: "...", l2: "..."}, ...]

    if (!state) return pairs;

    const l2Field = `drift_${config.l2_lang}`;
    const recapL2Field = `recap_${config.l2_lang}`;

    // ── PART 1: The keepsake speaks (origin memory) ──
    if (state.keepsake_en) {
      pairs.push({
        en: state.keepsake_en,
        l2: state[`keepsake_${config.l2_lang}`] || state.keepsake_ar || ''
      });
    }

    // ── PART 2: How the world infiltrated ──
    if (state.justification_en) {
      pairs.push({
        en: state.justification_en,
        l2: state.justification_l2 || ''
      });
    }

    // ── PART 3: The drift direction (persona voice) ──
    if (state.drift_direction) {
      pairs.push({
        en: state.drift_direction,
        l2: '' // drift_direction is meta-text, no L2 equivalent typically
      });
    }

    // ── PART 4: Infiltrating imagery ──
    const imagery = state.infiltrating_imagery || [];
    if (imagery.length > 0) {
      const imageryText = `New imagery emerges: ${imagery.slice(0, 4).join(', ')}.`;
      pairs.push({
        en: imageryText,
        l2: ''
      });
    }

    // ── PART 5: The current drift text (the transformed memory) ──
    if (state.drift_en) {
      pairs.push({
        en: state.drift_en,
        l2: state[l2Field] || state.drift_ar || ''
      });
    }

    // ── PART 6: Recap / summary ──
    if (state.recap_en) {
      pairs.push({
        en: state.recap_en,
        l2: state[recapL2Field] || state.recap_ar || ''
      });
    }

    // ── PART 7: Evolutionary context from history ──
    if (allDrifts && allDrifts.length > 1) {
      // Origin anchor
      const drift0 = allDrifts[0];
      const anchors = drift0.sensory_anchors || [];
      if (anchors.length > 0) {
        const anchorText = `This memory began with ${anchors.slice(0, 3).join(', ')}.`;
        pairs.push({
          en: anchorText,
          l2: drift0[recapL2Field] || drift0.recap_ar || ''
        });
      }

      // Cumulative imagery from middle drifts
      if (allDrifts.length >= 4) {
        const middleDrifts = allDrifts.slice(1, -1);
        const allImagery = middleDrifts.flatMap(d => {
          const img = d.infiltrating_imagery;
          return Array.isArray(img) ? img : [];
        });
        const unique = [...new Set(allImagery)].slice(0, 4);
        if (unique.length > 0) {
          pairs.push({
            en: `Over ${allDrifts.length - 1} drifts, imagery accumulated: ${unique.join(', ')}.`,
            l2: ''
          });
        }
      }

      // Headlines that shaped the drift
      const headlines = [];
      for (const d of allDrifts.slice(1)) {
        if (d.curated_headlines && d.curated_headlines.length > 0) {
          headlines.push(d.curated_headlines[0]);
        }
      }
      if (headlines.length > 0) {
        const headlineText = `The world brought: ${headlines.slice(-3).join(' — ')}.`;
        pairs.push({
          en: headlineText,
          l2: ''
        });
      }
    }

    // ── PART 8: Transitional bridge (enables seamless loop) ──
    pairs.push({
      en: 'The memory continues to drift...',
      l2: 'الذاكرة تستمر في الانجراف...'
    });

    // Filter out pairs where both languages are empty
    return pairs.filter(p => p.en || p.l2);
  }

  // Split long text into sentences for paced display
  function splitIntoSentences(text) {
    if (!text || !text.trim()) return [''];

    // Split on sentence boundaries: . ! ? followed by space or end
    // Also handle Arabic sentence endings (،)
    const sentences = text.match(/[^.!?؟]+[.!?؟]+/g) || [text];
    return sentences.map(s => s.trim()).filter(s => s.length > 0);
  }

  // ─── WIPE REVEAL ANIMATION ────────────────────────────────────
  // Wipe uses overflow:hidden + max-width animation on a wrapper.
  // LTR: wrapper grows from 0 to full width (left-aligned).
  // RTL: wrapper grows from 0 to full width (right-aligned via margin-left:auto).

  function wipeReveal(text, element, isRTL = false, duration = WIPE_DURATION) {
    return new Promise((resolve) => {
      if (!text || !text.trim()) {
        element.textContent = '';
        element.style.opacity = '1';
        resolve();
        return;
      }

      // Clear element and create wrapper structure:
      // <element> → <div.wipe-wrapper style="overflow:hidden; max-width:0">
      //               <span.wipe-inner style="white-space:nowrap / normal">text</span>
      //             </div>
      element.textContent = '';
      element.style.opacity = '1';

      const wrapper = document.createElement('div');
      wrapper.style.overflow = 'hidden';
      wrapper.style.maxWidth = '0';
      wrapper.style.display = 'inline-block';
      if (isRTL) {
        wrapper.style.marginLeft = 'auto'; // push to right for RTL reveal
        wrapper.style.direction = 'rtl';
      }

      const inner = document.createElement('span');
      inner.style.display = 'inline-block';
      inner.style.whiteSpace = 'normal';
      inner.style.width = '90vw'; // wide enough for full text
      inner.textContent = text;

      wrapper.appendChild(inner);
      element.appendChild(wrapper);

      // Animate max-width from 0 to 100%
      const anim = wrapper.animate(
        [
          { maxWidth: '0px' },
          { maxWidth: '90vw' }
        ],
        { duration, easing: 'ease-out', fill: 'forwards' }
      );

      anim.onfinish = () => {
        // Clean up: replace wrapper structure with plain text
        element.textContent = text;
        resolve();
      };

      // Store for cancellation
      element._wipeAnim = anim;
      element._wipeResolve = resolve;
    });
  }

  // Cancel any running wipe animation
  function cancelWipe(element) {
    if (element._wipeAnim) {
      element._wipeAnim.cancel();
      element._wipeAnim = null;
    }
    if (element._wipeResolve) {
      element._wipeResolve();
      element._wipeResolve = null;
    }
  }

  // Fade out element using Web Animations API (no CSS transition needed)
  function fadeOut(element, duration = FADE_OUT_DURATION) {
    return new Promise((resolve) => {
      const anim = element.animate(
        [{ opacity: 1 }, { opacity: 0 }],
        { duration, fill: 'forwards' }
      );
      anim.onfinish = () => {
        element.style.opacity = '0';
        resolve();
      };
    });
  }

  // ─── BACKGROUND IMAGE (FULL BRIGHTNESS) ───────────────────────

  let imageHistory = [];
  let currentImageIndex = 0;
  let imageCycleInterval = null;

  function updateBackgroundImage(imageUrl) {
    if (!imageUrl) return;

    // Crossfade: fade out, swap, fade in at full brightness
    bgImage.style.opacity = '0';

    setTimeout(() => {
      bgImage.src = imageUrl;
      bgImage.style.opacity = '1'; // Full brightness
    }, 3000);
  }

  function startImageCycling(state) {
    if (imageCycleInterval) {
      clearInterval(imageCycleInterval);
    }

    const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
    imageHistory = [];
    if (state.image_url) imageHistory.push(apiBase + state.image_url);
    if (state.image_prev1) imageHistory.push(apiBase + state.image_prev1);
    if (state.image_prev2) imageHistory.push(apiBase + state.image_prev2);

    if (imageHistory.length === 0) return;

    if (imageHistory.length === 1) {
      updateBackgroundImage(imageHistory[0]);
      return;
    }

    currentImageIndex = 0;
    updateBackgroundImage(imageHistory[0]);

    // Cycle every 2 minutes
    imageCycleInterval = setInterval(() => {
      currentImageIndex = (currentImageIndex + 1) % imageHistory.length;
      updateBackgroundImage(imageHistory[currentImageIndex]);
    }, 120000);
  }

  // ─── MAIN DISPLAY LOOP (SEAMLESS) ─────────────────────────────

  async function displayLoop() {
    // Fetch all data
    const [state, versions] = await Promise.all([
      fetchState(),
      fetchVersionHistory()
    ]);

    if (!state) {
      console.error('No state data');
      await new Promise(r => setTimeout(r, 30000));
      return;
    }

    // Fetch historical drifts (limit to last 5 for performance)
    const recentVersions = versions.slice(-5);
    const allDrifts = [];
    for (const v of recentVersions) {
      const d = await fetchStateAt(v);
      if (d) allDrifts.push(d);
    }

    // Start background image cycling at full brightness
    startImageCycling(state);

    // Build dialogue from state data
    const dialoguePairs = buildDialogue(state, allDrifts);

    console.log(`[Museum] Built ${dialoguePairs.length} dialogue pairs for ${PERSONA}`);

    if (dialoguePairs.length === 0) {
      await new Promise(r => setTimeout(r, 30000));
      return;
    }

    // Expand pairs into individual sentences for pacing
    const displaySequence = [];
    for (const pair of dialoguePairs) {
      const enSentences = splitIntoSentences(pair.en);
      const l2Sentences = splitIntoSentences(pair.l2);
      const maxLen = Math.max(enSentences.length, l2Sentences.length);

      for (let i = 0; i < maxLen; i++) {
        displaySequence.push({
          en: enSentences[i] || '',
          l2: l2Sentences[i] || ''
        });
      }
    }

    // Loop through dialogue seamlessly
    while (true) {
      for (let i = 0; i < displaySequence.length; i++) {
        const { en, l2 } = displaySequence[i];

        // Clear previous
        cancelWipe(subtitlePrimary);
        cancelWipe(subtitleSecondary);
        subtitlePrimary.textContent = '';
        subtitleSecondary.textContent = '';
        subtitlePrimary.style.opacity = '0';
        subtitleSecondary.style.opacity = '0';

        // Brief pause before next sentence
        await new Promise(r => setTimeout(r, 300));

        // Wipe reveal: LTR for English, RTL for Arabic
        const isL2RTL = config.l2_dir === 'rtl';
        await Promise.all([
          wipeReveal(en, subtitlePrimary, false, WIPE_DURATION),
          wipeReveal(l2, subtitleSecondary, isL2RTL, WIPE_DURATION)
        ]);

        // Hold: let viewer read (scale with text length)
        const wordCount = (en + ' ' + l2).split(/\s+/).filter(w => w.length > 0).length;
        const readTime = Math.max(4000, Math.min(8000, wordCount * 350));
        await new Promise(r => setTimeout(r, readTime));

        // Fade out
        await Promise.all([
          fadeOut(subtitlePrimary),
          fadeOut(subtitleSecondary)
        ]);

        // 6 second pause between sentences
        await new Promise(r => setTimeout(r, PAUSE_BETWEEN_SENTENCES));
      }

      // After full loop, brief pause before restart
      console.log('[Museum] Dialogue loop complete, restarting...');
      await new Promise(r => setTimeout(r, 3000));
    }
  }

  // ─── INIT ─────────────────────────────────────────────────────

  async function startStream() {
    await fetchLangConfig();

    console.log(`[Museum] Stream initialized for ${PERSONA}`);

    // Start the seamless display loop
    displayLoop();

    // Re-fetch fresh data every tick and restart loop
    setInterval(async () => {
      console.log('[Museum] Tick: refreshing data...');
      // The loop is infinite; on next tick we'd ideally restart with fresh data.
      // For now the loop runs until page refresh or new tick triggers reload.
      window.location.reload();
    }, TICK_INTERVAL * 1000);
  }

  // Start on load — on Pages, wait for tunnel URL before fetching backend data
  document.addEventListener('DOMContentLoaded', () => {
    if (window.KD_API_CONFIG && window.KD_API_CONFIG.API_BASE_URL) {
      startStream();
    } else {
      window.addEventListener('kd-config-ready', () => startStream(), { once: true });
      setTimeout(() => {
        if (!document.querySelector('.subtitle-primary')?.textContent) {
          console.warn('[Museum] kd-config-ready never fired, starting with fallback');
          startStream();
        }
      }, 6000);
    }
  });
})();
