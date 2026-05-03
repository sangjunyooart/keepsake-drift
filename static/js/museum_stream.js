// Museum Stream Runtime — single-persona, GR animation style
// L2 wipes in char-by-char → EN fades in → hold → L2 wipes out → EN fades out

(function() {
  const pageMap = {
    'museum_human.html': 'human', 'museum_human': 'human', 'museum1': 'human',
    'museum_liminal.html': 'liminal', 'museum_liminal': 'liminal', 'museum2': 'liminal',
    'museum_environment.html': 'environment', 'museum_environment': 'environment', 'museum3': 'environment',
    'museum_digital.html': 'digital', 'museum_digital': 'digital',
    'museum_infrastructure.html': 'infrastructure', 'museum_infrastructure': 'infrastructure',
    'museum_more_than_human.html': 'more_than_human', 'museum_more_than_human': 'more_than_human'
  };
  const currentPage = window.location.pathname.split('/').pop();
  const PERSONA = pageMap[currentPage] || 'human';

  const LABELS = {
    'human': 'Human Time', 'liminal': 'Liminal Time', 'environment': 'Environmental Time',
    'digital': 'Digital Time', 'infrastructure': 'Infrastructure Time',
    'more_than_human': 'More-than-Human Time'
  };

  const WIPE_DURATION            = 5000;
  const WIPE_DISAPPEAR_DURATION  = 4000;
  const FADE_IN_DURATION         = 1200;
  const FADE_OUT_DURATION        = 1500;
  const HOLD_MIN                 = 5000;
  const HOLD_MAX                 = 11000;
  const HOLD_MS_PER_CHAR         = 80;

  let config = { l2_lang: 'ar', l2_dir: 'rtl', l2_enabled: true };
  let streamStarted = false;

  const subtitlePrimary   = document.getElementById('subtitlePrimary');
  const subtitleSecondary = document.getElementById('subtitleSecondary');
  const tempLabel         = document.getElementById('tempLabel');
  const bgImageA          = document.getElementById('bgImageA');
  const bgImageB          = document.getElementById('bgImageB');
  let activeBg = 'A';

  tempLabel.textContent = LABELS[PERSONA] || PERSONA;

  // ─── API ──────────────────────────────────────────────────────

  async function fetchLangConfig() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const data = await (await fetch(`${apiBase}/lang_config`)).json();
      config.l2_lang    = data.second_lang || 'ar';
      config.l2_dir     = data.direction   || 'rtl';
      config.l2_enabled = data.enable_translation !== false;
      if (config.l2_dir === 'rtl' && config.l2_enabled) {
        subtitleSecondary.classList.add('rtl');
      } else {
        subtitleSecondary.classList.remove('rtl');
      }
    } catch (e) { console.error('lang_config failed:', e); }
  }

  async function fetchVersionHistory() {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      const data = await (await fetch(`${apiBase}/versions?persona=${PERSONA}`)).json();
      return data.versions || [];
    } catch (e) { return []; }
  }

  async function fetchStateAt(version) {
    try {
      const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
      return await (await fetch(`${apiBase}/state_at?persona=${PERSONA}&version=${version}`)).json();
    } catch (e) { return null; }
  }

  // ─── DIALOGUE ─────────────────────────────────────────────────

  function buildDialogue(state) {
    const pairs = [];
    if (!state) return pairs;
    const l2Field = `drift_${config.l2_lang}`;
    const useL2   = config.l2_enabled;

    if (state.keepsake_en) {
      pairs.push({ en: state.keepsake_en, l2: useL2 ? (state[`keepsake_${config.l2_lang}`] || state.keepsake_ar || '') : '' });
    }
    const narrationEn = state.museum_narration_en || state.drift_en || '';
    const narrationL2 = useL2 ? (state.museum_narration_en
      ? (state[`museum_narration_${config.l2_lang}`] || state.museum_narration_ar || '')
      : (state[l2Field] || state.drift_ar || '')) : '';
    if (narrationEn) pairs.push({ en: narrationEn, l2: narrationL2 });

    return pairs.filter(p => p.en || p.l2);
  }

  function splitIntoSentences(text) {
    if (!text || !text.trim()) return [''];
    const raw = text.match(/[^.!?؟]+[.!?؟]+/g) || [];
    const lastPuncIdx = Math.max(...['.','!','?','؟'].map(p => text.lastIndexOf(p)));
    const afterLast = lastPuncIdx >= 0 ? text.slice(lastPuncIdx + 1).trim() : '';
    if (afterLast) raw.push(afterLast);
    if (raw.length === 0) return [text];
    const merged = [raw[0].trim()];
    for (let i = 1; i < raw.length; i++) {
      const s = raw[i].trim();
      if (!s) continue;
      if (s.length < 40 || merged[merged.length - 1].length < 40) merged[merged.length - 1] += ' ' + s;
      else merged.push(s);
    }
    return merged.filter(s => s.length > 0);
  }

  // ─── ANIMATIONS ───────────────────────────────────────────────

  function cancelWipe(element) {
    if (element._wipeInterval) { clearInterval(element._wipeInterval); element._wipeInterval = null; }
    if (element._wipeResolve) { const fn = element._wipeResolve; element._wipeResolve = null; fn(); }
  }

  function wipeReveal(text, element, isRTL = false, duration = WIPE_DURATION) {
    return new Promise((resolve) => {
      cancelWipe(element);
      if (!text || !text.trim()) { element.innerHTML = ''; resolve(); return; }

      const padded = isRTL ? ` ${text} ` : text;
      const chars  = [...padded];
      const speed  = Math.max(30, Math.floor(duration / chars.length));
      const CHAR_FADE_MS = 400;
      let revealed = 0, resolved = false;

      element.innerHTML = '';
      element.style.transition = 'none';
      element.style.opacity = '1';
      const span = document.createElement('span');
      element.appendChild(span);

      function finish() {
        if (resolved) return;
        resolved = true;
        clearInterval(interval);
        while (revealed < chars.length) {
          const cs = document.createElement('span');
          cs.textContent = chars[revealed++];
          span.appendChild(cs);
        }
        element.style.transition = '';
        element._wipeInterval = null;
        element._wipeResolve = null;
        resolve();
      }

      const interval = setInterval(() => {
        if (revealed >= chars.length) { finish(); return; }
        const cs = document.createElement('span');
        cs.textContent = chars[revealed++];
        cs.style.opacity = '0';
        cs.style.transition = `opacity ${CHAR_FADE_MS}ms ease-in`;
        span.appendChild(cs);
        void cs.offsetWidth;
        cs.style.opacity = '1';
      }, speed);

      element._wipeInterval = interval;
      element._wipeResolve  = finish;
    });
  }

  function splitIntoLines(element) {
    const outerSpan = element.querySelector(':scope > span');
    if (!outerSpan) return;
    const charSpans = Array.from(outerSpan.children);
    if (charSpans.length === 0) return;
    const lines = [];
    let currentLine = [], lastTop = null;
    charSpans.forEach(cs => {
      const top = cs.getBoundingClientRect().top;
      if (lastTop !== null && Math.abs(top - lastTop) > 2) { lines.push(currentLine); currentLine = []; }
      currentLine.push(cs);
      lastTop = top;
    });
    if (currentLine.length > 0) lines.push(currentLine);
    element.innerHTML = '';
    const lineDivs = [];
    lines.forEach(lineChars => {
      const lineDiv = document.createElement('div');
      lineDiv.className = 'subtitle-line';
      const lineSpan = document.createElement('span');
      lineChars.forEach(cs => lineSpan.appendChild(cs));
      lineDiv.appendChild(lineSpan);
      element.appendChild(lineDiv);
      lineDivs.push(lineDiv);
    });
    void element.offsetWidth;
    lineDivs.forEach(div => {
      const span = div.querySelector(':scope > span');
      if (span) { div.style.height = (span.getBoundingClientRect().height + 4) + 'px'; div.style.overflow = 'visible'; }
    });
  }

  function wipeDisappear(element, isRTL = false, duration = WIPE_DISAPPEAR_DURATION) {
    return new Promise((resolve) => {
      cancelWipe(element);
      const allSpans  = Array.from(element.querySelectorAll('span'));
      const charSpans = allSpans.filter(s => s.children.length === 0 && s.textContent.length <= 2);
      if (charSpans.length === 0) { element.innerHTML = ''; element.style.opacity = '0'; resolve(); return; }
      const speed = Math.max(20, Math.floor(duration / charSpans.length));
      const CHAR_FADE_MS = 300;
      let removed = 0, resolved = false;

      function finish() {
        if (resolved) return;
        resolved = true;
        clearInterval(interval);
        element.innerHTML = '';
        element.style.opacity = '0';
        element.style.transition = '';
        element._wipeInterval = null;
        element._wipeResolve  = null;
        resolve();
      }

      const interval = setInterval(() => {
        if (removed >= charSpans.length) { finish(); return; }
        const cs = charSpans[removed++];
        cs.style.transition = `opacity ${CHAR_FADE_MS}ms ease-out`;
        cs.style.opacity = '0';
      }, speed);

      element._wipeInterval = interval;
      element._wipeResolve  = finish;
    });
  }

  function fadeIn(text, element, duration = FADE_IN_DURATION) {
    return new Promise((resolve) => {
      if (!text || !text.trim()) { element.innerHTML = ''; resolve(); return; }
      element.innerHTML = '';
      const span = document.createElement('span');
      span.textContent = text;
      element.appendChild(span);
      element.style.transition = 'none';
      element.style.opacity = '0';
      void element.offsetWidth;
      element.style.transition = `opacity ${duration}ms ease-in`;
      element.style.opacity = '1';
      setTimeout(() => { element.style.transition = ''; resolve(); }, duration + 50);
    });
  }

  function fadeOut(element, duration = FADE_OUT_DURATION) {
    return new Promise((resolve) => {
      element.style.transition = `opacity ${duration}ms ease-out`;
      element.style.opacity = '0';
      setTimeout(() => { element.style.transition = ''; resolve(); }, duration + 50);
    });
  }

  function resetSubtitles() {
    [subtitlePrimary, subtitleSecondary].forEach(el => {
      cancelWipe(el);
      el.style.transition = 'none';
      el.style.opacity = '0';
      el.innerHTML = '';
    });
  }

  // ─── BACKGROUND ───────────────────────────────────────────────

  function updateBackgroundImage(imageUrl) {
    return new Promise((resolve) => {
      if (!imageUrl) { resolve(); return; }
      const current  = activeBg === 'A' ? bgImageA : bgImageB;
      const incoming = activeBg === 'A' ? bgImageB : bgImageA;
      const preload  = new Image();
      const onReady  = () => {
        incoming.style.transition = 'none';
        incoming.style.opacity    = '0';
        incoming.style.zIndex     = '1';
        current.style.zIndex      = '0';
        incoming.src = imageUrl;
        void incoming.offsetWidth;
        incoming.style.transition = 'opacity 3s ease-in-out';
        incoming.style.opacity    = '1';
        setTimeout(() => { current.style.opacity = '0'; activeBg = activeBg === 'A' ? 'B' : 'A'; resolve(); }, 3000);
      };
      preload.onload = onReady;
      preload.onerror = onReady;
      preload.src = imageUrl;
    });
  }

  // ─── PLAY ONE DRIFT ───────────────────────────────────────────

  async function playOneDrift(version, prefetchedState) {
    const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
    const state   = prefetchedState || await fetchStateAt(version);
    if (!state) return false;

    if (config.l2_enabled) {
      const l2Text = state[`drift_${config.l2_lang}`] || state.drift_ar || '';
      const hasArabic = /[؀-ۿ]/.test(l2Text);
      if (!l2Text || (config.l2_dir === 'rtl' && !hasArabic)) return false;
    }

    resetSubtitles();
    document.querySelectorAll('.subtitle-clone-departing').forEach(el => el.remove());

    const dialoguePairs = buildDialogue(state);
    if (dialoguePairs.length === 0) return false;

    const displaySequence = [];
    for (const pair of dialoguePairs) {
      const enSentences = splitIntoSentences(pair.en);
      const l2Sentences = splitIntoSentences(pair.l2);
      const maxLen = Math.max(enSentences.length, l2Sentences.length);
      for (let i = 0; i < maxLen; i++) {
        displaySequence.push({ en: enSentences[i] || '', l2: l2Sentences[i] || '' });
      }
    }

    for (let i = 0; i < displaySequence.length; i++) {
      const { en, l2 } = displaySequence[i];
      if (i === 0) {
        resetSubtitles();
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      }

      const hasL2 = l2 && l2.trim();
      if (hasL2) {
        await wipeReveal(l2, subtitleSecondary, config.l2_dir === 'rtl', WIPE_DURATION);
        splitIntoLines(subtitleSecondary);
        await fadeIn(en, subtitlePrimary);
      } else {
        await wipeReveal(en, subtitleSecondary, false, WIPE_DURATION);
        splitIntoLines(subtitleSecondary);
      }

      const textLen  = Math.max((en || '').length, (l2 || '').length);
      const holdTime = Math.min(HOLD_MAX, Math.max(HOLD_MIN, textLen * HOLD_MS_PER_CHAR));
      await new Promise(r => setTimeout(r, holdTime));

      const disappearPromises = [];
      if (subtitlePrimary.innerHTML.trim()) disappearPromises.push(fadeOut(subtitlePrimary, FADE_OUT_DURATION));
      if (subtitleSecondary.innerHTML.trim()) disappearPromises.push(wipeDisappear(subtitleSecondary, config.l2_dir === 'rtl', WIPE_DISAPPEAR_DURATION));
      await Promise.all(disappearPromises);
      resetSubtitles();
    }

    resetSubtitles();
    document.querySelectorAll('.subtitle-clone-departing').forEach(el => el.remove());
    return true;
  }

  // ─── MAIN LOOP ────────────────────────────────────────────────

  async function displayLoop() {
    while (true) {
      const versions = await fetchVersionHistory();
      const reversed = [...versions].reverse();

      if (reversed.length === 0) { await new Promise(r => setTimeout(r, 30000)); continue; }

      let firstDrift = true;
      for (const version of reversed) {
        const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
        const state   = await fetchStateAt(version);
        if (!state) continue;

        const imageUrl = state.image_url ? apiBase + state.image_url : null;

        if (firstDrift) {
          if (imageUrl) {
            const bg = activeBg === 'A' ? bgImageA : bgImageB;
            bg.style.transition = 'none';
            bg.style.opacity    = '1';
            await new Promise(resolve => { bg.onload = resolve; bg.onerror = resolve; bg.src = imageUrl; });
          }
          firstDrift = false;
        } else if (imageUrl) {
          await updateBackgroundImage(imageUrl);
        }

        await playOneDrift(version, state);
        await new Promise(r => setTimeout(r, 1000));
      }

      console.log('[Museum] Cycle complete, restarting...');
    }
  }

  // ─── INIT ─────────────────────────────────────────────────────

  async function startStream() {
    if (streamStarted) return;
    streamStarted = true;
    await fetchLangConfig();
    console.log(`[Museum] Initialized: ${PERSONA} | l2_enabled: ${config.l2_enabled}`);
    displayLoop();
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (window.KD_API_CONFIG && window.KD_API_CONFIG.API_BASE_URL != null) {
      startStream();
    } else {
      window.addEventListener('kd-config-ready', () => startStream(), { once: true });
      setTimeout(() => { if (!streamStarted) { console.warn('[Museum] fallback start'); startStream(); } }, 3000);
    }
  });
})();
