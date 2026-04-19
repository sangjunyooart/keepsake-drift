// static/js/chat_runtime.js
(function () {
  const CFG = window.KD_CHAT_CONFIG;
  if (!CFG) {
    console.error("KD_CHAT_CONFIG missing");
    return;
  }

  document.addEventListener("DOMContentLoaded", () => {
    const params = new URLSearchParams(window.location.search);
    let currentPersona = (params.get("persona") || "liminal").trim();
    if (!CFG.VALID_PERSONAS.includes(currentPersona)) currentPersona = "liminal";

    // Store current persona for back navigation from index page
    localStorage.setItem("kd_last_persona", currentPersona);

    // Active second language — fetched from /lang_config (defaults to "ar")
    let secondLang = "ar";
    let secondLangDirection = "rtl";
    let enableTranslation = CFG.ENABLE_ARABIC;

    // Strip emoji, wingdings, dingbats, symbols, and other decorative Unicode from headlines
    function stripEmoji(str) {
      return str.replace(/[\u{1F000}-\u{1FFFF}\u{2600}-\u{27BF}\u{FE00}-\u{FE0F}\u{200D}\u{20E3}\u{E0020}-\u{E007F}\u{2300}-\u{23FF}\u{2B50}\u{2B55}\u{2934}\u{2935}\u{25A0}-\u{25FF}\u{2700}-\u{27BF}\u{2190}-\u{21FF}\u{2022}\u{2023}\u{25AA}\u{25AB}\u{25B6}\u{25C0}\u{25FB}-\u{25FE}\u{2602}-\u{2660}\u{2663}\u{2665}\u{2666}\u{2668}\u{267B}\u{267F}\u{2692}-\u{269F}\u{26A0}-\u{26FF}\u{2702}-\u{27B0}\u{3297}\u{3299}\u{FE0F}\u{200B}-\u{200F}\u{2028}\u{2029}\u{202A}-\u{202E}\u{2060}-\u{206F}]/gu, "").replace(/\s{2,}/g, " ").trim();
    }

    const savedLang = localStorage.getItem(CFG.STORAGE_LANG_KEY);
    let currentLangView = (enableTranslation && savedLang && savedLang !== "en") ? savedLang : "en";

    // Fetch active L2 config from server (non-blocking)
    const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
    fetch(`${apiBase}/lang_config`).then(r => r.json()).then(data => {
      if (data && data.second_lang) {
        secondLang = data.second_lang;
        secondLangDirection = data.direction || "ltr";
        enableTranslation = !!data.enable_translation;
        // Update the EN langButton label in CFG to show the active L2 toggle
        if (data.toggle_label && CFG.uiText && CFG.uiText.en) {
          CFG.uiText.en.langButton = data.toggle_label;
        }
        // If stored lang isn't the active L2, reset to "en"
        if (currentLangView !== "en" && currentLangView !== secondLang) {
          currentLangView = "en";
          localStorage.setItem(CFG.STORAGE_LANG_KEY, currentLangView);
        }
        // Default to L2 when translation is enabled and user has no saved preference
        if (enableTranslation && !savedLang && currentLangView === "en") {
          currentLangView = secondLang;
          localStorage.setItem(CFG.STORAGE_LANG_KEY, currentLangView);
        }
        setLanguageUI();
        setMidTextFromStateOrEmpty();
        // Re-render archive entries if already fetched (lang may have changed)
        if (archiveFetched && archiveData.length > 0) {
          renderArchive(archiveData);
        }
      }
    }).catch(() => {});

    function getOrCreateSessionId(persona) {
      const key = CFG.STORAGE_SESSION_PREFIX + persona;
      let sid = localStorage.getItem(key);
      if (!sid) {
        sid = "sess-" + Math.random().toString(36).slice(2, 10);
        localStorage.setItem(key, sid);
      }
      return sid;
    }
    const sessionId = getOrCreateSessionId(currentPersona);

    // Accent colors (underline + buttons)
    const PERSONA_COLORS = {
      human: "#E6C15A",
      liminal: "#5BC0F0",
      environment: "#4DD88A",
      digital: "#A08EFF",
      infrastructure: "#FF9A60",
      more_than_human: "#F0A0D8"
    };

    function hexToRgb(hex) {
      const h = String(hex || "").replace("#", "").trim();
      const full = (h.length === 3) ? (h[0] + h[0] + h[1] + h[1] + h[2] + h[2]) : h;
      if (full.length !== 6) return { r: 230, g: 193, b: 90 };
      const n = parseInt(full, 16);
      return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
    }

    function setPersonaAccent(persona) {
      const hex = PERSONA_COLORS[persona] || PERSONA_COLORS.human;
      const { r, g, b } = hexToRgb(hex);

      document.documentElement.style.setProperty("--persona-accent", hex);
      document.documentElement.style.setProperty("--persona-accent-rgb", `${r}, ${g}, ${b}`);
      document.documentElement.style.setProperty("--persona-btn-bg", `rgba(${r}, ${g}, ${b}, 0.5)`);
      document.documentElement.style.setProperty("--persona-btn-border", `rgba(${r}, ${g}, ${b}, 0.35)`);
      document.documentElement.style.setProperty("--persona-btn-bg-active", `rgba(${r}, ${g}, ${b}, 0.7)`);
      document.documentElement.style.setProperty("--persona-btn-border-active", `rgba(${r}, ${g}, ${b}, 0.7)`);
    }
    setPersonaAccent(currentPersona);

    const menuIcon = document.getElementById("menuIcon");
    const langToggle = document.getElementById("langToggle") || document.getElementById("langBtn");
    const langLabel = document.getElementById("langLabel");
    const personaTitleChip = document.getElementById("personaTitleChip");

    // Home overlay elements
    const homeOverlay = document.getElementById("homeOverlay");
    const homeCloseBtn = document.getElementById("homeCloseBtn");
    const homeLangToggle = document.getElementById("homeLangToggle");
    const homeLangLabel = document.getElementById("homeLangLabel");

    const driftBtn = document.getElementById("driftBtn");
    const driftPanel = document.getElementById("driftPanel");
    const driftText = document.getElementById("driftText");

    const bgImage = document.getElementById("bgImage");
    const bgPrev1 = document.getElementById("bgPrev1");
    const bgPrev2 = document.getElementById("bgPrev2");
    const stageEl = document.querySelector(".stage");

    // Keepsake shell elements
    const keepsakeShell = document.getElementById("keepsakeShell");
    const keepsakeHandle = document.getElementById("keepsakeHandle");
    const archiveEntries = document.getElementById("archiveEntries");
    const fragmentInput = document.getElementById("fragmentInput");
    const fragmentSendBtn = document.getElementById("fragmentSendBtn");
    const fragmentComposer = document.getElementById("fragmentComposer");

    // No-op: stage is always visible (no fade-in on load)
    function revealStage() {}

    // Set initial background per persona (loads asynchronously, fades in via CSS)
    if (bgImage && CFG.personaImageMap && CFG.personaImageMap[currentPersona]) {
      bgImage.src = CFG.personaImageMap[currentPersona];
    }

    const stateCache = {
      hasState: false,
      version: 0,
      drift_en: "",
      drift_ar: "",
      recap_en: "",
      recap_ar: "",
      keepsake_en: "",
      delta_json: "",
      delta_json_ar: "",
      image_url: null,
      drifted_at: ""
    };

    let driftCycle = 0;
    let midMode = "idle"; // idle | drift
    let lastImageUrl = null;
    let archiveFetched = false;
    let archiveData = [];       // cached archive entries
    let fragmentBusy = false;

    function T() {
      return (CFG.uiText && CFG.uiText[currentLangView]) ? CFG.uiText[currentLangView] : (CFG.uiText.en || {});
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    // ======================================================
    // Underline highlighting (EN + AR)
    // ======================================================
    const RE_EN = /([A-Za-z0-9]+(?:'[A-Za-z0-9]+)?)|([^A-Za-z0-9]+)/g;
    const RE_U = /([\p{L}\p{N}]+)|([^\p{L}\p{N}]+)/gu;

    function _parseDeltaOps(deltaJson) {
      if (!deltaJson) return null;
      let delta;
      try { delta = JSON.parse(deltaJson); } catch { return null; }

      const ops = Array.isArray(delta && delta.token_ops) ? delta.token_ops
               : Array.isArray(delta && delta.ops) ? delta.ops
               : Array.isArray(delta && delta.operations) ? delta.operations
               : null;
      if (!ops || ops.length === 0) return null;
      return ops;
    }

    function _collectChangedWordIndicesFromOps(ops) {
      const changed = new Set();

      for (const op of ops) {
        if (!op) continue;

        const tagRaw = String(op.op || op.type || op.kind || op.action || "").toLowerCase();
        const isChange =
          tagRaw === "insert" || tagRaw === "replace" || tagRaw === "substitute" ||
          tagRaw === "ins" || tagRaw === "rep" || tagRaw === "add" || tagRaw === "added" ||
          tagRaw === "patch" || tagRaw === "edit" || tagRaw === "change";

        const span = Array.isArray(op.cur_span) ? op.cur_span
                   : Array.isArray(op.curSpan) ? op.curSpan
                   : Array.isArray(op.span) ? op.span
                   : null;
        if (!span) continue;

        const a = Number(span[0]);
        const b = Number(span[1]);
        if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
        if (b <= a) continue;

        if (!isChange && tagRaw) {
          if (tagRaw === "equal" || tagRaw === "eq" || tagRaw === "keep" || tagRaw === "same") continue;
        }

        for (let i = a; i < b; i++) changed.add(i);
      }

      return changed;
    }

    // Small connecting words to skip highlighting
    const _STOP_WORDS = new Set([
      // English
      "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of",
      "is", "it", "its", "my", "me", "i", "we", "he", "she", "they",
      "this", "that", "with", "for", "by", "as", "if", "so", "no", "not",
      "be", "am", "are", "was", "were", "has", "had", "do", "did",
      "from", "up", "out", "all", "just", "our", "his", "her",
      "off", "more", "where", "which", "when", "what", "who", "how",
      "than", "then", "there", "here", "into", "over", "some", "such",
      "been", "being", "have", "would", "could", "should", "will",
      "about", "after", "before", "between", "through", "during",
      "each", "every", "any", "own", "same", "other", "another",
      "much", "many", "most", "very", "too", "also", "still", "even",
      "only", "yet", "now", "back", "down", "away", "again",
      // English contractions
      "i'm", "i'd", "i'll", "i've", "it's", "he's", "she's", "we're", "we've", "we'd", "we'll",
      "they're", "they've", "they'd", "they'll", "you're", "you've", "you'd", "you'll",
      "there's", "there'd", "that's", "that'd", "what's", "who's", "who'd",
      "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "couldn't", "shouldn't",
      "doesn't", "didn't", "don't", "hasn't", "hadn't", "haven't", "can't",
      // Arabic
      "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
      "هو", "هي", "هم", "هن", "نحن", "أنا", "أنت", "أنتم",
      "و", "أو", "لكن", "بل", "ثم", "أن", "إن", "لا", "لم", "لن",
      "كان", "كانت", "كانوا", "يكون", "تكون",
      "ما", "التي", "الذي", "الذين", "اللتي", "اللذين",
      "قد", "بعد", "قبل", "بين", "حتى", "عند", "فوق", "تحت",
      "كل", "بعض", "أي", "غير", "أكثر", "جدا",
      "ال", "لل", "بال", "وال",
      // Greek
      "ο", "η", "το", "οι", "τα", "τον", "την", "του", "της", "των", "τους",
      "και", "ή", "αλλά", "αν", "ότι", "όπως", "γιατί", "επειδή",
      "σε", "από", "με", "για", "στο", "στη", "στον", "στην", "στα",
      "είναι", "ήταν", "θα", "να", "δεν", "μη", "μην",
      "αυτό", "αυτή", "αυτός", "αυτά", "αυτοί", "εκείνο", "εκείνη",
      "εγώ", "εσύ", "εμείς", "εσείς", "αυτοί",
      "πολύ", "πιο", "κάθε", "κάποιο", "όλα", "μόνο", "ακόμα",
      "που", "πού", "πώς", "πότε", "τι",
      // Portuguese
      "o", "a", "os", "as", "um", "uma", "uns", "umas",
      "e", "ou", "mas", "porém", "que", "se", "não", "nem",
      "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
      "por", "para", "com", "sem", "sobre", "entre", "até",
      "é", "são", "foi", "era", "eram", "ser", "estar",
      "ele", "ela", "eles", "elas", "eu", "nós", "você", "vocês",
      "esse", "essa", "este", "esta", "isso", "isto", "aquele", "aquela",
      "seu", "sua", "seus", "suas", "meu", "minha",
      "mais", "muito", "cada", "todo", "toda", "todos", "todas",
      "onde", "como", "quando", "qual", "quem",
      "já", "ainda", "também", "só", "apenas",
    ]);

    var _lastHighlightMeaningfulCount = 0;

    function highlightDriftText(rawText, deltaJson, lang) {
      _lastHighlightMeaningfulCount = 0;
      const raw = String(rawText || "");
      if (!raw.trim()) return "";
      const ops = _parseDeltaOps(deltaJson);
      if (!ops) return escapeHtml(raw);

      const changed = _collectChangedWordIndicesFromOps(ops);
      if (!changed || changed.size === 0) return escapeHtml(raw);

      // Helper: is word index a "meaningful" drift (changed AND not a stop word)?
      // A stop word counts as drifted if it bridges two meaningful drifted words
      // within the same sentence.
      // sentenceBreaks: Set of word indices that START a new sentence
      // (i.e., the first word after a sentence-ending punctuation mark).
      function _buildEffectiveSet(wordCount, isChangedFn, wordTextFn, sentenceBreaks) {
        const meaningful = new Set();
        for (let w = 0; w < wordCount; w++) {
          if (isChangedFn(w) && !_STOP_WORDS.has(wordTextFn(w).toLowerCase())) {
            meaningful.add(w);
          }
        }
        // Bridge: a single stop word between two meaningful words joins the run,
        // but NEVER across a sentence boundary.
        const effective = new Set(meaningful);
        for (let w = 1; w < wordCount - 1; w++) {
          if (effective.has(w)) continue;
          if (!_STOP_WORDS.has(wordTextFn(w).toLowerCase())) continue;
          // Don't bridge if this word or the next starts a new sentence
          if (sentenceBreaks.has(w) || sentenceBreaks.has(w + 1)) continue;
          if (meaningful.has(w - 1) && meaningful.has(w + 1)) {
            effective.add(w);
          }
        }
        // Count = number of rendered `.drift-new` bundles (contiguous runs),
        // not total words inside them. A multi-word bundle like
        // "skillfully wheeled on" counts as 1, matching what the viewer sees.
        // Sentence breaks split a run into separate bundles.
        let runCount = 0;
        let inRunCnt = false;
        for (let w = 0; w < wordCount; w++) {
          if (effective.has(w)) {
            if (!inRunCnt || sentenceBreaks.has(w)) {
              runCount++;
              inRunCnt = true;
            }
          } else {
            inRunCnt = false;
          }
        }
        _lastHighlightMeaningfulCount = runCount;
        return effective;
      }

      try {
        // Intl.Segmenter disabled — it tokenizes differently from the backend
        // regex ([^\W_]+), causing highlight indices to land on wrong words.
        // All languages now use the regex path for consistent alignment.
        if (false && lang === "en" && typeof Intl !== "undefined" && Intl.Segmenter) {
          const locale = "en";
          const seg = new Intl.Segmenter(locale, { granularity: "word" });
          const parts = [...seg.segment(raw)];
          const wordIdxMap = new Map();
          const wordParts = []; // wordIdx -> part index
          let wi = 0;
          for (let i = 0; i < parts.length; i++) {
            if (parts[i].isWordLike) { wordIdxMap.set(i, wi); wordParts.push(i); wi++; }
          }
          // Find sentence breaks: word indices that follow sentence-ending punctuation
          // A bare hyphen "-" between two word-like segments is a compound word (post-church, face-down),
          // NOT a sentence break. Only emdash/endash or hyphen with surrounding space counts.
          function _isCompoundHyphen(parts, idx) {
            if (parts[idx].segment !== "-") return false;
            let prevWord = false, nextWord = false;
            for (let b = idx - 1; b >= 0; b--) { if (parts[b].isWordLike) { prevWord = true; break; } if (/\s/.test(parts[b].segment)) break; }
            for (let a = idx + 1; a < parts.length; a++) { if (parts[a].isWordLike) { nextWord = true; break; } if (/\s/.test(parts[a].segment)) break; }
            return prevWord && nextWord;
          }
          const _sentBreaks = new Set();
          let _sawBreak = false;
          for (let i = 0; i < parts.length; i++) {
            if (!parts[i].isWordLike) {
              if (/[.!?,;:\u2014\u2013]/.test(parts[i].segment)) _sawBreak = true;
              else if (parts[i].segment === "-" && !_isCompoundHyphen(parts, i)) _sawBreak = true;
            } else {
              if (_sawBreak) { _sentBreaks.add(wordIdxMap.get(i)); _sawBreak = false; }
            }
          }
          const effective = _buildEffectiveSet(
            wi,
            (w) => changed.has(w),
            (w) => parts[wordParts[w]].segment,
            _sentBreaks
          );
          let out = "";
          let inRun = false;
          let sawSentenceBreak = false;
          for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            const txt = part.segment;
            if (part.isWordLike) {
              const wordIdx = wordIdxMap.get(i);
              if (effective.has(wordIdx)) {
                if (!inRun) { out += `<span class="drift-new">`; inRun = true; }
                out += escapeHtml(txt);
              } else {
                if (inRun) { out += `</span>`; inRun = false; }
                out += escapeHtml(txt);
              }
              sawSentenceBreak = false;
            } else {
              // Punctuation/whitespace: close run
              if (inRun) { out += `</span>`; inRun = false; }
              out += escapeHtml(txt);
              // Track sentence-break punctuation across multiple non-word segments
              // (e.g. "." and " " as separate segments)
              // Bare hyphen between words = compound word (post-church), not a break
              if (/[.!?,;:\u2014\u2013]/.test(txt)) sawSentenceBreak = true;
              else if (txt === "-" && !_isCompoundHyphen(parts, i)) sawSentenceBreak = true;
              // Reopen span if next word is effective, BUT not across sentence boundaries
              if (!inRun && !sawSentenceBreak) {
                let nextEffective = false;
                for (let j = i + 1; j < parts.length; j++) {
                  if (parts[j].isWordLike) {
                    nextEffective = effective.has(wordIdxMap.get(j));
                    break;
                  }
                }
                if (nextEffective) { out += `<span class="drift-new">`; inRun = true; }
              }
            }
          }
          if (inRun) out += `</span>`;
          return out;
        }
      } catch (_) {}

      // Fallback regex
      const re = (lang !== "en") ? RE_U : RE_EN;
      re.lastIndex = 0;
      const tokens = [];
      let m;
      while ((m = re.exec(raw)) !== null) {
        tokens.push({ text: m[0], isWord: (m[1] != null), index: m.index });
      }
      const wordIdxMap = new Map();
      const wordTokens = [];
      let wordIdx = 0;
      for (let i = 0; i < tokens.length; i++) {
        if (tokens[i].isWord) { wordIdxMap.set(i, wordIdx); wordTokens.push(i); wordIdx++; }
      }
      // Sentence breaks for fallback path
      // Bare hyphen between words = compound word, not a break
      function _isCompoundHyphenFB(tokens, idx) {
        if (tokens[idx].text !== "-") return false;
        let prevWord = false, nextWord = false;
        for (let b = idx - 1; b >= 0; b--) { if (tokens[b].isWord) { prevWord = true; break; } if (/\s/.test(tokens[b].text)) break; }
        for (let a = idx + 1; a < tokens.length; a++) { if (tokens[a].isWord) { nextWord = true; break; } if (/\s/.test(tokens[a].text)) break; }
        return prevWord && nextWord;
      }
      const _sentBreaks2 = new Set();
      let _sawBreak2 = false;
      for (let i = 0; i < tokens.length; i++) {
        if (!tokens[i].isWord) {
          if (/[.!?,;:\u2014\u2013]/.test(tokens[i].text)) _sawBreak2 = true;
          else if (tokens[i].text === "-" && !_isCompoundHyphenFB(tokens, i)) _sawBreak2 = true;
        } else {
          if (_sawBreak2) { _sentBreaks2.add(wordIdxMap.get(i)); _sawBreak2 = false; }
        }
      }
      const effective = _buildEffectiveSet(
        wordIdx,
        (w) => changed.has(w),
        (w) => tokens[wordTokens[w]].text,
        _sentBreaks2
      );
      let out = "";
      let inRun = false;
      let sawSentenceBreak2 = false;
      for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        if (tok.isWord) {
          const wi2 = wordIdxMap.get(i);
          if (effective.has(wi2)) {
            if (!inRun) { out += `<span class="drift-new">`; inRun = true; }
            out += escapeHtml(tok.text);
          } else {
            if (inRun) { out += `</span>`; inRun = false; }
            out += escapeHtml(tok.text);
          }
          sawSentenceBreak2 = false;
        } else {
          if (inRun) { out += `</span>`; inRun = false; }
          out += escapeHtml(tok.text);
          if (/[.!?,;:\u2014\u2013]/.test(tok.text)) sawSentenceBreak2 = true;
          else if (tok.text === "-" && !_isCompoundHyphenFB(tokens, i)) sawSentenceBreak2 = true;
          if (!inRun && !sawSentenceBreak2) {
            let nextEffective = false;
            for (let j = i + 1; j < tokens.length; j++) {
              if (tokens[j].isWord) {
                nextEffective = effective.has(wordIdxMap.get(j));
                break;
              }
            }
            if (nextEffective) { out += `<span class="drift-new">`; inRun = true; }
          }
        }
      }
      if (inRun) out += `</span>`;
      return out;
    }

    function getPersonaLabel() {
      const t = T();
      return (t.personaLabel && t.personaLabel[currentPersona]) ? t.personaLabel[currentPersona] : currentPersona;
    }

    function setPersonaTitle() {
      if (personaTitleChip) personaTitleChip.textContent = getPersonaLabel();
    }

    function applyTextDirection() {
      const rtl = (currentLangView !== "en" && secondLangDirection === "rtl");
      if (driftPanel) driftPanel.classList.toggle("rtl", rtl);
      if (fragmentInput) {
        fragmentInput.dir = rtl ? "rtl" : "ltr";
      }
      if (archiveEntries) {
        archiveEntries.dir = rtl ? "rtl" : "ltr";
      }
    }

    function setLanguageUI() {
      const t = T();
      if (langLabel) langLabel.textContent = (t.langButton || "Language");
      setPersonaTitle();
      applyTextDirection();
      // Hide lang toggle when translation is disabled
      if (langToggle) langToggle.style.display = enableTranslation ? "" : "none";
      // Also hide home overlay lang toggle
      const homeLangEl = document.getElementById("homeLangToggle");
      if (homeLangEl) homeLangEl.style.display = enableTranslation ? "" : "none";
      // Update keepsake handle and fragment input placeholder
      const handleLabel = keepsakeShell && keepsakeShell.querySelector(".keepsake-handle-label");
      if (handleLabel) handleLabel.textContent = (t.keepsakeBtn || "Keepsake");
      if (fragmentInput) fragmentInput.placeholder = (t.placeholder || "leave a memory fragment...");
      updateButtonsAndPanels();
    }

    function _checkScrollGradient(panel, predictive) {
      const wrapper = panel && panel.closest('.top-content');
      if (!wrapper) return;
      let scrollable;
      if (predictive) {
        // Panel is collapsed — compare full content height vs target open max-height
        const openMaxHeight = window.innerHeight - 260;
        scrollable = panel.scrollHeight > openMaxHeight;
      } else {
        // Panel is open — standard check
        scrollable = panel.scrollHeight > panel.clientHeight + 2;
      }
      wrapper.classList.toggle('has-scroll', scrollable);
    }

    function _slidePanel(panel, shouldOpen) {
      if (!panel) return;
      if (shouldOpen) {
        // Only reset scroll if panel is currently closed (not already open)
        const wasAlreadyOpen = panel.classList.contains("open");
        panel.classList.remove("hidden");
        // Only reset scroll position when actually opening (not when already open)
        if (!wasAlreadyOpen) {
          panel.scrollTop = 0;
        }
        // Pre-calculate if content will overflow when fully expanded
        _checkScrollGradient(panel, true);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            panel.classList.add("open");
            // Only reset scroll if we just opened it
            if (!wasAlreadyOpen) {
              panel.scrollTop = 0;
            }
          });
        });
      } else {
        if (panel.classList.contains("hidden") && !panel.classList.contains("open")) return;
        panel.classList.remove("open");
        const wrapper = panel.closest('.top-content');
        if (wrapper) wrapper.classList.remove('has-scroll');
        const onEnd = () => {
          if (!panel.classList.contains("open")) panel.classList.add("hidden");
          panel.removeEventListener("transitionend", onEnd);
        };
        panel.addEventListener("transitionend", onEnd);
        setTimeout(() => { if (!panel.classList.contains("open")) panel.classList.add("hidden"); }, 500);
      }
    }

    function updateButtonsAndPanels() {
      if (!driftPanel) return;
      _slidePanel(driftPanel, midMode === "drift");
      if (driftBtn) driftBtn.classList.toggle("active", midMode === "drift");
      const t = T();
      if (driftBtn) {
        const lbl = driftBtn.querySelector(".drift-label");
        if (lbl) lbl.textContent = formatDriftLabel(stateCache.drifted_at, t);
      }
    }

    // Format drift timestamp for display: always "Apr 12, 14:32"
    function formatDriftLabel(isoStr, t) {
      if (!isoStr) return "";
      try {
        const d = new Date(isoStr.replace(" ", "T") + (isoStr.includes("Z") ? "" : "Z"));
        if (isNaN(d)) return "";
        const hh = String(d.getHours()).padStart(2, "0");
        const mm = String(d.getMinutes()).padStart(2, "0");
        const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
        return `${months[d.getMonth()]} ${d.getDate()}, ${hh}:${mm}`;
      } catch (e) { return ""; }
    }

    // Version tracking for word-slide animation
    let _lastRenderedDriftVersion = 0;
    let _lastAnimatedVersion = 0;  // version last shown WITH enter animation
    let _driftAnimPending = false;
    let _hasPlayedEnterAnim = false; // animate only once per page load

    function _applyDriftEnterAnim() {
      const spans = Array.from(driftText.querySelectorAll('.drift-new'));
      if (!spans.length) return;

      // Letters cascade within each word in order, words cascade in reading order.
      // cursor tracks when the next word can start (after the previous word's last letter).
      const LETTER_STAGGER = 45;  // ms between letters within a word
      const WORD_GAP       = 80;  // ms gap after each word before the next begins

      let cursor = 0;

      spans.forEach(s => {
        s.classList.remove('kd-pre-enter');

        // Split into per-letter spans
        const text = s.textContent;
        s.innerHTML = '';

        const letterEls = [];
        for (const char of text) {
          const el = document.createElement('span');
          el.className = 'kd-letter';
          el.textContent = char;
          if (char.trim() === '') {
            el.style.opacity = '1'; // spaces always visible
          } else {
            el.style.opacity = '0';
            letterEls.push(el);
          }
          s.appendChild(el);
        }

        // Schedule each letter in sequence starting at cursor
        letterEls.forEach((el, i) => {
          const t = cursor + i * LETTER_STAGGER;
          setTimeout(() => { el.style.opacity = '1'; }, t);
        });

        // Advance cursor: wait for last letter of this word, then add gap
        cursor += letterEls.length * LETTER_STAGGER + WORD_GAP;
      });

      _lastAnimatedVersion = stateCache.version;
      _hasPlayedEnterAnim = true;
    }

    function _renderDriftHtml(html, playAnim) {
      if (playAnim) {
        // kd-pre-enter is CSS-only (no animation) — spans are invisible the instant they land.
        html = html.replace(/class="drift-new"/g, 'class="drift-new kd-pre-enter"');
      }
      driftText.innerHTML = html;
      _lastRenderedDriftVersion = stateCache.version || _lastRenderedDriftVersion;
      if (playAnim) {
        // rAF lets the invisible frame commit before the animation starts — no flash.
        requestAnimationFrame(_applyDriftEnterAnim);
      }
      if (driftPanel) _checkScrollGradient(driftPanel);
    }

    function setMidTextFromStateOrEmpty() {
      if (!driftText) return;

      const isL2 = (currentLangView !== "en");
      const drift = isL2
        ? (stateCache.drift_ar || stateCache.drift_en || "")
        : (stateCache.drift_en || "");

      const patch = isL2
        ? (stateCache.delta_json_ar && stateCache.delta_json_ar.trim() ? stateCache.delta_json_ar : (stateCache.delta_json || ""))
        : (stateCache.delta_json || "");

      const newHtml = (drift && drift.trim())
        ? highlightDriftText(drift, patch, currentLangView)
        : "";

      // Real-time animate: panel is open, version just changed, spans to exit
      const versionChanged = stateCache.version > 0
        && stateCache.version !== _lastRenderedDriftVersion
        && _lastRenderedDriftVersion > 0;
      const panelOpen = driftPanel && !driftPanel.classList.contains('hidden');
      const exitSpans = versionChanged && !_driftAnimPending && panelOpen
        ? Array.from(driftText.querySelectorAll('.drift-new'))
        : [];

      if (exitSpans.length > 0) {
        _driftAnimPending = true;
        exitSpans.forEach(s => s.classList.add('kd-exiting'));
        setTimeout(() => {
          _driftAnimPending = false;
          _renderDriftHtml(newHtml, _hasPlayedEnterAnim ? false : true);  // enter anim only first time
        }, 920);
      } else if (!_driftAnimPending) {
        _renderDriftHtml(newHtml, false);   // silent update (panel closed or no change)
      }

      // Update drift-meta footer
      const metaEl = document.getElementById("driftMeta");
      if (metaEl) {
        var driftWordCount = _lastHighlightMeaningfulCount;
        const ver = stateCache.version || 0;
        let dateStr = "";
        if (stateCache.drifted_at) {
          const d = new Date(stateCache.drifted_at.replace(" ", "T") + "Z");
          if (!isNaN(d)) {
            const dd = d.toISOString().slice(0, 10).replace(/-/g, ".");
            const hh = String(d.getHours()).padStart(2, "0");
            const mm = String(d.getMinutes()).padStart(2, "0");
            dateStr = dd + " " + hh + ":" + mm;
          }
        }
        const t = T();
        const rtl = (currentLangView !== "en" && secondLangDirection === "rtl");
        const wLabel = driftWordCount === 1 ? (t.wordDrifted || "word drifted") : (t.wordsDrifted || "words drifted");
        const dLabel = t.driftLabel || "drift";
        metaEl.innerHTML = "";
        if (rtl) {
          // Two separate lines to avoid BiDi reordering issues
          // Line 1: word count + drift label
          var line1 = document.createElement("div");
          line1.dir = "rtl";
          line1.style.textAlign = "center";
          var wcParts = [];
          if (driftWordCount > 0) wcParts.push(wLabel + " " + driftWordCount);
          if (ver > 0) wcParts.push(dLabel + " " + ver);
          line1.textContent = wcParts.join(" \u00b7 ");
          metaEl.appendChild(line1);
          // Line 2: timestamp
          if (dateStr) {
            var line2 = document.createElement("div");
            line2.dir = "ltr";
            line2.style.textAlign = "center";
            line2.style.marginTop = "2px";
            line2.textContent = dateStr;
            metaEl.appendChild(line2);
          }
        } else {
          // Line 1: word count + drift label
          var line1 = document.createElement("div");
          line1.style.textAlign = "center";
          var parts = [];
          if (driftWordCount > 0) parts.push(driftWordCount + " " + wLabel);
          if (ver > 0) parts.push(dLabel + " " + ver);
          line1.textContent = parts.join(" \u00b7 ");
          metaEl.appendChild(line1);
          // Line 2: timestamp
          if (dateStr) {
            var line2 = document.createElement("div");
            line2.style.textAlign = "center";
            line2.style.marginTop = "2px";
            line2.textContent = dateStr;
            metaEl.appendChild(line2);
          }
        }
      }

    }

    // ======================================================
    // HOME OVERLAY — Main page as popup
    // ======================================================

    function openHomeOverlay() {
      if (!homeOverlay) return;
      homeOverlay.classList.add("visible");
      homeOverlay.setAttribute("aria-hidden", "false");
      updateHomeLanguageUI();
    }

    function closeHomeOverlay() {
      if (!homeOverlay) return;
      homeOverlay.classList.remove("visible");
      homeOverlay.setAttribute("aria-hidden", "true");
    }

    function updateHomeLanguageUI() {
      if (!homeLangLabel) return;
      // Show opposite language in toggle button
      const langLabel = (currentLangView === "en")
        ? (CFG.uiText.en.langButton || secondLang.toUpperCase())
        : "EN";
      homeLangLabel.textContent = langLabel;

      // Update persona names and descriptions
      const uiText = CFG.uiText[currentLangView];
      if (!uiText) return;

      const homeTitle = document.getElementById("homeTitle");
      const homeSectionLabel = document.getElementById("homePersonaSectionLabel");
      if (homeTitle) homeTitle.textContent = uiText.headerTitle || "Keepsakes ~ Drift";
      if (homeSectionLabel) homeSectionLabel.textContent = uiText.personaSectionLabel || "Choose a temporality";

      // Update all persona cards
      const personas = ["Human", "Liminal", "Env", "Digital", "Infra", "Mth"];
      personas.forEach(p => {
        const mainEl = document.getElementById(`homePersona${p}Main`);
        const subEl = document.getElementById(`homePersona${p}Sub`);
        if (mainEl && uiText.personas && uiText.personas[`${p.toLowerCase()}Main`]) {
          mainEl.textContent = uiText.personas[`${p.toLowerCase()}Main`];
        }
        if (subEl && uiText.personas && uiText.personas[`${p.toLowerCase()}Sub`]) {
          subEl.textContent = uiText.personas[`${p.toLowerCase()}Sub`];
        }
      });

      // Update project description
      const homeProjectDesc = document.getElementById("homeProjectDesc");
      if (homeProjectDesc && uiText.projectDesc) {
        homeProjectDesc.textContent = uiText.projectDesc;
      }

      // Update privacy line
      const homePrivacyLine = document.getElementById("homePrivacyLine");
      if (homePrivacyLine && uiText.privacyLine) {
        homePrivacyLine.innerHTML = "";
        homePrivacyLine.appendChild(document.createTextNode(uiText.privacyLine + " "));
        const link = document.createElement("a");
        link.href = "/privacy";
        link.id = "homePrivacyLink";
        link.textContent = uiText.privacyLinkText || "Data & privacy";
        homePrivacyLine.appendChild(link);
      }
    }

    if (menuIcon) {
      menuIcon.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        openHomeOverlay();
      });
    }

    if (homeCloseBtn) {
      homeCloseBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        closeHomeOverlay();
      });
    }

    // Home overlay persona card navigation
    if (homeOverlay) {
      const homePersonaCards = homeOverlay.querySelectorAll(".home-persona-card");
      homePersonaCards.forEach(card => {
        card.addEventListener("click", () => {
          const persona = card.dataset.persona;
          if (persona) {
            // Navigate to the selected persona
            window.location.href = "/chat?persona=" + encodeURIComponent(persona);
          }
        });
      });
    }

    // Home overlay language toggle
    if (homeLangToggle) {
      homeLangToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!enableTranslation) return;
        currentLangView = (currentLangView === "en") ? secondLang : "en";
        localStorage.setItem(CFG.STORAGE_LANG_KEY, currentLangView);
        setLanguageUI();
        updateHomeLanguageUI();
        setMidTextFromStateOrEmpty();
        // Re-render archive entries in the new language
        if (archiveFetched && archiveData.length > 0) {
          renderArchive(archiveData);
        }
      });
    }

    // Close home overlay when clicking outside
    if (homeOverlay) {
      homeOverlay.addEventListener("click", (e) => {
        if (e.target === homeOverlay) {
          closeHomeOverlay();
        }
      });
    }

    // Countdown clock moved to index.html (main page)

    // Language toggle — stopPropagation so it doesn't close open panels
    if (langToggle) {
      langToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!enableTranslation) return;
        currentLangView = (currentLangView === "en") ? secondLang : "en";
        localStorage.setItem(CFG.STORAGE_LANG_KEY, currentLangView);
        setLanguageUI();
        setMidTextFromStateOrEmpty();
        // Re-render archive entries in the new language
        if (archiveFetched && archiveData.length > 0) {
          renderArchive(archiveData);
        }
      });
    }

    // Drift toggle — close keepsake if opening drift
    if (driftBtn) {
      driftBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const opening = (midMode !== "drift");
        midMode = opening ? "drift" : "idle";
        if (opening && isKeepsakeExpanded()) {
          setKeepsakeExpanded(false);
        }
        setMidTextFromStateOrEmpty();
        updateButtonsAndPanels();
        // Animate .drift-new spans on first panel open only (once per page load).
        if (opening && !_driftAnimPending && !_hasPlayedEnterAnim) {
          driftText.querySelectorAll('.drift-new').forEach(s => s.classList.add('kd-pre-enter'));
          setTimeout(_applyDriftEnterAnim, 200);
        }
      });
    }

    // ======================================================
    // KEEPSAKE SHELL — archive expand/collapse
    // ======================================================
    function setKeepsakeExpanded(expanded) {
      if (!keepsakeShell) return;
      keepsakeShell.classList.toggle("expanded", expanded);

      // Close drift panel when opening keepsake
      if (expanded && midMode === "drift") {
        midMode = "idle";
        updateButtonsAndPanels();
      }

      if (expanded && !archiveFetched) {
        fetchArchive();
      }

      // Clear tag when closing keepsake without submitting
      if (!expanded) setFragmentTag(null);
    }
    function isKeepsakeExpanded() {
      return !!(keepsakeShell && keepsakeShell.classList.contains("expanded"));
    }

    // Handle click: toggle keepsake
    if (keepsakeHandle) {
      keepsakeHandle.addEventListener("click", (e) => {
        e.stopPropagation();
        const opening = !isKeepsakeExpanded();
        setKeepsakeExpanded(opening);
      });

      // Add swipe gesture support for keepsake handle
      let touchStartY = 0;
      let touchEndY = 0;
      const swipeThreshold = 50; // minimum distance for swipe

      keepsakeHandle.addEventListener("touchstart", (e) => {
        touchStartY = e.changedTouches[0].screenY;
      }, { passive: true });

      keepsakeHandle.addEventListener("touchend", (e) => {
        touchEndY = e.changedTouches[0].screenY;
        const swipeDistance = touchStartY - touchEndY;

        // Swipe up (positive distance) = open
        if (swipeDistance > swipeThreshold && !isKeepsakeExpanded()) {
          e.preventDefault();
          setKeepsakeExpanded(true);
        }
        // Swipe down (negative distance) = close
        else if (swipeDistance < -swipeThreshold && isKeepsakeExpanded()) {
          e.preventDefault();
          setKeepsakeExpanded(false);
        }
      }, { passive: false });
    }

    // Input tap: ensure keepsake is expanded
    if (fragmentInput) {
      const ensureKeepsakeOpen = (e) => {
        const wasExpanded = isKeepsakeExpanded();

        if (!wasExpanded) {
          e.preventDefault();
          e.stopPropagation();
          setKeepsakeExpanded(true);
          // Give keepsake time to expand, then focus input
          setTimeout(() => {
            if (fragmentInput) {
              fragmentInput.focus();
            }
          }, 100);
        } else {
          // Already expanded, just ensure focus happens
          // This allows normal input behavior when keepsake is already open
        }
      };

      // Use both touchstart and click for better mobile support
      fragmentInput.addEventListener("touchstart", ensureKeepsakeOpen, { passive: false });
      fragmentInput.addEventListener("click", ensureKeepsakeOpen);

      // Also handle direct tap on input to ensure it's visible
      fragmentInput.addEventListener("touchend", (e) => {
        // Ensure keepsake is expanded on touch end (for mobile)
        if (!isKeepsakeExpanded()) {
          e.preventDefault();
          setKeepsakeExpanded(true);
          setTimeout(() => {
            if (fragmentInput) {
              fragmentInput.focus();
            }
          }, 100);
        }
      }, { passive: false });

      // Handle focus: scroll archive to bottom and highlight input
      fragmentInput.addEventListener("focus", () => {
        // Scroll archive to show latest entry above input
        const archiveEntries = document.querySelector('.archive-entries');
        if (archiveEntries) {
          archiveEntries.scrollTo({
            top: archiveEntries.scrollHeight,
            behavior: 'smooth'
          });
        }

        // Delay to allow keyboard animation to start
        setTimeout(() => {
          if (fragmentInput) {
            fragmentInput.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }
        }, 300);
      });

      // Ensure input stays visible during typing
      fragmentInput.addEventListener("input", () => {
        if (document.activeElement === fragmentInput) {
          fragmentInput.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      });
    }

    // --- Drift-text tagging: tap highlighted word → keepsake tag ---
    const fragmentTag = document.getElementById("fragmentTag");
    const fragmentTagText = document.getElementById("fragmentTagText");
    const fragmentTagClose = document.getElementById("fragmentTagClose");
    let activeTag = null; // the tagged drift text, or null

    function setFragmentTag(text) {
      activeTag = text || null;
      if (fragmentTag && fragmentTagText) {
        if (activeTag) {
          fragmentTagText.textContent = activeTag;
          fragmentTag.style.display = "";
          // RTL-aware tag direction
          const rtl = (currentLangView !== "en" && secondLangDirection === "rtl");
          fragmentTag.dir = rtl ? "rtl" : "ltr";
        } else {
          fragmentTag.style.display = "none";
          fragmentTagText.textContent = "";
        }
      }
      // Adjust collapsed shell height when tag is active
      if (keepsakeShell) {
        keepsakeShell.classList.toggle("has-tag", !!activeTag);
      }
    }

    if (fragmentTagClose) {
      fragmentTagClose.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        setFragmentTag(null);
      });
    }

    // Delegate click on .drift-new spans inside driftText
    // Walks siblings to collect the full connected highlighted phrase
    const driftTextEl = document.getElementById("driftText");
    if (driftTextEl) {
      driftTextEl.addEventListener("click", (e) => {
        const span = e.target.closest(".drift-new");
        if (!span) return;
        e.stopPropagation(); // prevent document click from closing keepsake

        // Collect connected phrase: walk backward then forward from clicked span
        // A "connected" region is adjacent .drift-new spans with only whitespace/punctuation text nodes between them
        const isConnector = (node) => {
          if (!node) return false;
          if (node.nodeType === Node.TEXT_NODE) {
            // Allow whitespace and punctuation between highlighted spans
            return /^[\s,.\-;:!?'"""''—–]+$/.test(node.textContent);
          }
          return false;
        };

        const collected = [span];

        // Walk backward
        let prev = span.previousSibling;
        while (prev) {
          if (prev.nodeType === Node.ELEMENT_NODE && prev.classList && prev.classList.contains("drift-new")) {
            collected.unshift(prev);
            prev = prev.previousSibling;
          } else if (isConnector(prev)) {
            // Check if the node before this connector is also drift-new
            const before = prev.previousSibling;
            if (before && before.nodeType === Node.ELEMENT_NODE && before.classList && before.classList.contains("drift-new")) {
              collected.unshift(prev);
              prev = before;
            } else {
              break;
            }
          } else {
            break;
          }
        }

        // Walk forward
        let next = span.nextSibling;
        while (next) {
          if (next.nodeType === Node.ELEMENT_NODE && next.classList && next.classList.contains("drift-new")) {
            collected.push(next);
            next = next.nextSibling;
          } else if (isConnector(next)) {
            const after = next.nextSibling;
            if (after && after.nodeType === Node.ELEMENT_NODE && after.classList && after.classList.contains("drift-new")) {
              collected.push(next);
              next = after;
            } else {
              break;
            }
          } else {
            break;
          }
        }

        // Build phrase from collected nodes
        const phrase = collected.map(n => n.textContent).join("").trim();
        if (!phrase) return;
        setFragmentTag(phrase);
        // Open keepsake and focus input
        setKeepsakeExpanded(true);
        if (fragmentInput) {
          setTimeout(() => {
            fragmentInput.focus();
            fragmentInput.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }, 350);
        }
      });
    }

    // Send button
    if (fragmentSendBtn) {
      fragmentSendBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!isKeepsakeExpanded()) setKeepsakeExpanded(true);
        submitFragment();
      });
    }

    // Clicks inside shell: don't close
    if (keepsakeShell) {
      keepsakeShell.addEventListener("click", (e) => { e.stopPropagation(); });
    }

    // Tap anywhere outside: close if open
    document.addEventListener("click", () => {
      if (isKeepsakeExpanded()) setKeepsakeExpanded(false);
    });

    // Enter key sends fragment
    if (fragmentInput) {
      fragmentInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          if (!isKeepsakeExpanded()) setKeepsakeExpanded(true);
          submitFragment();
        }
      });
    }

    // ======================================================
    // ARCHIVE: fetch + render
    // ======================================================

    function relativeTime(isoStr) {
      if (!isoStr) return "";
      try {
        const d = new Date(isoStr.includes("T") ? isoStr : isoStr.replace(" ", "T") + "Z");
        const diff = (Date.now() - d.getTime()) / 1000;
        const t = T();
        if (diff < 60) return t.timeJustNow || "just now";
        if (diff < 3600) return `${Math.floor(diff / 60)}${t.timeMinAgo || "m ago"}`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}${t.timeHrAgo || "h ago"}`;
        if (diff < 604800) return `${Math.floor(diff / 86400)}${t.timeDayAgo || "d ago"}`;
        return d.toLocaleDateString(currentLangView === "en" ? "en" : secondLang);
      } catch (_) { return isoStr; }
    }

    function showArchiveLoading() {
      if (!archiveEntries) return;
      archiveEntries.innerHTML = '<div class="archive-loading"><span class="typing-dots"><span></span><span></span><span></span></span></div>';
    }

    async function fetchArchive() {
      archiveFetched = true;
      showArchiveLoading();
      try {
        const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
        const r = await fetch(`${apiBase}/keepsake_archive?persona=${encodeURIComponent(currentPersona)}`, { cache: "no-store" });
        if (!r.ok) { renderArchive([]); return; }
        const data = await r.json();
        archiveData = data.entries || [];
        renderArchive(archiveData);
      } catch (_) {
        renderArchive([]);
      }
    }

    function getEntryText(entry) {
      if (currentLangView !== "en") {
        const l2 = (entry.keepsake_text_ar || "").trim();
        if (l2) return l2;
      }
      return entry.keepsake_text || "";
    }

    function buildEntryEl(entry) {
      const isVisitor = (entry.source === "visitor");
      const el = document.createElement("div");
      el.className = "keepsake-entry" + (isVisitor ? " visitor" : "");

      const isL2 = (currentLangView !== "en");
      const isRTL = (isL2 && secondLangDirection === "rtl");

      // Header: label + time
      const header = document.createElement("div");
      header.className = "keepsake-entry-header";

      const t = T();
      const versionBadge = document.createElement("span");
      versionBadge.className = "keepsake-entry-version";
      versionBadge.textContent = isVisitor
        ? (isL2 ? (t.you || "Visitor") : "Visitor")
        : formatDriftLabel(entry.created_at, t);

      const timeEl = document.createElement("span");
      timeEl.className = "keepsake-entry-time";
      timeEl.textContent = relativeTime(entry.created_at);

      header.appendChild(versionBadge);
      header.appendChild(timeEl);

      // Text — pick L2 if available and language is L2
      const textEl = document.createElement("p");
      textEl.className = "keepsake-entry-text";
      const displayText = getEntryText(entry);
      textEl.textContent = displayText;
      // Append #tag at the end, translated for current language view
      const tagEn = (entry.tagged_text || "").trim();
      const tagAr = (entry.tagged_text_ar || "").trim();
      const tagDisplay = (isL2 && tagAr) ? tagAr : tagEn;
      if (tagDisplay) {
        textEl.appendChild(document.createTextNode(" "));
        const suffix = document.createElement("span");
        suffix.className = "keepsake-tag-suffix";
        suffix.textContent = "#" + tagDisplay;
        textEl.appendChild(suffix);
      }
      if (isRTL) textEl.dir = "rtl";

      el.appendChild(header);
      el.appendChild(textEl);

      // Confidence / hallucination state
      if (typeof entry.resonance === "number") {
        const res = entry.resonance;
        const stateEl = document.createElement("div");
        stateEl.className = "keepsake-confidence";
        let stateCls;
        if (res < 0.3) {
          stateCls = "hallucinating";
        } else if (res < 0.6) {
          stateCls = "drifting";
        } else {
          stateCls = "grounded";
        }
        const stateLabel = stateCls === "grounded" ? (t.confidenceGrounded || "grounded")
          : stateCls === "drifting" ? (t.confidenceDrifting || "drifting")
          : (t.confidenceHallucinating || "hallucinating");
        stateEl.classList.add(stateCls);
        const dot = document.createElement("span");
        dot.className = "confidence-dot";
        const lbl = document.createElement("span");
        lbl.className = "confidence-label";
        lbl.textContent = stateLabel;
        const pct = document.createElement("span");
        pct.className = "confidence-pct";
        pct.textContent = Math.round(res * 100) + "%";
        stateEl.appendChild(dot);
        stateEl.appendChild(lbl);
        stateEl.appendChild(pct);
        el.appendChild(stateEl);
      }

      // Headlines fold (only for drift entries)
      if (!isVisitor) {
        const headlines_en = entry.curated_headlines || [];
        const headlines_ar = entry.curated_headlines_ar || [];
        const isL2 = (currentLangView !== "en");
        const headlines = (isL2 && headlines_ar.length > 0) ? headlines_ar : headlines_en;
        console.log('[KD] headlines:', currentLangView, 'isL2:', isL2, 'ar:', headlines_ar.length, 'using:', isL2 && headlines_ar.length > 0 ? 'L2' : 'EN');
        if (headlines.length > 0) {
          const fold = document.createElement("div");
          fold.className = "headlines-fold";

          const toggle = document.createElement("button");
          toggle.className = "headlines-toggle";

          const arrow = document.createElement("span");
          arrow.className = "fold-arrow";
          arrow.innerHTML = "&#9654;";

          const label = document.createElement("span");
          label.className = "fold-label";
          label.textContent = T().sourceLabel || "source";

          toggle.appendChild(arrow);
          toggle.appendChild(label);

          const list = document.createElement("ul");
          list.className = "headlines-list hidden";
          if (isL2 && secondLangDirection === "rtl") list.setAttribute("dir", "rtl");
          headlines.forEach((h) => {
            const clean = stripEmoji(h);
            if (!clean) return;
            const li = document.createElement("li");
            li.textContent = clean;
            list.appendChild(li);
          });

          toggle.addEventListener("click", () => {
            const isOpen = list.classList.toggle("visible");
            list.classList.toggle("hidden", !isOpen);
            arrow.classList.toggle("open", isOpen);
          });

          fold.appendChild(toggle);
          fold.appendChild(list);

          // Note when AI-interpreted non-headline data contributed
          const rtCount = entry.realtime_source_count || (entry.realtime_interpretations || []).length || 0;
          if (rtCount > 0) {
            const rtNote = document.createElement("div");
            rtNote.className = "rt-source-note hidden";
            rtNote.textContent = "AI-interpreted non-headline data";
            list.parentNode.insertBefore(rtNote, list.nextSibling);
            // Show/hide with the fold
            const origToggle = toggle.onclick;
            toggle.addEventListener("click", () => {
              rtNote.classList.toggle("hidden", list.classList.contains("hidden"));
            });
          }

          el.appendChild(fold);
        }
      }

      return el;
    }

    function scrollArchiveToBottom() {
      if (archiveEntries) {
        archiveEntries.scrollTop = archiveEntries.scrollHeight;
      }
    }

    function renderArchive(entries) {
      if (!archiveEntries) return;

      // Snapshot which folds are open (by index) before destroying DOM
      const openFolds = new Set();
      archiveEntries.querySelectorAll(".keepsake-entry").forEach((el, idx) => {
        const list = el.querySelector(".headlines-list");
        if (list && list.classList.contains("visible")) {
          openFolds.add(idx);
        }
      });

      archiveEntries.innerHTML = "";

      if (!entries || entries.length === 0) {
        archiveEntries.innerHTML = '<div class="archive-empty">No keepsake entries yet. The archive grows with each drift.</div>';
        return;
      }

      // Entries are already oldest-first from backend
      entries.forEach((entry, idx) => {
        const el = buildEntryEl(entry);
        // Restore fold state if it was open before re-render
        if (openFolds.has(idx)) {
          const list = el.querySelector(".headlines-list");
          const arrow = el.querySelector(".fold-arrow");
          if (list) {
            list.classList.remove("hidden");
            list.classList.add("visible");
          }
          if (arrow) arrow.classList.add("open");
        }
        archiveEntries.appendChild(el);
      });

      // Scroll to bottom to show newest entry
      requestAnimationFrame(() => scrollArchiveToBottom());
    }

    // ======================================================
    // FRAGMENT SUBMISSION
    // ======================================================

    function lockComposer() {
      if (fragmentInput) fragmentInput.disabled = true;
      if (fragmentSendBtn) fragmentSendBtn.disabled = true;
      if (fragmentComposer) fragmentComposer.classList.add("disabled");
    }
    function unlockComposer() {
      if (fragmentInput) fragmentInput.disabled = false;
      if (fragmentSendBtn) fragmentSendBtn.disabled = false;
      if (fragmentComposer) fragmentComposer.classList.remove("disabled");
    }

    let _fbTimer = null;
    let _fbTimer2 = null;
    function showFeedback(msg, isError, duration) {
      // Insert or reuse feedback element above composer
      let fb = document.getElementById("fragmentFeedback");
      if (!fb) {
        fb = document.createElement("div");
        fb.id = "fragmentFeedback";
        fb.className = "fragment-feedback";
        if (fragmentComposer && fragmentComposer.parentNode) {
          fragmentComposer.parentNode.insertBefore(fb, fragmentComposer);
        }
      }
      if (_fbTimer) { clearTimeout(_fbTimer); _fbTimer = null; }
      if (_fbTimer2) { clearTimeout(_fbTimer2); _fbTimer2 = null; }

      // Reset state and set content
      fb.classList.remove("visible", "slide-out", "error");
      fb.textContent = msg;
      fb.classList.toggle("error", !!isError);

      // Slide in on next frame
      requestAnimationFrame(() => {
        fb.classList.add("visible");
      });

      // Schedule slide-out
      const ms = duration || (isError ? 4000 : 2500);
      _fbTimer = setTimeout(() => {
        fb.classList.remove("visible");
        fb.classList.add("slide-out");
        _fbTimer2 = setTimeout(() => { fb.classList.remove("slide-out"); }, 500);
      }, ms);
    }

    function showListeningPopup() {
      let popup = document.getElementById("listeningPopup");
      if (!popup) {
        popup = document.createElement("div");
        popup.id = "listeningPopup";
        popup.className = "listening-popup";
        popup.innerHTML = `
          <div class="listening-popup-text">
            <span>listening</span>
            <span class="listening-dots">
              <span></span>
              <span></span>
              <span></span>
            </span>
          </div>
        `;
        document.body.appendChild(popup);
      }
      requestAnimationFrame(() => {
        popup.classList.add("visible");
      });
      return popup;
    }

    function hideListeningPopup() {
      const popup = document.getElementById("listeningPopup");
      if (popup) {
        popup.classList.remove("visible");
      }
    }

    async function submitFragment() {
      if (!fragmentInput || fragmentBusy) return;
      const text = (fragmentInput.value || "").trim();
      if (!text) return;

      fragmentBusy = true;
      fragmentInput.value = "";
      lockComposer();

      // Show listening popup
      showListeningPopup();

      try {
        const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';

        // Create AbortController for timeout (60 seconds for AI processing)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const fragBody = {
            persona: currentPersona,
            text: text,
            session_id: sessionId,
        };
        if (activeTag) {
          fragBody.tagged_text = activeTag;
          fragBody.tagged_lang = currentLangView;
        }

        const r = await fetch(`${apiBase}/leave_fragment`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(fragBody),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        const data = await r.json();

        if (data.ok) {
          setFragmentTag(null); // clear tag after successful submit
          // Show the reframed visitor entry in the archive immediately
          const reframed = data.reframed || text;
          const reframedAr = data.reframed_ar || "";
          const visitorEntry = {
            source: "visitor",
            version: null,
            keepsake_text: reframed,
            keepsake_text_ar: reframedAr,
            tagged_text: data.tagged_text || "",
            tagged_text_ar: data.tagged_text_ar || "",
            resonance: (typeof data.resonance === "number") ? data.resonance : undefined,
            created_at: new Date().toISOString(),
            curated_headlines: [],
          };
          archiveData.push(visitorEntry);

          if (archiveEntries) {
            // Remove empty state if present
            const emptyEl = archiveEntries.querySelector(".archive-empty");
            if (emptyEl) emptyEl.remove();

            archiveEntries.appendChild(buildEntryEl(visitorEntry));
            requestAnimationFrame(() => scrollArchiveToBottom());
          }

          // Open shell if not already expanded so user sees the entry
          if (!isKeepsakeExpanded()) {
            setKeepsakeExpanded(true);
          }

          // Hide listening popup when entry appears in archive
          hideListeningPopup();
        } else if (data.trivial) {
          // Non-meaningful input — brief neutral acknowledgment, not archived
          hideListeningPopup();
          showFeedback(data.message || "Noted.", false, 2500);
        } else {
          hideListeningPopup();
          showFeedback(data.message || "Could not leave fragment.", true);
        }
      } catch (err) {
        hideListeningPopup();
        showFeedback("Network error. Try again.", true);
      } finally {
        fragmentBusy = false;
        unlockComposer();
        if (fragmentInput) fragmentInput.focus();
      }
    }

    // ======================================================
    // POLLING /state
    // ======================================================
    function isFiniteNumber(v) { return typeof v === "number" && Number.isFinite(v); }

    let _kdConsecutiveFailures = 0;

    function _kdUpdateFeedIndicator(feedStatus, latencyMs) {
      let ind = document.getElementById('kdFeedIndicator');
      if (!ind) {
        ind = document.createElement('div');
        ind.id = 'kdFeedIndicator';
        ind.style.cssText = 'position:fixed;top:10px;right:10px;width:7px;height:7px;border-radius:50%;opacity:0.7;z-index:9999;transition:background-color 0.5s ease;';
        document.body.appendChild(ind);
      }
      const colors = { live: '#7BE3A1', cached: '#E6C15A', offline: '#FF6B6B', unknown: 'rgba(255,255,255,0.2)' };
      ind.style.backgroundColor = colors[feedStatus] || colors.unknown;
      ind.title = (feedStatus || 'unknown').toUpperCase() + ' / ' + latencyMs + 'ms';
    }

    async function pollStateOnce() {
      const t0 = performance.now();
      try {
        const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
        const r = await fetch(`${apiBase}/state?persona=${encodeURIComponent(currentPersona)}`, { cache: "no-store" });
        const latencyMs = Math.round(performance.now() - t0);

        if (!r.ok) {
          _kdConsecutiveFailures++;
          if (_kdConsecutiveFailures >= 2) document.body.classList.add('kd-disrupted');
          return;
        }
        _kdConsecutiveFailures = 0;
        document.body.classList.remove('kd-disrupted');

        const s = await r.json();
        if (!s) return;

        // Network visibility: feed status indicator
        _kdUpdateFeedIndicator(s.feed_status || 'unknown', latencyMs);

        // Resonance-driven CSS variable (drives subtle text looseness)
        const resonance = (typeof s.resonance === "number") ? s.resonance : 0.5;
        document.documentElement.style.setProperty('--resonance', resonance.toFixed(3));

        const prevVersion = stateCache.version;
        stateCache.hasState = true;

        // Continuous drift: no clock-boundary gating. Apply updates as they arrive.
        if (isFiniteNumber(s.version)) {
          stateCache.version = Math.max(0, Number(s.version));
          driftCycle = stateCache.version;
        }

        // Background image (apiBase already declared at function start)
        const imgUrl = (typeof s.image_url === "string" && s.image_url)
          ? apiBase + s.image_url
          : (CFG.personaImageMap && CFG.personaImageMap[currentPersona]) || null;

        if (imgUrl) {
          stateCache.image_url = imgUrl;
          if (bgImage && imgUrl !== lastImageUrl) {
            const isFirstLoad = !lastImageUrl;
            lastImageUrl = imgUrl;
            // Mobile: request smaller JPEG via mobile=1 param
            const isMobile = window.innerWidth <= 768;
            const mobileSuffix = isMobile ? (imgUrl.includes("?") ? "&mobile=1" : "?mobile=1") : "";
            // No extra ?t= — the version number in the filename (human_v67.png) is the
            // cache key. Cloudflare caches the JPEG at the edge; when a new image is
            // generated the filename changes, busting the cache automatically.
            const newSrc = imgUrl + mobileSuffix;

            const prev1 = (typeof s.image_prev1 === "string" && s.image_prev1) ? apiBase + s.image_prev1 : null;
            const prev2 = (typeof s.image_prev2 === "string" && s.image_prev2) ? apiBase + s.image_prev2 : null;

            if (isFirstLoad) {
              bgImage.src = newSrc;
              if (!isMobile) {
                bgImage.addEventListener("load", function _loadPrev() {
                  bgImage.removeEventListener("load", _loadPrev);
                  if (bgPrev1 && prev1) bgPrev1.src = prev1;
                  if (bgPrev2 && prev2) bgPrev2.src = prev2;
                }, { once: true });
              }
            } else {
              bgImage.classList.add("loading");
              const tmpImg = new Image();
              tmpImg.onload = () => { bgImage.src = newSrc; bgImage.classList.remove("loading"); };
              tmpImg.onerror = () => { bgImage.classList.remove("loading"); };
              tmpImg.src = newSrc;
              if (!isMobile && bgPrev1 && prev1) bgPrev1.src = prev1;
              if (!isMobile && bgPrev2 && prev2) bgPrev2.src = prev2;
            }
          }
        }

        if (typeof s.drift_en === "string") stateCache.drift_en = s.drift_en;
        if (typeof s.drift_ar === "string") stateCache.drift_ar = s.drift_ar;
        if (typeof s.recap_en === "string") stateCache.recap_en = s.recap_en;
        if (typeof s.recap_ar === "string") stateCache.recap_ar = s.recap_ar;
        if (typeof s.keepsake_en === "string") stateCache.keepsake_en = s.keepsake_en;
        if (Array.isArray(s.curated_headlines)) stateCache.curated_headlines = s.curated_headlines;
        if (typeof s.realtime_source_count === "number") stateCache.realtime_source_count = s.realtime_source_count;

        if (typeof s.delta_json === "string") stateCache.delta_json = s.delta_json;
        if (typeof s.delta_json_ar === "string") stateCache.delta_json_ar = s.delta_json_ar;
        if (typeof s.drifted_at === "string") stateCache.drifted_at = s.drifted_at;

        setMidTextFromStateOrEmpty();
        updateButtonsAndPanels();

        // If version changed and archive is open, re-fetch to get new entry
        if (archiveFetched && stateCache.version > prevVersion && prevVersion > 0) {
          fetchArchive();
        }
      } catch (_) {}
    }

    // Init
    setPersonaTitle();
    setLanguageUI();
    updateButtonsAndPanels();

    // Wait for API config to be ready (important for Cloudflare Pages dynamic tunnel URL)
    function startPolling() {
      pollStateOnce();
      setInterval(pollStateOnce, CFG.STATE_POLL_MS || 10000);
    }

    // Start polling immediately — KD_API_CONFIG is set synchronously by api_config.js
    // (API_BASE_URL can be '' for same-origin on keepsake-drift.net, which is valid)
    startPolling();

    // Prevent pull-to-refresh on mobile browsers
    let lastTouchY = 0;
    const appShell = document.querySelector('.app-shell');
    const stage = document.querySelector('.stage');

    if (appShell) {
      // Track touch start position
      appShell.addEventListener('touchstart', (e) => {
        lastTouchY = e.touches[0].clientY;
      }, { passive: true });

      // Prevent pull-to-refresh when scrolled to top
      appShell.addEventListener('touchmove', (e) => {
        const touchY = e.touches[0].clientY;
        const touchDelta = touchY - lastTouchY;

        // Check if touch is inside scrollable elements (allow scrolling there)
        const target = e.target;
        const archiveEntriesEl = target.closest('.archive-entries');
        const driftPanelEl = target.closest('.text-panel');

        if (archiveEntriesEl || driftPanelEl) {
          // Allow normal scrolling within archive and drift panel
          return;
        }

        // If user is pulling down (positive delta) and stage is at scroll top
        if (touchDelta > 0 && stage && stage.scrollTop === 0) {
          // Prevent the default pull-to-refresh behavior
          e.preventDefault();
        }
      }, { passive: false });
    }

    // iOS keyboard handling - adjust viewport when keyboard appears
    // iOS keyboard handling with Visual Viewport API
    if (window.visualViewport && fragmentInput) {
      const appHeader = document.querySelector('.app-header');
      const archiveEntriesWrap = document.querySelector('.archive-entries-wrap');

      const handleViewportResize = () => {
        // When keyboard opens, the visual viewport shrinks
        if (keepsakeShell && document.activeElement === fragmentInput) {
          const viewportHeight = window.visualViewport.height;
          const offsetTop = window.visualViewport.offsetTop;

          // Calculate available space
          const headerHeight = appHeader ? appHeader.offsetHeight : 52;
          const keepsakeHandleHeight = 42;
          const composerHeight = 70; // Input + padding
          const buffer = 8;

          // Total fixed elements height
          const fixedHeight = headerHeight + keepsakeHandleHeight + composerHeight + buffer;

          // Calculate available height for archive
          const availableHeight = viewportHeight - offsetTop - fixedHeight;

          // Set archive height to shrink, keeping header/handle/composer visible
          if (archiveEntriesWrap && availableHeight > 100) {
            archiveEntriesWrap.style.maxHeight = `${availableHeight}px`;
          }

          // Adjust keepsake shell total height
          keepsakeShell.style.maxHeight = `${viewportHeight - offsetTop}px`;
        }
      };

      const resetViewport = () => {
        // Reset when keyboard closes
        if (keepsakeShell) {
          keepsakeShell.style.maxHeight = '';
        }
        if (archiveEntriesWrap) {
          archiveEntriesWrap.style.maxHeight = '';
        }
      };

      window.visualViewport.addEventListener('resize', handleViewportResize);
      window.visualViewport.addEventListener('scroll', handleViewportResize);

      if (fragmentInput) {
        fragmentInput.addEventListener('blur', resetViewport);
      }
    }
  });
})();
