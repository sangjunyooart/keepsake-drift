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
        setLanguageUI();
        setMidTextFromStateOrEmpty();
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
      liminal: "#8ED6FF",
      environment: "#7BE3A1",
      digital: "#B8A6FF",
      infrastructure: "#FFB38A",
      more_than_human: "#FFD6F3"
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
      document.documentElement.style.setProperty("--persona-btn-bg", `rgba(${r}, ${g}, ${b}, 0.22)`);
      document.documentElement.style.setProperty("--persona-btn-border", `rgba(${r}, ${g}, ${b}, 0.45)`);
      document.documentElement.style.setProperty("--persona-btn-bg-active", `rgba(${r}, ${g}, ${b}, 0.34)`);
      document.documentElement.style.setProperty("--persona-btn-border-active", `rgba(${r}, ${g}, ${b}, 0.60)`);
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

    // Fade-from-black: reveal stage once initial image loads
    function revealStage() {
      if (stageEl && !stageEl.classList.contains("revealed")) {
        stageEl.classList.add("revealed");
      }
    }

    // Set initial background per persona (before first poll)
    if (bgImage && CFG.personaImageMap && CFG.personaImageMap[currentPersona]) {
      bgImage.src = CFG.personaImageMap[currentPersona];
      bgImage.addEventListener("load", revealStage, { once: true });
      if (bgImage.complete && bgImage.naturalWidth > 0) revealStage();
    } else {
      revealStage();
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
      image_url: null
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
    const RE_U = /([^\W_]+)|([\W_]+)/gu;

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

    function highlightDriftText(rawText, deltaJson, lang) {
      const raw = String(rawText || "");
      if (!raw.trim()) return "";
      const ops = _parseDeltaOps(deltaJson);
      if (!ops) return escapeHtml(raw);

      const changed = _collectChangedWordIndicesFromOps(ops);
      if (!changed || changed.size === 0) return escapeHtml(raw);

      try {
        if (typeof Intl !== "undefined" && Intl.Segmenter) {
          const locale = (lang === "en") ? "en" : (lang || "en");
          const seg = new Intl.Segmenter(locale, { granularity: "word" });
          const parts = [...seg.segment(raw)];
          const wordIdxMap = new Map();
          let wi = 0;
          for (let i = 0; i < parts.length; i++) {
            if (parts[i].isWordLike) { wordIdxMap.set(i, wi); wi++; }
          }
          let out = "";
          let inRun = false;
          for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            const txt = part.segment;
            if (part.isWordLike) {
              const wordIdx = wordIdxMap.get(i);
              const isChanged = changed.has(wordIdx);
              if (isChanged) {
                if (!inRun) { out += `<span class="drift-new">`; inRun = true; }
                out += escapeHtml(txt);
              } else {
                if (inRun) { out += `</span>`; inRun = false; }
                out += escapeHtml(txt);
              }
            } else {
              if (inRun) {
                let nextDrifted = false;
                for (let j = i + 1; j < parts.length; j++) {
                  if (parts[j].isWordLike) { nextDrifted = changed.has(wordIdxMap.get(j)); break; }
                }
                if (nextDrifted) { out += escapeHtml(txt); }
                else { out += `</span>`; inRun = false; out += escapeHtml(txt); }
              } else { out += escapeHtml(txt); }
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
      let wordIdx = 0;
      for (let i = 0; i < tokens.length; i++) {
        if (tokens[i].isWord) { wordIdxMap.set(i, wordIdx); wordIdx++; }
      }
      let out = "";
      let inRun = false;
      for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        if (tok.isWord) {
          const wi2 = wordIdxMap.get(i);
          const isChanged = changed.has(wi2);
          if (isChanged) {
            if (!inRun) { out += `<span class="drift-new">`; inRun = true; }
            out += escapeHtml(tok.text);
          } else {
            if (inRun) { out += `</span>`; inRun = false; }
            out += escapeHtml(tok.text);
          }
        } else {
          if (inRun) {
            let nextDrifted = false;
            for (let j = i + 1; j < tokens.length; j++) {
              if (tokens[j].isWord) { nextDrifted = changed.has(wordIdxMap.get(j)); break; }
            }
            if (nextDrifted) { out += escapeHtml(tok.text); }
            else { out += `</span>`; inRun = false; out += escapeHtml(tok.text); }
          } else { out += escapeHtml(tok.text); }
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
      // Update keepsake handle and fragment input placeholder
      const handleLabel = keepsakeShell && keepsakeShell.querySelector(".keepsake-handle-label");
      if (handleLabel) handleLabel.textContent = (t.keepsakeBtn || "Keepsake");
      if (fragmentInput) fragmentInput.placeholder = (t.placeholder || "leave a memory fragment...");
      updateButtonsAndPanels();
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
      if (driftBtn) driftBtn.textContent = `${t.driftBtnPrefix || "Drift"} ${driftCycle}`;
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

      driftText.innerHTML = (drift && drift.trim())
        ? highlightDriftText(drift, patch, currentLangView)
        : "";
    }

    // ======================================================
    // HOME OVERLAY — Main page as popup
    // ======================================================
    function openHomeOverlay() {
      if (!homeOverlay) return;
      homeOverlay.classList.add("visible");
      homeOverlay.setAttribute("aria-hidden", "false");
      // Update home overlay language labels
      updateHomeLanguageUI();
    }

    function closeHomeOverlay() {
      if (!homeOverlay) return;
      homeOverlay.classList.remove("visible");
      homeOverlay.setAttribute("aria-hidden", "true");
    }

    function updateHomeLanguageUI() {
      if (!homeLangLabel) return;
      // Show opposite language in toggle button (PT for Portuguese)
      const langLabel = (currentLangView === "en")
        ? (secondLang === "pt-br" ? "PT" : secondLang.toUpperCase())
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
        if (diff < 60) return "just now";
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
        return d.toLocaleDateString();
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
        : `${t.driftBtnPrefix || "Drift"} ${entry.version}`;

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
      if (isRTL) textEl.dir = "rtl";

      el.appendChild(header);
      el.appendChild(textEl);

      // Headlines fold (only for drift entries)
      if (!isVisitor) {
        const headlines = entry.curated_headlines || [];
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
          label.textContent = "source";

          toggle.appendChild(arrow);
          toggle.appendChild(label);

          const list = document.createElement("ul");
          list.className = "headlines-list hidden";
          headlines.forEach((h) => {
            const li = document.createElement("li");
            li.textContent = h;
            list.appendChild(li);
          });

          toggle.addEventListener("click", () => {
            const isOpen = list.classList.toggle("visible");
            list.classList.toggle("hidden", !isOpen);
            arrow.classList.toggle("open", isOpen);
          });

          fold.appendChild(toggle);
          fold.appendChild(list);
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
      archiveEntries.innerHTML = "";

      if (!entries || entries.length === 0) {
        archiveEntries.innerHTML = '<div class="archive-empty">No keepsake entries yet. The archive grows with each drift.</div>';
        return;
      }

      // Entries are already oldest-first from backend
      entries.forEach((entry) => {
        archiveEntries.appendChild(buildEntryEl(entry));
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

        const r = await fetch(`${apiBase}/leave_fragment`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            persona: currentPersona,
            text: text,
            session_id: sessionId,
          }),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        const data = await r.json();

        if (data.ok) {
          // Show the reframed visitor entry in the archive immediately
          const reframed = data.reframed || text;
          const reframedAr = data.reframed_ar || "";
          const visitorEntry = {
            source: "visitor",
            version: null,
            keepsake_text: reframed,
            keepsake_text_ar: reframedAr,
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

    async function pollStateOnce() {
      try {
        const apiBase = window.KD_API_CONFIG?.API_BASE_URL || '';
        const r = await fetch(`${apiBase}/state?persona=${encodeURIComponent(currentPersona)}`, { cache: "no-store" });
        if (!r.ok) return;
        const s = await r.json();
        if (!s) return;

        const prevVersion = stateCache.version;
        stateCache.hasState = true;

        // Gate: if the server has a newer version but the countdown clock
        // hasn't reached 00:00 yet, silently pre-load the image but do NOT
        // update text or UI — the page reload at clock-zero will display it.
        const serverVersion = isFiniteNumber(s.version) ? Math.max(0, Number(s.version)) : prevVersion;
        const clockGate = typeof window.kdNextTickMs === "number" ? window.kdNextTickMs : 0;
        const isNewVersion = serverVersion > prevVersion && prevVersion > 0;
        const clockNotReady = clockGate > 0 && Date.now() < (clockGate - 2000); // 2s grace

        if (isNewVersion && clockNotReady) {
          // Pre-load image into browser cache so it's instant on reload
          const earlyImg = (typeof s.image_url === "string" && s.image_url) ? s.image_url : null;
          if (earlyImg) { const p = new Image(); p.src = earlyImg + (earlyImg.includes("?") ? "&" : "?") + "t=" + Date.now(); }
          return; // Skip UI update — let the clock-zero reload handle it
        }

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
            lastImageUrl = imgUrl;
            const ts = Date.now();
            const newSrc = imgUrl + (imgUrl.includes("?") ? "&" : "?") + "t=" + ts;
            bgImage.classList.add("loading");
            const tmpImg = new Image();
            tmpImg.onload = () => { bgImage.src = newSrc; bgImage.classList.remove("loading"); revealStage(); };
            tmpImg.onerror = () => { bgImage.classList.remove("loading"); };
            tmpImg.src = newSrc;

            const prev1 = (typeof s.image_prev1 === "string" && s.image_prev1) ? apiBase + s.image_prev1 : null;
            const prev2 = (typeof s.image_prev2 === "string" && s.image_prev2) ? apiBase + s.image_prev2 : null;
            if (bgPrev1 && prev1) bgPrev1.src = prev1 + (prev1.includes("?") ? "&" : "?") + "t=" + ts;
            if (bgPrev2 && prev2) bgPrev2.src = prev2 + (prev2.includes("?") ? "&" : "?") + "t=" + ts;
          }
        }

        if (typeof s.drift_en === "string") stateCache.drift_en = s.drift_en;
        if (typeof s.drift_ar === "string") stateCache.drift_ar = s.drift_ar;
        if (typeof s.recap_en === "string") stateCache.recap_en = s.recap_en;
        if (typeof s.recap_ar === "string") stateCache.recap_ar = s.recap_ar;
        if (typeof s.keepsake_en === "string") stateCache.keepsake_en = s.keepsake_en;
        if (Array.isArray(s.curated_headlines)) stateCache.curated_headlines = s.curated_headlines;

        if (typeof s.delta_json === "string") stateCache.delta_json = s.delta_json;
        if (typeof s.delta_json_ar === "string") stateCache.delta_json_ar = s.delta_json_ar;

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

    // Check if config is already ready (localhost case) or wait for event (pages.dev)
    if (window.KD_API_CONFIG?.API_BASE_URL) {
      // Config already set (localhost), start immediately
      startPolling();
    } else {
      // Wait for dynamic config to load (Cloudflare Pages)
      window.addEventListener('kd-config-ready', () => {
        console.log('[Keepsake Drift] Config ready, starting state polling');
        startPolling();
      }, { once: true });

      // Fallback: if event doesn't fire within 5 seconds, start anyway
      setTimeout(() => {
        if (!window.KD_API_CONFIG?.API_BASE_URL) {
          console.warn('[Keepsake Drift] Config timeout, starting polling with potentially incomplete config');
        }
        startPolling();
      }, 5000);
    }

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
