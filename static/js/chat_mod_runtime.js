// static/js/chat_mod_runtime.js
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const driftPrev = $("driftPrev");
  const driftNext = $("driftNext");
  const keepPrev  = $("keepPrev");
  const keepNext  = $("keepNext");

  const driftLabel = $("driftLabel");
  const keepLabel  = $("keepLabel");
  const driftVer   = $("driftVer");
  const keepVer    = $("keepVer");

  const personaSelect = $("personaSelect");
  const comparePill   = $("comparePill");
  const swapSides     = $("swapSides");

  const modePill      = $("modePill");
  const toggleDelta   = $("toggleDelta");

  const langPill      = $("langPill");
  const toggleLang    = $("toggleLang");

  const leftMeta   = $("leftMeta");
  const rightMeta  = $("rightMeta");
  const leftStamp  = $("leftStamp");
  const rightStamp = $("rightStamp");
  const leftMode   = $("leftMode");
  const rightMode  = $("rightMode");
  const leftText   = $("leftText");
  const rightText  = $("rightText");

  // Trigger info elements
  const triggerVer      = $("triggerVer");
  const triggerEvents   = $("triggerEvents");
  const triggerSegments = $("triggerSegments");
  const triggerAnchors  = $("triggerAnchors");

  // Image prompt elements
  const imagePromptPanel    = $("imagePromptPanel");
  const imagePromptPersona  = $("imagePromptPersona");
  const imagePromptVer      = $("imagePromptVer");
  const imagePromptThumb    = $("imagePromptThumb");
  const imagePromptText     = $("imagePromptText");

  // ---- config
  const KD = window.KD_CHAT_CONFIG || {};
  const PERSONAS = Array.isArray(KD.VALID_PERSONAS) && KD.VALID_PERSONAS.length
    ? KD.VALID_PERSONAS
    : ["human","liminal","environment","digital","infrastructure","more_than_human"];

  // ---- state
  const url = new URL(location.href);
  let persona = url.searchParams.get("persona") || "human";

  let versions = [0];
  let minV = 0;
  let maxV = 0;
  let curV = 0;

  let mode = "DRIFT";     // DRIFT or KEEPSAKE
  let lang = localStorage.getItem("keepsake_lang") || "en";  // Read from localStorage, default to en
  let swapped = false;
  let showDelta = false;

  const fmtVer = (v) => `v${v}`;

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function setRTL(el, on) {
    el.style.direction = on ? "rtl" : "ltr";
    el.style.textAlign = on ? "right" : "left";
  }

  async function apiJSON(path) {
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) {
      const t = await r.text().catch(() => "");
      throw new Error(`${r.status} ${r.statusText} ${t}`.trim());
    }
    return await r.json();
  }

  function initPersonaSelect() {
    personaSelect.innerHTML = "";
    for (const key of PERSONAS) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = key;
      personaSelect.appendChild(opt);
    }
    if (!PERSONAS.includes(persona)) persona = PERSONAS[0];
    personaSelect.value = persona;
  }

  async function loadVersions() {
    const out = await apiJSON(`/versions?persona=${encodeURIComponent(persona)}`);
    versions = Array.isArray(out.versions) && out.versions.length ? out.versions : [0];
    minV = Number(out.min || 0);
    maxV = Number(out.max || 0);
    curV = Number(out.latest || maxV || 0);
    if (Number.isNaN(curV)) curV = 0;
  }

  function updateControls() {
    const canOlder = curV > minV;
    const canNewer = curV < maxV;

    driftPrev.disabled = !canOlder;
    driftNext.disabled = !canNewer;
    keepPrev.disabled  = !canOlder;
    keepNext.disabled  = !canNewer;

    driftVer.textContent = fmtVer(curV);
    keepVer.textContent  = fmtVer(curV);

    modePill.textContent = mode;
    comparePill.textContent = swapped ? "n vs n-1" : "n-1 vs n";
    langPill.textContent = (lang === "ar") ? "AR" : "EN";

    toggleDelta.textContent = showDelta ? "text" : "delta";
  }

  async function fetchAt(v) {
    return await apiJSON(`/state_at?persona=${encodeURIComponent(persona)}&version=${encodeURIComponent(String(v))}`);
  }

  function pick(stateObj) {
    const base = {
      delta_en: stateObj.delta_json || "",
      delta_ar: stateObj.delta_json_ar || "",
      stamp: stateObj.created_at || "",
      selected_segments: stateObj.selected_segments || [],
      sensory_anchors: stateObj.sensory_anchors || [],
      invariables: stateObj.invariables || [],
      event_ids: stateObj.event_ids || [],
      event_titles: stateObj.event_titles || [],
      drifted_keywords: stateObj.drifted_keywords || [],
      justification_en: stateObj.justification_en || "",
      justification_ar: stateObj.justification_ar || "",
      drift_direction: stateObj.drift_direction || "",
      prev_drift_en: stateObj.prev_drift_en || "",
      keepsake_en: stateObj.keepsake_en || "",
      image_prompt: stateObj.image_prompt || "",
      image_url: stateObj.image_url || "",
    };
    if (mode === "DRIFT") {
      return { ...base, en: stateObj.drift_en || "", ar: stateObj.drift_ar || "", label: "drift_text" };
    }
    const keepEn = (stateObj.keepsake_en || "").trim();
    return {
      ...base,
      en: keepEn ? keepEn : (stateObj.recap_en || ""),
      ar: stateObj.recap_ar || "",
      label: keepEn ? "keepsake_text" : "summary_text",
    };
  }

  // ---- Semantic delta underlines (replacement-region aware) ----

  const RE_EN_TOK = /([A-Za-z0-9]+(?:'[A-Za-z0-9]+)?)|([^A-Za-z0-9]+)/g;
  const RE_U_TOK  = /([^\W_]+)|([\W_]+)/gu;

  /**
   * Render underlines ONLY on tokens that belong to drifted replacement regions.
   * Reads delta_json produced by build_segment_aware_delta() on the backend.
   * Uses connected underlines: neighboring drifted words share one continuous
   * underline span (separators between them stay inside the span).
   * Returns HTML string, or null if no delta available.
   */
  function highlightReplacementTokens(rawText, deltaJson, lang, invariables) {
    const text = String(rawText || "");
    if (!text.trim() || !deltaJson) return null;

    let delta;
    try { delta = JSON.parse(deltaJson); } catch { return null; }

    const ops = Array.isArray(delta.token_ops) ? delta.token_ops : null;
    if (!ops || ops.length === 0) return null;

    // Collect changed word indices
    const changed = new Set();
    for (const op of ops) {
      const tag = String(op.op || "").toLowerCase();
      if (tag === "equal" || tag === "delete") continue;
      const span = op.cur_span;
      if (!Array.isArray(span) || span.length < 2) continue;
      for (let i = span[0]; i < span[1]; i++) changed.add(i);
    }
    if (changed.size === 0) return null;

    // Build invariable phrase index for combined coloring
    const invRanges = [];
    const hasInv = invariables && invariables.length > 0;
    if (hasInv) {
      for (const inv of invariables) {
        const phrase = inv.phrase || (typeof inv === "string" ? inv : "");
        if (!phrase) continue;
        const cat = inv.category || "sensory";
        const cssClass = cat === "temporal" ? "inv-temporal"
                       : cat === "proper_noun" ? "inv-proper"
                       : "inv-sensory";
        try {
          const regex = new RegExp(escapeRegex(phrase), "gi");
          let match;
          while ((match = regex.exec(text)) !== null) {
            invRanges.push({ start: match.index, end: match.index + match[0].length, cssClass });
          }
        } catch (_) {}
      }
      invRanges.sort((a, b) => a.start - b.start || b.end - a.end);
    }

    // Tokenize using same regex as backend
    const re = (lang === "ar") ? RE_U_TOK : RE_EN_TOK;
    re.lastIndex = 0;

    // Build token list with char offsets
    const tokens = [];
    let m;
    while ((m = re.exec(text)) !== null) {
      const word = m[1];
      tokens.push({
        text: m[0],
        isWord: (word != null),
        start: m.index,
        end: m.index + m[0].length,
      });
    }

    // Helper: check if this token falls inside an invariable range
    function getInvClass(tok) {
      if (!hasInv) return null;
      for (const r of invRanges) {
        if (r.start > tok.end) break;
        if (r.end > tok.start && r.start < tok.end) return r.cssClass;
      }
      return null;
    }

    // Helper: find the next word token's index and whether it's drifted
    function nextWordIsDrifted(tokenIdx) {
      for (let j = tokenIdx + 1; j < tokens.length; j++) {
        if (tokens[j].isWord) {
          // Count word index for this token
          let wi = 0;
          for (let k = 0; k <= j; k++) {
            if (k === j) return changed.has(wi);
            if (tokens[k].isWord) wi++;
          }
        }
      }
      return false;
    }

    // Pre-compute word index for each word token
    const wordIdxMap = new Map(); // token array index -> word index
    let wi = 0;
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].isWord) {
        wordIdxMap.set(i, wi);
        wi++;
      }
    }

    // Render with connected underlines (peek-ahead algorithm)
    let out = "";
    let inRun = false;

    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i];

      if (tok.isWord) {
        const wordIdx = wordIdxMap.get(i);
        const isDrifted = changed.has(wordIdx);
        const invClass = getInvClass(tok);

        if (isDrifted) {
          if (!inRun) {
            out += `<span class="drift-new-mod">`;
            inRun = true;
          }
          // Render word with optional invariable color inside the drift span
          if (invClass) {
            out += `<span class="${invClass}">${esc(tok.text)}</span>`;
          } else {
            out += esc(tok.text);
          }
        } else {
          // Not drifted — close any open run
          if (inRun) {
            out += `</span>`;
            inRun = false;
          }
          if (invClass) {
            out += `<span class="${invClass}">${esc(tok.text)}</span>`;
          } else {
            out += esc(tok.text);
          }
        }
      } else {
        // Separator token
        if (inRun) {
          // Peek ahead: is the next word also drifted?
          let nextDrifted = false;
          for (let j = i + 1; j < tokens.length; j++) {
            if (tokens[j].isWord) {
              const nwi = wordIdxMap.get(j);
              nextDrifted = changed.has(nwi);
              break;
            }
          }
          if (nextDrifted) {
            // Keep separator inside the underline span (connected)
            out += esc(tok.text);
          } else {
            // Close the span, emit separator outside
            out += `</span>`;
            inRun = false;
            out += esc(tok.text);
          }
        } else {
          out += esc(tok.text);
        }
      }
    }

    if (inRun) out += `</span>`;

    return out;
  }

  // ---- Colorization (word/phrase-level) ----

  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  /**
   * Merge ranges so there are no overlaps. Earlier ranges win.
   */
  function mergeRanges(ranges) {
    const result = [];
    let lastEnd = 0;
    for (const r of ranges) {
      if (r.start < lastEnd) continue;
      result.push(r);
      lastEnd = r.end;
    }
    return result;
  }

  /**
   * Colorize text at word/phrase level:
   * - invariables: color by category (sensory=red, temporal=purple, proper_noun=cyan)
   * - selected segments: yellow (sentence-level, drifted content)
   * Falls back to legacy sensory_anchors if no invariables present.
   * Returns HTML string.
   */
  function colorizeText(fullText, segments, anchors, invariables) {
    const text = (fullText || "").trim();
    if (!text) return esc(text);

    const hasInv = invariables && invariables.length > 0;
    const hasAnch = anchors && anchors.length > 0;
    const hasSeg = segments && segments.length > 0;

    if (!hasInv && !hasAnch && !hasSeg) return esc(text);

    const ranges = [];

    // 1. Invariable phrase highlighting (word/phrase level)
    if (hasInv) {
      for (const inv of invariables) {
        const phrase = inv.phrase || (typeof inv === "string" ? inv : "");
        if (!phrase) continue;
        const cat = inv.category || "sensory";
        const cssClass = cat === "temporal" ? "inv-temporal"
                       : cat === "proper_noun" ? "inv-proper"
                       : "inv-sensory";
        try {
          const regex = new RegExp(escapeRegex(phrase), "gi");
          let match;
          while ((match = regex.exec(text)) !== null) {
            ranges.push({ start: match.index, end: match.index + match[0].length, cssClass });
          }
        } catch (_) { /* skip bad regex */ }
      }
    } else if (hasAnch) {
      // Legacy fallback: sensory anchors as phrases
      for (const a of anchors) {
        const phrase = typeof a === "string" ? a : (a.text || "");
        if (!phrase) continue;
        try {
          const regex = new RegExp(escapeRegex(phrase), "gi");
          let match;
          while ((match = regex.exec(text)) !== null) {
            ranges.push({ start: match.index, end: match.index + match[0].length, cssClass: "inv" });
          }
        } catch (_) {}
      }
    }

    // 2. Drifted segment highlighting (sentence-level, lower priority)
    if (hasSeg) {
      for (const seg of segments) {
        const phrase = typeof seg === "string" ? seg : (seg.text || "");
        if (!phrase) continue;
        const idx = text.toLowerCase().indexOf(phrase.toLowerCase());
        if (idx >= 0) {
          ranges.push({ start: idx, end: idx + phrase.length, cssClass: "drifted" });
        }
      }
    }

    // Sort by start position, then by longer spans first
    ranges.sort((a, b) => a.start - b.start || b.end - a.end);

    // Build HTML with non-overlapping spans
    const merged = mergeRanges(ranges);
    let result = "";
    let pos = 0;
    for (const r of merged) {
      if (r.start > pos) result += esc(text.slice(pos, r.start));
      result += `<span class="${r.cssClass}">${esc(text.slice(r.start, r.end))}</span>`;
      pos = r.end;
    }
    if (pos < text.length) result += esc(text.slice(pos));
    return result;
  }

  // ---- Causality Panel rendering ----

  const causalityPanel     = $("causalityPanel");
  const causalityPersona   = $("causalityPersona");
  const causalityVer       = $("causalityVer");
  const causalityDir       = $("causalityDirection");
  const causalityJust      = $("causalityJustification");
  const causalityFlow      = $("causalityFlow");
  const causalityKeepsake  = $("causalityKeepsake");
  const driftedKeywords    = $("driftedKeywords");

  /**
   * Given the new drift text and a list of keywords, return HTML with keywords
   * wrapped in <span class="kw-hit">. Matches case-insensitively.
   */
  function highlightKeywordsInText(text, keywords) {
    if (!text || !keywords || keywords.length === 0) return esc(text);

    // Build char-level hit map
    const lower = text.toLowerCase();
    const hits = new Uint8Array(text.length); // 0 = no hit, 1 = hit

    // Sort keywords longest-first so longer phrases take precedence
    const sorted = [...keywords].sort((a, b) => b.length - a.length);
    for (const kw of sorted) {
      const kwl = kw.toLowerCase();
      let pos = 0;
      while (true) {
        const idx = lower.indexOf(kwl, pos);
        if (idx < 0) break;
        for (let i = idx; i < idx + kwl.length; i++) hits[i] = 1;
        pos = idx + 1;
      }
    }

    // Render with spans around hit regions
    let out = "";
    let inHit = false;
    for (let i = 0; i < text.length; i++) {
      const isHit = hits[i] === 1;
      if (isHit && !inHit) { out += `<span class="kw-hit">`; inHit = true; }
      if (!isHit && inHit) { out += `</span>`; inHit = false; }
      out += esc(text[i]);
    }
    if (inHit) out += `</span>`;
    return out;
  }

  /**
   * Split text into sentences (same as backend _split_sentences).
   */
  function splitSentences(text) {
    if (!text || !text.trim()) return [];
    return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
  }

  /**
   * Given the prev_drift_en, drift_en and selected_segments, extract the
   * corresponding new text for each segment by sentence-index mapping.
   *
   * Segments are sentences from prev. We find which sentence index they
   * correspond to in prev, then pull the same sentence index from drift.
   */
  function extractSegmentReplacements(prevText, driftText, segments) {
    if (!prevText || !driftText || !segments || segments.length === 0) return [];

    const prevSents = splitSentences(prevText);
    const driftSents = splitSentences(driftText);

    const results = [];
    for (const seg of segments) {
      const oldText = (seg.text || "").trim();
      if (!oldText) continue;

      // Find which sentence index this segment maps to in prev
      let sentIdx = -1;
      for (let i = 0; i < prevSents.length; i++) {
        // Match by containment (segment may be truncated in params_json)
        if (prevSents[i].startsWith(oldText.substring(0, 60)) || oldText.startsWith(prevSents[i].substring(0, 60))) {
          sentIdx = i;
          break;
        }
      }

      if (sentIdx < 0) {
        // Fuzzy: try finding any sentence that shares significant overlap
        for (let i = 0; i < prevSents.length; i++) {
          const words = oldText.toLowerCase().split(/\s+/).slice(0, 6);
          if (words.length >= 3 && prevSents[i].toLowerCase().includes(words.join(" "))) {
            sentIdx = i;
            break;
          }
        }
      }

      if (sentIdx < 0) continue;

      // Pull the corresponding sentence from drift
      const newText = (sentIdx < driftSents.length) ? driftSents[sentIdx] : "";
      if (!newText) continue;

      // Skip if identical (no visible change)
      if (newText === prevSents[sentIdx]) continue;

      results.push({
        segment_id: seg.segment_id || "?",
        old_text: prevSents[sentIdx],
        new_text: newText,
      });
    }
    return results;
  }

  function renderCausalityPanel(stateObjCur, stateObjPrev) {
    if (!causalityPanel) return;

    const ver = stateObjCur.version || curV;
    if (causalityPersona) causalityPersona.textContent = persona;
    if (causalityVer) causalityVer.textContent = ver;

    // Drift direction (Stage 1 lens interpretation)
    if (causalityDir) {
      const dir = (stateObjCur.drift_direction || "").trim();
      if (dir) {
        causalityDir.innerHTML =
          `<div class="dir-label">Lens interpretation</div>` +
          esc(dir);
      } else {
        causalityDir.innerHTML =
          `<div class="dir-label">Lens interpretation</div>` +
          `<span style="opacity:0.4;">not generated (pre-v6)</span>`;
      }
    }

    // Justification
    const just = (lang === "ar")
      ? (stateObjCur.justification_ar || stateObjCur.justification_en || "")
      : (stateObjCur.justification_en || "");
    if (causalityJust) {
      if (just) {
        causalityJust.innerHTML = esc(just);
        causalityJust.style.opacity = "";
      } else {
        causalityJust.innerHTML = `<span style="opacity:0.4;">no justification available</span>`;
      }
    }

    // Segment before → after flow
    if (causalityFlow) {
      const segs = stateObjCur.selected_segments || [];
      const kws = stateObjCur.drifted_keywords || [];
      const prevEn = stateObjCur.prev_drift_en || stateObjPrev.drift_en || "";
      const driftEn = stateObjCur.drift_en || "";

      const replacements = extractSegmentReplacements(prevEn, driftEn, segs);

      if (replacements.length > 0) {
        let html = "";
        for (const rep of replacements) {
          html += `<div class="seg-diff">`;
          html += `<div class="seg-diff-label">segment ${esc(rep.segment_id)}</div>`;
          html += `<div class="seg-diff-old">${esc(rep.old_text)}</div>`;
          html += `<div class="seg-diff-arrow">↓ drifted into</div>`;
          html += `<div class="seg-diff-new">${highlightKeywordsInText(rep.new_text, kws)}</div>`;
          html += `</div>`;
        }
        causalityFlow.innerHTML = html;
      } else if (segs.length > 0) {
        // Show segments without replacement extraction
        let html = "";
        for (const seg of segs) {
          html += `<div class="seg-diff">`;
          html += `<div class="seg-diff-label">segment ${esc(seg.segment_id || "?")}</div>`;
          html += `<div class="seg-diff-old">${esc((seg.text || "").substring(0, 200))}</div>`;
          html += `</div>`;
        }
        causalityFlow.innerHTML = html;
      } else {
        causalityFlow.innerHTML = `<span style="opacity:0.4; font-size:12px;">no segment changes</span>`;
      }
    }

    // Keepsake narration
    if (causalityKeepsake) {
      const keepText = stateObjCur.keepsake_en || "";
      if (keepText) {
        causalityKeepsake.innerHTML =
          `<div class="kp-label">Keepsake narration</div>` +
          esc(keepText);
      } else {
        causalityKeepsake.innerHTML =
          `<div class="kp-label">Keepsake narration</div>` +
          `<span style="opacity:0.4;">not generated yet</span>`;
      }
    }
  }

  // Render drifted keywords in technical metadata
  function renderDriftedKeywords(stateObj) {
    if (driftedKeywords) {
      const kws = stateObj.drifted_keywords || [];
      if (kws.length > 0) {
        driftedKeywords.innerHTML = kws.map(kw =>
          `<span class="tag drifted-kw">${esc(String(kw))}</span>`
        ).join("");
      } else {
        driftedKeywords.innerHTML = `<span style="opacity:0.4;">none</span>`;
      }
    }
  }

  // ---- Image prompt rendering ----

  function renderImagePrompt(stateObj) {
    if (!imagePromptPanel) return;

    const ver = stateObj.version || curV;
    if (imagePromptPersona) imagePromptPersona.textContent = persona;
    if (imagePromptVer) imagePromptVer.textContent = ver;

    // Show thumbnail if image_url exists
    const imgUrl = stateObj.image_url || "";
    if (imagePromptThumb) {
      if (imgUrl) {
        imagePromptThumb.src = imgUrl;
        imagePromptThumb.style.display = "block";
      } else {
        imagePromptThumb.style.display = "none";
      }
    }

    // Show prompt text
    const prompt = (stateObj.image_prompt || "").trim();
    if (imagePromptText) {
      if (prompt) {
        imagePromptText.textContent = prompt;
        imagePromptText.style.opacity = "";
      } else {
        imagePromptText.innerHTML = `<span class="image-prompt-none">no image prompt available for this drift</span>`;
      }
    }
  }

  // ---- Trigger info rendering ----

  function renderTriggerInfo(stateObj) {
    const ver = stateObj.version || curV;
    triggerVer.textContent = fmtVer(ver);

    // RSS Events
    const events = stateObj.event_titles || [];
    const eventIds = stateObj.event_ids || [];
    if (events.length > 0) {
      triggerEvents.innerHTML = events.map(e => {
        const title = esc(e.title || `Event #${e.event_id}`);
        if (e.url) {
          return `<span class="tag event"><a href="${esc(e.url)}" target="_blank">${title}</a></span>`;
        }
        return `<span class="tag event">${title}</span>`;
      }).join("");
    } else if (eventIds.length > 0) {
      triggerEvents.innerHTML = eventIds.map(id =>
        `<span class="tag event">event #${esc(String(id))}</span>`
      ).join("");
    } else {
      triggerEvents.innerHTML = `<span style="opacity:0.4;">none</span>`;
    }

    // Selected Segments
    const segs = stateObj.selected_segments || [];
    if (segs.length > 0) {
      triggerSegments.innerHTML = segs.map(s => {
        const id = esc(s.segment_id || "?");
        const txt = esc((s.text || "").substring(0, 80));
        return `<span class="tag segment" title="${esc(s.text || "")}">[${id}] ${txt}${(s.text || "").length > 80 ? "..." : ""}</span>`;
      }).join("");
    } else {
      triggerSegments.innerHTML = `<span style="opacity:0.4;">none</span>`;
    }

    // Invariables (categorized) or legacy sensory anchors
    const invs = stateObj.invariables || [];
    if (invs.length > 0) {
      triggerAnchors.innerHTML = invs.map(inv => {
        const phrase = esc(inv.phrase || "");
        const cat = inv.category || "sensory";
        const catClass = cat === "temporal" ? "temporal"
                       : cat === "proper_noun" ? "proper"
                       : "anchor";
        return `<span class="tag ${catClass}" title="${esc(cat)}">${phrase}</span>`;
      }).join("");
    } else {
      const anch = stateObj.sensory_anchors || [];
      if (anch.length > 0) {
        triggerAnchors.innerHTML = anch.map(a => {
          const txt = esc((a || "").substring(0, 80));
          return `<span class="tag anchor" title="${esc(a || "")}">${txt}${(a || "").length > 80 ? "..." : ""}</span>`;
        }).join("");
      } else {
        triggerAnchors.innerHTML = `<span style="opacity:0.4;">none</span>`;
      }
    }
  }

  // ---- Main render ----

  async function render() {
    updateControls();

    const leftV = Math.max(minV, curV - 1);
    const rightV = curV;

    const [a, b] = await Promise.all([fetchAt(leftV), fetchAt(rightV)]);
    const leftPick = pick(a);
    const rightPick = pick(b);

    const L = swapped ? rightPick : leftPick;
    const R = swapped ? leftPick : rightPick;

    leftMeta.textContent  = swapped ? `Right (${fmtVer(rightV)})` : `Left (${fmtVer(leftV)})`;
    rightMeta.textContent = swapped ? `Left (${fmtVer(leftV)})` : `Right (${fmtVer(rightV)})`;

    leftStamp.textContent  = L.stamp || "";
    rightStamp.textContent = R.stamp || "";

    leftMode.textContent  = L.label;
    rightMode.textContent = R.label;

    const leftOut  = (lang === "ar") ? (L.ar || "") : (L.en || "");
    const rightOut = showDelta
      ? ((lang === "ar") ? (R.delta_ar || "") : (R.delta_en || ""))
      : ((lang === "ar") ? (R.ar || "") : (R.en || ""));

    // Colorize: use the RIGHT (current version) metadata for both panels
    // Left panel uses the same anchors (invariant across versions)
    // Right panel highlights drifted segments specifically
    const curMeta = swapped ? leftPick : rightPick;
    const prevMeta = swapped ? rightPick : leftPick;

    if (showDelta) {
      // Delta view: plain text (JSON), no colorization
      leftText.textContent = leftOut;
      rightText.textContent = rightOut;
    } else {
      // Left panel: colorize invariables + selected segments (what was picked for drift)
      leftText.innerHTML = colorizeText(
        leftOut,
        prevMeta.selected_segments,
        prevMeta.sensory_anchors,
        prevMeta.invariables
      );

      // Right panel: semantic delta underlines on replacement regions
      // Combined with invariable coloring. Falls back to colorizeText if no delta.
      const deltaKey = (lang === "ar") ? R.delta_ar : R.delta_en;
      const deltaHtml = highlightReplacementTokens(
        rightOut, deltaKey, lang, curMeta.invariables
      );
      if (deltaHtml) {
        rightText.innerHTML = deltaHtml;
      } else {
        rightText.innerHTML = colorizeText(
          rightOut,
          curMeta.selected_segments,
          curMeta.sensory_anchors,
          curMeta.invariables
        );
      }
    }

    setRTL(leftText, lang === "ar");
    setRTL(rightText, lang === "ar");

    // Render trigger info from the CURRENT version (b = rightV)
    renderTriggerInfo(b);
    renderDriftedKeywords(b);
    renderCausalityPanel(b, a);
    renderImagePrompt(b);
  }

  function setMode(next) {
    mode = next;
    showDelta = false;
    render().catch(showError);
  }

  function step(dir) {
    const next = curV + dir;
    if (next < minV || next > maxV) return;
    curV = next;
    showDelta = false;
    render().catch(showError);
  }

  function showError(err) {
    console.error(err);
    leftText.textContent = `Error: ${err.message || err}`;
    rightText.textContent = `Error: ${err.message || err}`;
  }

  function bind() {
    driftPrev.addEventListener("click", () => step(-1));
    driftNext.addEventListener("click", () => step(+1));
    keepPrev.addEventListener("click", () => step(-1));
    keepNext.addEventListener("click", () => step(+1));

    driftLabel.addEventListener("click", () => setMode("DRIFT"));
    keepLabel.addEventListener("click", () => setMode("KEEPSAKE"));

    swapSides.addEventListener("click", () => {
      swapped = !swapped;
      render().catch(showError);
    });

    toggleDelta.addEventListener("click", () => {
      showDelta = !showDelta;
      render().catch(showError);
    });

    toggleLang.addEventListener("click", () => {
      lang = (lang === "en") ? "ar" : "en";
      localStorage.setItem("keepsake_lang", lang);  // Save to localStorage
      showDelta = false;
      render().catch(showError);
    });

    personaSelect.addEventListener("change", async () => {
      persona = personaSelect.value;
      const u = new URL(location.href);
      u.searchParams.set("persona", persona);
      history.replaceState({}, "", u.toString());
      await boot();
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") step(-1);
      else if (e.key === "ArrowRight") step(+1);
      else if (e.key === "d" || e.key === "D") setMode("DRIFT");
      else if (e.key === "k" || e.key === "K") setMode("KEEPSAKE");
      else if (e.key === "l" || e.key === "L") {
        lang = (lang === "en") ? "ar" : "en";
        localStorage.setItem("keepsake_lang", lang);  // Save to localStorage
        showDelta = false;
        render().catch(showError);
      }
    });
  }

  async function boot() {
    initPersonaSelect();
    await loadVersions();
    updateControls();
    await render();
  }

  bind();
  boot().catch(showError);
})();
