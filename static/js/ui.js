// static/js/ui.js
// Safe helper for legacy pages.
// v1.65 standard: use kd_lang only. Do not write to "lang".

(function initializeLangState() {
  const STORAGE_LANG_KEY = "kd_lang";
  const current = localStorage.getItem(STORAGE_LANG_KEY) || "en";
  window.currentLang = current;
})();

function toggleMenu() {
  const overlay = document.getElementById("menuOverlay");
  if (!overlay) return;
  overlay.classList.toggle("visible");
}

document.addEventListener("click", (e) => {
  const overlay = document.getElementById("menuOverlay");
  if (!overlay) return;
  if (e.target === overlay) overlay.classList.remove("visible");
});

function toggleLang() {
  const STORAGE_LANG_KEY = "kd_lang";
  const current = localStorage.getItem(STORAGE_LANG_KEY) || "en";
  const next = current === "en" ? "ar" : "en";

  localStorage.setItem(STORAGE_LANG_KEY, next);
  window.currentLang = next;

  if (typeof applyPersonaLang === "function") applyPersonaLang();
  if (typeof applyLanguageToChatUI === "function") applyLanguageToChatUI();
  if (typeof applyPrivacyLang === "function") applyPrivacyLang();
  if (typeof applyIndexLang === "function") applyIndexLang();

  const lbl = document.getElementById("langLabel");
  if (lbl) lbl.textContent = next === "en" ? "عرب" : "EN";
}