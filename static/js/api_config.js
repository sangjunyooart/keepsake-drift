// API Configuration
//
// Local dev (localhost / LAN):  use localhost:8000 directly
// pages.dev (Cloudflare preview): fetch live tunnel URL from fallback worker
// Production custom domains (*.keepsake-drift.net): use relative URLs (FastAPI on same origin)

(function () {
  const h = window.location.hostname;
  const isLocal = h === 'localhost' || h === '127.0.0.1'
    || h.startsWith('192.168.') || h.startsWith('10.');
  const isPagesDev = h.includes('pages.dev');

  const FALLBACK_TUNNEL = 'https://crew-acceptable-fish-exciting.trycloudflare.com';

  window.KD_API_CONFIG = { API_BASE_URL: null };

  function _ready(url) {
    window.KD_API_CONFIG.API_BASE_URL = url;
    console.log('[Keepsake Drift] API base:', url || '(relative)');
    window.dispatchEvent(new CustomEvent('kd-config-ready'));
  }

  if (isLocal) {
    _ready('http://localhost:8000');

  } else if (isPagesDev) {
    fetch(`${FALLBACK_TUNNEL}/tunnel_url`)
      .then(r => r.json())
      .then(d => _ready(d.tunnel_url || FALLBACK_TUNNEL))
      .catch(() => _ready(FALLBACK_TUNNEL));

  } else {
    // Production custom domain (*.keepsake-drift.net):
    // FastAPI is served directly on this origin — no tunnel lookup needed.
    _ready('');
  }
})();
