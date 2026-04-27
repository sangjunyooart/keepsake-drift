// API Configuration
//
// Local dev (localhost / LAN):  use localhost:8000 directly
// pages.dev (Cloudflare preview): fetch live tunnel URL from fallback worker
// Production custom domains (*.keepsake-drift.net):
//   fetch live tunnel URL from en.keepsake-drift.net (primary server),
//   fall back to hardcoded worker URL, then to relative URLs.

(function () {
  const h = window.location.hostname;
  const isLocal = h === 'localhost' || h === '127.0.0.1'
    || h.startsWith('192.168.') || h.startsWith('10.');
  const isPagesDev = h.includes('pages.dev');

  // Cloudflare tunnel worker URL — used as bootstrap to fetch the live API tunnel URL.
  // Update this constant when the Cloudflare tunnel worker URL changes.
  const FALLBACK_TUNNEL = 'https://crew-acceptable-fish-exciting.trycloudflare.com';

  // Primary production server — always routes to the FastAPI backend.
  const PRIMARY_SERVER = 'https://en.keepsake-drift.net';

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
    // 1. Try primary server for the live tunnel URL (CORS allowed for *.keepsake-drift.net)
    // 2. Fall back to hardcoded FALLBACK_TUNNEL worker
    // 3. Fall back to relative URLs (works if this subdomain serves FastAPI directly)
    fetch(`${PRIMARY_SERVER}/tunnel_url`)
      .then(r => r.json())
      .then(d => _ready(d.tunnel_url || ''))
      .catch(() =>
        fetch(`${FALLBACK_TUNNEL}/tunnel_url`)
          .then(r => r.json())
          .then(d => _ready(d.tunnel_url || FALLBACK_TUNNEL))
          .catch(() => _ready(''))
      );
  }
})();
