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
    // Each subdomain runs its own FastAPI — try relative URLs first (fastest path).
    // Only fall back to tunnel URL lookup if the local API is unreachable.
    fetch('/state?persona=human', { method: 'HEAD', signal: AbortSignal.timeout(2000) })
      .then(() => _ready(''))  // local API works — use relative URLs, zero extra latency
      .catch(() =>
        // Local API unreachable — fall back to tunnel URL lookup
        fetch(`${PRIMARY_SERVER}/tunnel_url`, { signal: AbortSignal.timeout(4000) })
          .then(r => r.json())
          .then(d => _ready(d.tunnel_url || ''))
          .catch(() =>
            fetch(`${FALLBACK_TUNNEL}/tunnel_url`, { signal: AbortSignal.timeout(4000) })
              .then(r => r.json())
              .then(d => _ready(d.tunnel_url || FALLBACK_TUNNEL))
              .catch(() => _ready(''))
          )
      );
  }
})();
