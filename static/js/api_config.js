// API Configuration
//
// Local dev (localhost / LAN):  use localhost:8000 directly
// pages.dev (Cloudflare preview): fetch live tunnel URL from fallback worker
// Production custom domains (*.keepsake-drift.net and similar):
//   use relative URLs — all subdomains route through Cloudflare to the same
//   FastAPI backend, so /state, /lang_config etc. resolve on the same server.

(function () {
  const h = window.location.hostname;
  const isLocal = h === 'localhost' || h === '127.0.0.1'
    || h.startsWith('192.168.') || h.startsWith('10.');
  const isPagesDev = h.includes('pages.dev');

  if (isLocal) {
    window.KD_API_CONFIG = { API_BASE_URL: 'http://localhost:8000' };
    window.dispatchEvent(new CustomEvent('kd-config-ready'));

  } else if (isPagesDev) {
    // Cloudflare Pages preview — backend is behind a dynamic tunnel
    window.KD_API_CONFIG = { API_BASE_URL: null };
    const FALLBACK_TUNNEL = 'https://crew-acceptable-fish-exciting.trycloudflare.com';
    fetch(`${FALLBACK_TUNNEL}/tunnel_url`)
      .then(res => res.json())
      .then(data => {
        window.KD_API_CONFIG.API_BASE_URL = data.tunnel_url || FALLBACK_TUNNEL;
        console.log('[Keepsake Drift] Tunnel URL:', window.KD_API_CONFIG.API_BASE_URL);
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      })
      .catch(() => {
        window.KD_API_CONFIG.API_BASE_URL = FALLBACK_TUNNEL;
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      });

  } else {
    // Production custom domain (en/ar/gr/br.keepsake-drift.net etc.)
    // Relative URLs — same-origin requests go directly to the FastAPI server.
    window.KD_API_CONFIG = { API_BASE_URL: '' };
    window.dispatchEvent(new CustomEvent('kd-config-ready'));
  }
})();
