// API Configuration
// Local dev: uses localhost:8000 directly.
// All remote deployments (pages.dev, *.keepsake-drift.net): fetch live tunnel URL
// from the fallback Cloudflare tunnel, then switch to the current live URL.

(function () {
  const h = window.location.hostname;
  const isLocal = h === 'localhost' || h === '127.0.0.1' || h.startsWith('192.168.') || h.startsWith('10.');

  // Hardcoded fallback tunnel — used to bootstrap the live tunnel URL lookup.
  // Update this when the Cloudflare tunnel worker URL changes.
  const FALLBACK_TUNNEL = 'https://crew-acceptable-fish-exciting.trycloudflare.com';

  window.KD_API_CONFIG = {
    API_BASE_URL: isLocal ? 'http://localhost:8000' : null
  };

  if (isLocal) {
    // Local dev: config is already set, fire event immediately
    window.dispatchEvent(new CustomEvent('kd-config-ready'));
  } else {
    // Remote (pages.dev or *.keepsake-drift.net or any other host):
    // Fetch the live tunnel URL from the fallback worker, then update config.
    fetch(`${FALLBACK_TUNNEL}/tunnel_url`)
      .then(res => res.json())
      .then(data => {
        window.KD_API_CONFIG.API_BASE_URL = data.tunnel_url || FALLBACK_TUNNEL;
        console.log('[Keepsake Drift] API base:', window.KD_API_CONFIG.API_BASE_URL);
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      })
      .catch(() => {
        window.KD_API_CONFIG.API_BASE_URL = FALLBACK_TUNNEL;
        console.warn('[Keepsake Drift] Tunnel URL fetch failed, using fallback:', FALLBACK_TUNNEL);
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      });
  }
})();
