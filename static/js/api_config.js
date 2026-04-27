// API Configuration for Cloudflare Pages deployment
// This file sets the backend API URL based on environment

(function () {
  const h = window.location.hostname;
  const isPagesDev = h.includes('pages.dev');
  const isLocal = h === 'localhost' || h === '127.0.0.1' || h.startsWith('192.168.') || h.startsWith('10.');

  // Initial config
  window.KD_API_CONFIG = {
    API_BASE_URL: isLocal ? 'http://localhost:8000' : null
  };

  if (isPagesDev) {
    // Cloudflare Pages preview: fetch tunnel URL from fallback
    const FALLBACK_TUNNEL = 'https://crew-acceptable-fish-exciting.trycloudflare.com';
    fetch(`${FALLBACK_TUNNEL}/tunnel_url`)
      .then(res => res.json())
      .then(data => {
        window.KD_API_CONFIG.API_BASE_URL = data.tunnel_url || FALLBACK_TUNNEL;
        console.log('[Keepsake Drift] Updated tunnel URL:', window.KD_API_CONFIG.API_BASE_URL);
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      })
      .catch(() => {
        window.KD_API_CONFIG.API_BASE_URL = FALLBACK_TUNNEL;
        console.warn('[Keepsake Drift] Could not fetch tunnel URL, using fallback:', FALLBACK_TUNNEL);
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      });

  } else if (!isLocal) {
    // Production domain (keepsake-drift.net subdomains etc.)
    // Try to get tunnel URL from same-origin server; fall back to relative URLs ('')
    fetch('/tunnel_url')
      .then(res => res.json())
      .then(data => {
        window.KD_API_CONFIG.API_BASE_URL = data.tunnel_url || '';
        console.log('[Keepsake Drift] Production API base:', window.KD_API_CONFIG.API_BASE_URL);
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      })
      .catch(() => {
        window.KD_API_CONFIG.API_BASE_URL = '';
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      });

  } else {
    // Local: config already set, fire event immediately
    window.dispatchEvent(new CustomEvent('kd-config-ready'));
  }
})();
