// API Configuration for Cloudflare Pages deployment
// This file sets the backend API URL based on environment

// Initial config - will be updated dynamically on Cloudflare Pages
window.KD_API_CONFIG = {
  API_BASE_URL: window.location.hostname.includes('pages.dev')
    ? null  // Will be fetched dynamically
    : 'http://localhost:8000'
};

// On Cloudflare Pages, fetch the current tunnel URL dynamically
if (window.location.hostname.includes('pages.dev')) {
  // Use a hardcoded fallback URL for the initial fetch
  const FALLBACK_TUNNEL = 'https://crew-acceptable-fish-exciting.trycloudflare.com';

  fetch(`${FALLBACK_TUNNEL}/tunnel_url`)
    .then(res => res.json())
    .then(data => {
      if (data.tunnel_url) {
        window.KD_API_CONFIG.API_BASE_URL = data.tunnel_url;
        console.log('[Keepsake Drift] Updated tunnel URL:', data.tunnel_url);

        // Trigger a custom event so other scripts know the config is ready
        window.dispatchEvent(new CustomEvent('kd-config-ready'));
      }
    })
    .catch(err => {
      console.warn('[Keepsake Drift] Could not fetch tunnel URL, using fallback:', FALLBACK_TUNNEL);
      window.KD_API_CONFIG.API_BASE_URL = FALLBACK_TUNNEL;
      window.dispatchEvent(new CustomEvent('kd-config-ready'));
    });
}
