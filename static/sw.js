const CACHE = 'saturi-v1';
const PRECACHE = ['/', '/static/manifest.json', '/static/icon-192.png', '/static/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Only cache GET requests for static assets and the main page
  if (e.request.method !== 'GET') return;
  if (url.pathname.startsWith('/ocr/') || url.pathname.startsWith('/ai/')) return;

  e.respondWith(
    fetch(e.request).then(r => {
      if (r.ok && (url.pathname === '/' || url.pathname.startsWith('/static/'))) {
        const clone = r.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }
      return r;
    }).catch(() => caches.match(e.request))
  );
});
