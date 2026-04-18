const CACHE_NAME = "puyoai-shell-v2";

const APP_SHELL = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./src/app/main.js",
  "./src/app/render.js",
  "./src/app/state.js",
  "./src/app/styles.css",
  "./src/core/constants.js",
  "./src/core/board.js",
  "./src/core/engine.js",
  "./src/core/presets.js",
  "./src/core/randomizer.js",
  "./src/ai/features.js",
  "./src/ai/search.js",
  "./src/ai/dataset.js",
  "./src/worker/ai-worker.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)),
      ),
    ),
  );
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (!response || response.status !== 200 || response.type !== "basic") {
          return response;
        }

        const responseClone = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseClone);
        });
        return response;
      })
      .catch(() => caches.match(event.request)),
  );
});
