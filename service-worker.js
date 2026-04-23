const CACHE_NAME = "puyoai-shell-v35";

const APP_SHELL = [
  "./",
  "./index.html",
  "./batch.html",
  "./manifest.webmanifest",
  "./src/app/main.js",
  "./src/app/render.js",
  "./src/app/state.js",
  "./src/app/styles.css",
  "./src/app/version.js",
  "./src/batch/main.js",
  "./src/batch/render.js",
  "./src/core/constants.js",
  "./src/core/board.js",
  "./src/core/engine.js",
  "./src/core/presets.js",
  "./src/core/randomizer.js",
  "./src/ai/features.js",
  "./src/ai/action-vocab.js",
  "./src/ai/learned.js",
  "./src/ai/value.js",
  "./src/ai/search-profiles.js",
  "./src/ai/search.js",
  "./src/ai/dataset.js",
  "./src/ppsim2/adapter.js",
  "./src/ppsim2/v12-controller.js",
  "./models/manifest.json",
  "./models/policy_mlp.web.json",
  "./models/value_mlp.web.json",
  "./src/worker/ai-worker.js",
  "./src/worker/batch-worker.js",
  "./ppsim2/",
  "./ppsim2/index.html",
  "./ppsim2/style.css",
  "./ppsim2/online.css",
  "./ppsim2/puyoSim.js",
  "./ppsim2/online.js",
  "./ppsim2/manifest.json",
  "./ppsim2/apple-touch-icon.png",
  "./ppsim2/android-icon-192x192.png",
  "./ppsim2/android-icon-512x512.png",
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
