const CACHE_NAME = 'puyo-sim-v4';
const urlsToCache = [
    './',
    './index.html',
    './style.css',
    './online.css',
    './online.js',
    './puyoSim.js',
    './puyoAI.js',
    './manifest.json',
    './apple-touch-icon.png',
    './android-icon-192x192.png',
    './android-icon-512x512.png'
];

// インストール時にキャッシュを作成
self.addEventListener('install', function(event) {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(function(cache) {
                console.log('Opened cache');
                return cache.addAll(urlsToCache);
            })
            .catch(function(error) {
                console.log('Cache addAll failed:', error);
            })
    );
    self.skipWaiting();
});

// アクティベート時に古いキャッシュを削除
self.addEventListener('activate', function(event) {
    event.waitUntil(
        caches.keys().then(function(cacheNames) {
            return Promise.all(
                cacheNames.map(function(cacheName) {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    self.clients.claim();
});

// フェッチイベントのハンドリング（キャッシュ優先戦略）
self.addEventListener('fetch', function(event) {
    // GETリクエストのみをキャッシュ
    if (event.request.method !== 'GET') {
        return;
    }

    event.respondWith(
        caches.match(event.request)
            .then(function(response) {
                // キャッシュにあればそれを返す
                if (response) {
                    return response;
                }

                // キャッシュにない場合はネットワークからフェッチ
                return fetch(event.request).then(function(response) {
                    // ネットワークエラーの場合
                    if (!response || response.status !== 200 || response.type !== 'basic') {
                        return response;
                    }

                    // 成功時はキャッシュに追加
                    const responseToCache = response.clone();
                    caches.open(CACHE_NAME)
                        .then(function(cache) {
                            cache.put(event.request, responseToCache);
                        });

                    return response;
                }).catch(function(error) {
                    console.log('Fetch failed:', error);
                    // オフラインの場合のフォールバック
                    return new Response('Network error happened', {
                        status: 408,
                        headers: new Headers({ 'Content-Type': 'text/plain' })
                    });
                });
            })
    );
});
