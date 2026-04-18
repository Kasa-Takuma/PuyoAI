function runnerStatusLabel(state) {
  if (state.stopRequested) {
    return "停止要求中";
  }
  if (state.running) {
    return "実行中";
  }
  return "待機中";
}

function workerStatusLabel(worker) {
  switch (worker.status) {
    case "running":
      return "実行中";
    case "game-over":
      return "ゲーム終了";
    case "stop-requested":
      return "停止要求中";
    case "stopped":
      return "停止中";
    case "error":
      return "エラー";
    default:
      return "待機中";
  }
}

function summaryFromState(state) {
  const totalTurns = state.workers.reduce(
    (sum, worker) => sum + worker.totalTurns,
    0,
  );
  const totalCompletedGames = state.workers.reduce(
    (sum, worker) => sum + worker.completedGames,
    0,
  );
  const totalScore = state.workers.reduce((sum, worker) => sum + worker.score, 0);
  const bestChain = state.workers.reduce(
    (best, worker) => Math.max(best, worker.overallBestChain, worker.maxChains),
    0,
  );
  const activeWorkers = state.workers.filter((worker) =>
    ["running", "game-over", "stop-requested"].includes(worker.status),
  ).length;

  return {
    totalTurns,
    totalCompletedGames,
    totalScore,
    bestChain,
    activeWorkers,
  };
}

function workerCardMarkup(worker) {
  return `
    <article class="worker-card">
      <div class="worker-head">
        <div>
          <p class="panel-kicker">Worker ${worker.id}</p>
          <h3>Slot ${worker.id}</h3>
        </div>
        <span class="status-chip ${worker.status}">${workerStatusLabel(worker)}</span>
      </div>

      <div class="worker-metrics">
        <div class="mini-panel">
          <span class="metric-label">Current Turn</span>
          <strong>${worker.turn}</strong>
        </div>
        <div class="mini-panel">
          <span class="metric-label">Score</span>
          <strong>${worker.score}</strong>
        </div>
        <div class="mini-panel">
          <span class="metric-label">Max Chains</span>
          <strong>${worker.maxChains}</strong>
        </div>
        <div class="mini-panel">
          <span class="metric-label">Worker Total Turns</span>
          <strong>${worker.totalTurns}</strong>
        </div>
      </div>

      <div class="worker-note">
        <span>completed games: ${worker.completedGames}</span>
        <span>overall best: ${worker.overallBestChain}</span>
      </div>

      <div class="rows-box">
        <span class="metric-label">Current Seed</span>
        <pre>${worker.currentSeed || "まだ開始していません"}</pre>
      </div>

      <div class="worker-note">
        <span>last search: ${worker.lastSearchMs.toFixed(1)} ms</span>
        <span>${worker.error ? `error: ${worker.error}` : "headless simulation"}</span>
      </div>
    </article>
  `;
}

export function renderBatchApp(root, state) {
  const summary = summaryFromState(state);
  const controlsDisabled = state.running ? "disabled" : "";

  root.innerHTML = `
    <div class="shell">
      <header class="hero">
        <div>
          <p class="eyebrow">Headless Parallel Runner</p>
          <h1>PuyoAI Batch Runner</h1>
          <p class="hero-copy">
            フィールド描画を外し、複数 worker で探索AIをまとめて回すモードです。各 worker の進捗と全体の合計ターン数を追えます。
          </p>
          <div class="hero-actions">
            <a class="link-chip" href="./index.html">Viewer に戻る</a>
          </div>
        </div>
        <div class="status-strip">
          <div class="stat">
            <span class="stat-label">Status</span>
            <strong>${runnerStatusLabel(state)}</strong>
          </div>
          <div class="stat">
            <span class="stat-label">Parallel</span>
            <strong>${state.parallelCount}</strong>
          </div>
          <div class="stat">
            <span class="stat-label">Active Workers</span>
            <strong>${summary.activeWorkers}</strong>
          </div>
        </div>
      </header>

      <main class="batch-layout">
        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Controls</p>
              <h2>Batch Config</h2>
            </div>
          </div>

          <div class="control-grid">
            <label class="field">
              Parallel Count
              <input id="parallel-count" type="number" min="1" max="16" value="${
                state.parallelCount
              }" ${controlsDisabled} />
            </label>
            <label class="field">
              Seed Base
              <input id="batch-seed" type="text" value="${state.seedBase}" ${controlsDisabled} />
            </label>
            <label class="field">
              Depth
              <input id="batch-depth" type="number" min="1" max="4" value="${
                state.aiSettings.depth
              }" ${controlsDisabled} />
            </label>
            <label class="field">
              Beam Width
              <input id="batch-beam" type="number" min="4" max="96" value="${
                state.aiSettings.beamWidth
              }" ${controlsDisabled} />
            </label>
          </div>

          <div class="button-row">
            <button id="start-all" class="accent" ${
              state.running ? "disabled" : ""
            }>Start All</button>
            <button id="stop-all" class="soft" ${
              state.running ? "" : "disabled"
            }>Stop All</button>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Summary</p>
              <h2>Session Totals</h2>
            </div>
          </div>

          <div class="summary-grid">
            <div class="metric-card">
              <span class="metric-label">Total Turns</span>
              <strong>${summary.totalTurns}</strong>
            </div>
            <div class="metric-card">
              <span class="metric-label">Completed Games</span>
              <strong>${summary.totalCompletedGames}</strong>
            </div>
            <div class="metric-card">
              <span class="metric-label">Current Total Score</span>
              <strong>${summary.totalScore}</strong>
            </div>
            <div class="metric-card">
              <span class="metric-label">Best Chain Seen</span>
              <strong>${summary.bestChain}</strong>
            </div>
          </div>
        </section>

        <section class="panel worker-panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Workers</p>
              <h2>Parallel Slots</h2>
            </div>
          </div>

          <div class="worker-grid">
            ${state.workers.map((worker) => workerCardMarkup(worker)).join("")}
          </div>
        </section>
      </main>
    </div>
  `;
}
