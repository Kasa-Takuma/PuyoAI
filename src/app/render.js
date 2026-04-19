import { summarizeBestAction } from "../ai/dataset.js";
import { SEARCH_PROFILES } from "../ai/search-profiles.js";
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLOR_LABELS,
  COLORS,
  EVENT_TYPES,
  TOP_OUT_COLUMN,
  TOP_OUT_ROW,
  VISIBLE_HEIGHT,
} from "../core/constants.js";
import { boardToRows, encodeAction } from "../core/board.js";
import { APP_VERSION } from "./version.js";
import {
  getDisplayedBoard,
  getDisplayedEvent,
  getLegalActions,
  getPreset,
  previewQueue,
} from "./state.js";

const COLOR_CLASS = {
  [COLORS.EMPTY]: "empty",
  [COLORS.RED]: "red",
  [COLORS.GREEN]: "green",
  [COLORS.BLUE]: "blue",
  [COLORS.YELLOW]: "yellow",
};

function pairMarkup(pair) {
  return `
    <div class="pair">
      <span class="mini-puyo ${COLOR_CLASS[pair.child]}"></span>
      <span class="mini-puyo ${COLOR_CLASS[pair.axis]}"></span>
    </div>
  `;
}

function eventLabel(event) {
  if (!event) {
    return "まだリプレイがありません";
  }

  switch (event.type) {
    case EVENT_TYPES.PLACE:
      return "着地";
    case EVENT_TYPES.CLEAR:
      return `${event.chain}連鎖目 消去 (+${event.stepScore})`;
    case EVENT_TYPES.GRAVITY:
      return `${event.chain}連鎖目 落下`;
    case EVENT_TYPES.SETTLE:
      if (event.topout) {
        return "敗北";
      }
      return event.chain > 0 ? `${event.chain}連鎖で終了` : "連鎖なし";
    default:
      return event.type;
  }
}

function actionLabel(action, pair) {
  const base = `${action.orientation} / 列 ${action.column + 1}`;
  if (!pair) {
    return base;
  }
  return `${base} (${COLOR_LABELS[pair.axis]}・${COLOR_LABELS[pair.child]})`;
}

function sourceLabel(source) {
  switch (source) {
    case "learned":
      return "Learned";
    case "ai":
      return "AI";
    case "random":
      return "Random";
    default:
      return "Manual";
  }
}

function aiModeLabel(aiMode) {
  return aiMode === "learned" ? "Learned AI" : "Search AI";
}

function selectedLearnedModelLabel(state) {
  return (
    state.learnedModels.find((model) => model.id === state.selectedLearnedModelId)?.label ??
    "No model"
  );
}

function aiStatusLabel(state) {
  switch (state.aiStatus) {
    case "searching":
      return "探索中";
    case "auto-search":
      return state.aiContinuous
        ? "連続探索中"
        : `自動探索中 (${state.aiAutoRunRemaining + 1} 手目)`;
    case "ready":
      return "分析完了";
    case "auto-ready":
      return state.aiContinuous
        ? "連続実行中"
        : `連続実行中 残り ${state.aiAutoRunRemaining} 手`;
    case "complete":
      return "連続実行完了";
    case "restarting":
      return "1秒後に再スタート";
    case "stopped":
      return "ゲーム終了で停止";
    case "stop-requested":
      return "停止要求中";
    case "error":
      return "AIエラー";
    default:
      return "待機中";
  }
}

function historyMarkup(history) {
  if (history.length === 0) {
    return `<p class="empty-copy">まだ手を進めていません。</p>`;
  }

  return `
    <ul class="history-list">
      ${history
        .map(
          (entry) => `
            <li class="history-item">
              <span class="history-turn">Turn ${entry.turn}</span>
              <span class="history-body">${actionLabel(entry.action, entry.pair)}</span>
              <span class="history-meta">${entry.chains}連鎖 / +${entry.score}${
                entry.topout ? " / TOP OUT" : ""
              } / ${sourceLabel(entry.source)}</span>
            </li>
          `,
        )
        .join("")}
    </ul>
  `;
}

function candidateMarkup(analysis) {
  if (!analysis || analysis.candidates.length === 0) {
    return `<p class="empty-copy">まだ探索結果がありません。</p>`;
  }

  if (analysis.kind === "learned") {
    return `
      <ol class="candidate-list">
        ${analysis.candidates
          .slice(0, 5)
          .map(
            (candidate, index) => `
              <li class="candidate-item ${
                candidate.actionKey === analysis.bestActionKey ? "best" : ""
              }">
                <div class="candidate-rank">#${index + 1}</div>
                <div class="candidate-body">
                  <strong>${actionLabel(candidate.action)}</strong>
                  <span>確率 ${(candidate.probability * 100).toFixed(2)}%</span>
                  <span>logit ${candidate.logit.toFixed(3)}</span>
                </div>
              </li>
            `,
          )
          .join("")}
      </ol>
    `;
  }

  return `
    <ol class="candidate-list">
      ${analysis.candidates
        .slice(0, 5)
        .map(
          (candidate, index) => `
            <li class="candidate-item ${
              candidate.actionKey === analysis.bestActionKey ? "best" : ""
            }">
              <div class="candidate-rank">#${index + 1}</div>
              <div class="candidate-body">
                <strong>${actionLabel(candidate.action)}</strong>
                <span>探索値 ${Math.round(candidate.searchScore)} / 即時 ${
                  candidate.immediateScore
                } / ${candidate.bestDepth} 手読み</span>
                <span>line: ${candidate.line
                  .map((action) => `${action.orientation}:${action.column + 1}`)
                  .join(" -> ")}</span>
              </div>
            </li>
          `,
        )
        .join("")}
    </ol>
  `;
}

function analysisMarkup(state) {
  const analysis = state.aiAnalysis;
  if (!analysis) {
    return `
      <p class="empty-copy">
        「AI Analyze」または「AI Move」で ${aiModeLabel(state.aiMode)} の結果をここに表示します。
      </p>
    `;
  }

  if (analysis.kind === "learned") {
    return `
      <div class="analysis-head">
        <div class="metric-card">
          <span class="metric-label">Objective</span>
          <strong>${analysis.objective}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Best Action</span>
          <strong>${summarizeBestAction(analysis)}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Confidence</span>
          <strong>${(analysis.bestScore * 100).toFixed(2)}%</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Model</span>
          <strong>${analysis.modelName}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Elapsed</span>
          <strong>${analysis.elapsedMs.toFixed(1)} ms</strong>
        </div>
      </div>

      <div class="analysis-note">
        <span>turn ${analysis.snapshot.turn} の盤面に対する learned policy の結果です。</span>
        <span>${analysis.modelLabel ?? analysis.modelName} / input dim: ${analysis.inputDim}</span>
      </div>

      ${candidateMarkup(analysis)}
    `;
  }

  return `
      <div class="analysis-head">
      <div class="metric-card">
        <span class="metric-label">Objective</span>
        <strong>${analysis.objective}</strong>
      </div>
      <div class="metric-card">
        <span class="metric-label">Best Action</span>
        <strong>${summarizeBestAction(analysis)}</strong>
      </div>
      <div class="metric-card">
        <span class="metric-label">Search Score</span>
        <strong>${Math.round(analysis.bestScore)}</strong>
      </div>
      <div class="metric-card">
        <span class="metric-label">Expanded</span>
        <strong>${analysis.expandedNodeCount}</strong>
      </div>
      <div class="metric-card">
        <span class="metric-label">Elapsed</span>
        <strong>${analysis.elapsedMs.toFixed(1)} ms</strong>
      </div>
    </div>

    <div class="analysis-note">
      <span>turn ${analysis.snapshot.turn} の盤面に対する探索結果です。</span>
      <span>dataset sample: ${state.aiDataset.length}</span>
    </div>

    ${candidateMarkup(analysis)}
  `;
}

export function renderApp(root, state) {
  const board = getDisplayedBoard(state);
  const displayedEvent = getDisplayedEvent(state);
  const preset = getPreset(state.presetId);
  const legalActions = getLegalActions(state);
  const actionOptions = legalActions
    .map((action) => {
      const encoded = encodeAction(action);
      const selected = encoded === state.selectedAction ? "selected" : "";
      const suggested = encoded === state.aiAnalysis?.bestActionKey ? "AI" : "";
      return `<option value="${encoded}" ${selected}>${action.orientation} / 列 ${
        action.column + 1
      }${suggested ? ` / ${suggested}` : ""}</option>`;
    })
    .join("");
  const aiBusyDisabled = state.aiBusy ? "disabled" : "";
  const aiSettingsDisabled =
    state.aiBusy || state.aiMode === "learned" ? "disabled" : "";
  const learnedModelDisabled =
    state.aiBusy || state.learnedModels.length === 0 ? "disabled" : "";

  root.innerHTML = `
    <div class="shell">
      <div class="version-bar">PuyoAI ${APP_VERSION}</div>
      <header class="hero">
        <div>
          <p class="eyebrow">PPT2-like Solo Simulator</p>
          <h1>PuyoAI Viewer</h1>
          <p class="hero-copy">
            配置ベースのぷよシミュレータ上で探索AIを動かし、その結果を後から学習データとして使えるようにしてあります。
          </p>
          <div class="hero-actions">
            <a class="link-chip" href="./batch.html">Batch Runner</a>
          </div>
        </div>
        <div class="status-strip">
          <div class="stat">
            <span class="stat-label">Turn</span>
            <strong>${state.turn}</strong>
          </div>
          <div class="stat">
            <span class="stat-label">Total Score</span>
            <strong>${state.totalScore}</strong>
          </div>
          <div class="stat">
            <span class="stat-label">Max Chains</span>
            <strong>${state.maxChains}</strong>
          </div>
        </div>
      </header>

      <main class="layout">
        <section class="panel board-panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Observation</p>
              <h2>Field</h2>
            </div>
            <div class="event-pill ${displayedEvent?.type ?? "idle"}">
              ${eventLabel(displayedEvent)}
            </div>
          </div>

          <div class="board-wrap">
            <div class="board">
              ${Array.from({ length: BOARD_HEIGHT }, (_, indexFromTop) => {
                const y = BOARD_HEIGHT - 1 - indexFromTop;
                const hidden = y >= VISIBLE_HEIGHT ? " hidden-row" : "";
                return `
                  <div class="board-row${hidden}">
                    <span class="row-label">${y + 1}</span>
                    ${Array.from({ length: BOARD_WIDTH }, (_, x) => {
                      const color = board[y][x];
                      const deathCell =
                        x === TOP_OUT_COLUMN && y === TOP_OUT_ROW
                          ? " death-cell"
                          : "";
                      return `
                        <span class="cell ${COLOR_CLASS[color]}${deathCell}">
                          <span class="cell-core"></span>
                        </span>
                      `;
                    }).join("")}
                  </div>
                `;
              }).join("")}
            </div>

            <aside class="side-stack">
              <div class="mini-panel">
                <span class="mini-title">Current</span>
                ${pairMarkup(state.currentPair)}
              </div>
              <div class="mini-panel">
                <span class="mini-title">Next</span>
                ${previewQueue(state)
                  .map((pair) => pairMarkup(pair))
                  .join("")}
              </div>
              <div class="mini-panel board-note">
                <span class="mini-title">Visible</span>
                <p>13段目は消去判定に使わず、14段目に置かれたぷよは消滅します。×は敗北点です。</p>
              </div>
            </aside>
          </div>
        </section>

        <section class="panel ai-panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">${aiModeLabel(state.aiMode)}</p>
              <h2>Planner</h2>
            </div>
            <div class="event-pill ${state.aiBusy ? "clear" : "idle"}">
              ${aiStatusLabel(state)}
            </div>
          </div>

          <div class="control-grid ai-grid">
            <label class="field">
              <span>AI Mode</span>
              <select id="ai-mode" ${aiBusyDisabled}>
                <option value="search" ${state.aiMode === "search" ? "selected" : ""}>Search</option>
                <option value="learned" ${state.aiMode === "learned" ? "selected" : ""}>Learned</option>
              </select>
            </label>

            <label class="field">
              <span>Learned Model</span>
              <select id="learned-model" ${learnedModelDisabled}>
                ${
                  state.learnedModels.length > 0
                    ? state.learnedModels
                        .map(
                          (model) => `
                            <option value="${model.id}" ${
                              model.id === state.selectedLearnedModelId ? "selected" : ""
                            }>${model.label}</option>
                          `,
                        )
                        .join("")
                    : `<option value="">No learned models</option>`
                }
              </select>
            </label>

            <label class="field">
              <span>Depth</span>
              <select id="ai-depth" ${aiSettingsDisabled}>
                ${[1, 2, 3, 4]
                  .map(
                    (depth) => `
                      <option value="${depth}" ${
                        depth === state.aiSettings.depth ? "selected" : ""
                      }>${depth}</option>
                    `,
                  )
                  .join("")}
              </select>
            </label>

            <label class="field">
              <span>Beam Width</span>
              <input id="ai-beam" type="number" min="4" max="96" step="1" value="${
                state.aiSettings.beamWidth
              }" ${aiSettingsDisabled} />
            </label>

            <label class="field wide">
              <span>Search Profile</span>
              <select id="ai-search-profile" ${aiSettingsDisabled}>
                ${SEARCH_PROFILES.map(
                  (profile) => `
                    <option value="${profile.id}" ${
                      profile.id === state.aiSettings.searchProfile ? "selected" : ""
                    }>${profile.label}</option>
                  `,
                ).join("")}
              </select>
            </label>
          </div>

          <div class="button-row">
            <button id="ai-analyze" class="soft" ${
              state.gameOver || state.aiBusy ? "disabled" : ""
            }>AI Analyze</button>
            <button id="ai-move" class="accent" ${
              state.gameOver || state.aiBusy ? "disabled" : ""
            }>AI Move</button>
            <button id="ai-run10" class="soft" ${
              state.gameOver || state.aiBusy ? "disabled" : ""
            }>AI x10</button>
            <button id="ai-run" class="soft" ${
              state.gameOver || state.aiBusy ? "disabled" : ""
            }>AI Run</button>
            <button id="ai-stop" class="soft" ${
              !state.aiContinuous &&
              state.aiAutoRunRemaining <= 0 &&
              state.aiStatus !== "auto-search" &&
              state.aiStatus !== "auto-ready" &&
              state.aiStatus !== "stop-requested"
                ? "disabled"
                : ""
            }>Stop</button>
          </div>

          <div class="button-row">
            <button id="ai-export" class="soft" ${
              state.aiDataset.length === 0 || state.aiBusy ? "disabled" : ""
            }>Export Dataset</button>
            <button id="ai-clear-log" class="soft" ${
              state.aiDataset.length === 0 || state.aiBusy ? "disabled" : ""
            }>Clear Dataset</button>
          </div>

          <div class="ai-note">
            <span>${
              state.aiMode === "learned"
                ? `selected model: ${selectedLearnedModelLabel(state)}`
                : `dataset samples: ${state.aiDataset.length}`
            }</span>
            <span>${
              state.aiLastError
                ? `error: ${state.aiLastError}`
                : state.aiMode === "learned"
                  ? "learned mode は保存済み MLP をそのまま推論に使います。"
                  : "探索結果はそのまま学習用 JSON に出力できます。"
            }</span>
          </div>

          ${analysisMarkup(state)}
        </section>

        <section class="panel control-panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Controls</p>
              <h2>Scenario And Turn</h2>
            </div>
          </div>

          <div class="control-grid">
            <label class="field">
              <span>シナリオ</span>
              <select id="preset-select" ${aiBusyDisabled}>
                ${Object.values({
                  sandbox: getPreset("sandbox"),
                  singleChain: getPreset("singleChain"),
                  doubleChain: getPreset("doubleChain"),
                  topout: getPreset("topout"),
                })
                  .map(
                    (item) => `
                      <option value="${item.id}" ${
                        item.id === state.presetId ? "selected" : ""
                      }>${item.name}</option>
                    `,
                  )
                  .join("")}
              </select>
            </label>

            <label class="field">
              <span>Seed</span>
              <input id="seed-input" value="${state.seedBase}" ${aiBusyDisabled} />
            </label>

            <label class="field wide">
              <span>配置</span>
              <select id="action-select" ${state.gameOver || state.aiBusy ? "disabled" : ""}>
                ${actionOptions}
              </select>
            </label>
          </div>

          <p class="preset-copy">${preset.description}</p>
          <p class="preset-copy">Current game seed: ${state.seed}</p>

          <div class="button-row">
            <button id="reset-button" class="soft" ${aiBusyDisabled}>Reset</button>
            <button id="random-button" class="soft" ${
              state.gameOver || state.aiBusy ? "disabled" : ""
            }>Random Move</button>
            <button id="place-button" class="accent" ${
              state.gameOver || state.aiBusy ? "disabled" : ""
            }>Place Pair</button>
          </div>

          <div class="panel-head compact">
            <div>
              <p class="panel-kicker">Replay</p>
              <h2>Last Turn</h2>
            </div>
          </div>

          <div class="button-row replay-row">
            <button id="replay-prev" class="soft">Prev</button>
            <button id="replay-play" class="soft">Play</button>
            <button id="replay-next" class="soft">Next</button>
            <button id="replay-latest" class="soft">Latest</button>
          </div>

          <div class="replay-meta">
            <span>現在イベント: ${eventLabel(displayedEvent)}</span>
            <span>盤面行数: ${boardToRows(board).length}</span>
          </div>
        </section>

        <section class="panel detail-panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">State</p>
              <h2>Result Snapshot</h2>
            </div>
          </div>

          <div class="detail-grid">
            <div class="metric-card">
              <span class="metric-label">Last Turn Score</span>
              <strong>${state.lastResult?.totalScore ?? 0}</strong>
            </div>
            <div class="metric-card">
              <span class="metric-label">Last Turn Chains</span>
              <strong>${state.lastResult?.totalChains ?? 0}</strong>
            </div>
            <div class="metric-card">
              <span class="metric-label">All Clear</span>
              <strong>${state.lastResult?.allClear ? "Yes" : "No"}</strong>
            </div>
            <div class="metric-card">
              <span class="metric-label">Status</span>
              <strong>${state.gameOver ? "Topout" : "Alive"}</strong>
            </div>
          </div>

          <div class="rows-box">
            <h3>Displayed Board Rows</h3>
            <pre>${boardToRows(board).join("\n")}</pre>
          </div>
        </section>

        <section class="panel history-panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Log</p>
              <h2>Turn History</h2>
            </div>
          </div>
          ${historyMarkup(state.history)}
        </section>
      </main>
    </div>
  `;
}
