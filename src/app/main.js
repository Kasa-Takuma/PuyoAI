import { createDatasetFilename, serializeAiDataset } from "../ai/dataset.js";
import { searchBestMove } from "../ai/search.js";
import { renderApp } from "./render.js";
import {
  applyAction,
  applyRandomAction,
  applySelectedAction,
  canReplay,
  clearAiDataset,
  clearAiError,
  createAiRequestPayload,
  createAiTrainingSnapshot,
  createGameState,
  recordAiAnalysis,
  resetReplayToLatest,
  setAiError,
  setAiSetting,
  setAiStatus,
  setSelectedAction,
  startReplay,
  stepReplay,
  stopReplay,
} from "./state.js";

const root = document.querySelector("#app");
let state = createGameState();
let aiWorker = null;
let nextAiRequestId = 1;
const pendingAiRequests = new Map();

function rerender() {
  renderApp(root, state);
  bindEvents();
}

function retainedGameOptions() {
  return {
    existingAiDataset: state.aiDataset,
    aiSettings: state.aiSettings,
  };
}

function rebuildStateFromControls() {
  const presetSelect = document.querySelector("#preset-select");
  const seedInput = document.querySelector("#seed-input");
  stopReplay(state);
  state = createGameState({
    presetId: presetSelect?.value ?? state.presetId,
    seed: seedInput?.value ?? state.seed,
    ...retainedGameOptions(),
  });
}

function downloadTextFile(filename, content, mimeType = "application/json") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function finalizeAiAnalysis(requestId, analysis) {
  const pending = pendingAiRequests.get(requestId);
  if (!pending) {
    return;
  }

  pendingAiRequests.delete(requestId);
  recordAiAnalysis(state, pending.snapshot, analysis);

  if (pending.applyMove && analysis.bestAction) {
    state.selectedAction = analysis.bestActionKey;
    applyAction(state, analysis.bestAction, "ai");
  }

  setAiStatus(state, pending.autoRun ? "auto-ready" : "ready", false);
  rerender();

  if (pending.autoRun) {
    if (!state.gameOver && (state.aiContinuous || state.aiAutoRunRemaining > 0)) {
      window.setTimeout(runNextAiAutoTurn, 120);
      return;
    }

    state.aiContinuous = false;
    state.aiAutoRunRemaining = 0;
    setAiStatus(state, state.gameOver ? "stopped" : "complete", false);
    rerender();
  }
}

function failAiAnalysis(requestId, message) {
  pendingAiRequests.delete(requestId);
  state.aiContinuous = false;
  setAiError(state, message);
  state.aiAutoRunRemaining = 0;
  rerender();
}

function requestAiAnalysis({ applyMove = false, autoRun = false } = {}) {
  if (state.gameOver || state.aiBusy) {
    return;
  }

  stopReplay(state);
  clearAiError(state);
  setAiStatus(state, autoRun ? "auto-search" : "searching", true);
  const requestId = nextAiRequestId;
  nextAiRequestId += 1;
  const snapshot = createAiTrainingSnapshot(state);
  const payload = createAiRequestPayload(state);

  pendingAiRequests.set(requestId, {
    snapshot,
    applyMove,
    autoRun,
  });
  rerender();

  if (aiWorker) {
    aiWorker.postMessage({
      type: "analyze",
      requestId,
      payload,
    });
    return;
  }

  window.setTimeout(() => {
    try {
      const analysis = searchBestMove(payload);
      finalizeAiAnalysis(requestId, analysis);
    } catch (error) {
      failAiAnalysis(
        requestId,
        error instanceof Error ? error.message : String(error),
      );
    }
  }, 0);
}

function runNextAiAutoTurn() {
  if (state.gameOver) {
    state.aiContinuous = false;
    state.aiAutoRunRemaining = 0;
    setAiStatus(state, "stopped", false);
    rerender();
    return;
  }

  if (!state.aiContinuous && state.aiAutoRunRemaining <= 0) {
    setAiStatus(state, "complete", false);
    rerender();
    return;
  }

  if (!state.aiContinuous) {
    state.aiAutoRunRemaining -= 1;
  }
  requestAiAnalysis({ applyMove: true, autoRun: true });
}

function stopAiLoop() {
  if (
    !state.aiContinuous &&
    state.aiAutoRunRemaining <= 0 &&
    state.aiStatus !== "auto-search" &&
    state.aiStatus !== "auto-ready" &&
    state.aiStatus !== "stop-requested"
  ) {
    return;
  }

  state.aiContinuous = false;
  state.aiAutoRunRemaining = 0;

  if (state.aiBusy) {
    setAiStatus(state, "stop-requested", true);
  } else {
    setAiStatus(state, "idle", false);
  }

  rerender();
}

function bindEvents() {
  document.querySelector("#preset-select")?.addEventListener("change", () => {
    rebuildStateFromControls();
    rerender();
  });

  document.querySelector("#reset-button")?.addEventListener("click", () => {
    rebuildStateFromControls();
    rerender();
  });

  document.querySelector("#action-select")?.addEventListener("change", (event) => {
    setSelectedAction(state, event.target.value);
  });

  document.querySelector("#place-button")?.addEventListener("click", () => {
    stopReplay(state);
    applySelectedAction(state);
    rerender();
  });

  document.querySelector("#random-button")?.addEventListener("click", () => {
    stopReplay(state);
    applyRandomAction(state);
    rerender();
  });

  document.querySelector("#replay-prev")?.addEventListener("click", () => {
    stopReplay(state);
    stepReplay(state, -1);
    rerender();
  });

  document.querySelector("#replay-next")?.addEventListener("click", () => {
    stopReplay(state);
    stepReplay(state, 1);
    rerender();
  });

  document.querySelector("#replay-latest")?.addEventListener("click", () => {
    stopReplay(state);
    resetReplayToLatest(state);
    rerender();
  });

  document.querySelector("#replay-play")?.addEventListener("click", () => {
    if (!canReplay(state)) {
      return;
    }

    if (state.replayTimer !== null) {
      stopReplay(state);
      rerender();
      return;
    }

    startReplay(state, rerender);
    rerender();
  });

  document.querySelector("#ai-depth")?.addEventListener("change", (event) => {
    setAiSetting(state, "depth", event.target.value);
    rerender();
  });

  document.querySelector("#ai-beam")?.addEventListener("change", (event) => {
    setAiSetting(state, "beamWidth", event.target.value);
    rerender();
  });

  document.querySelector("#ai-analyze")?.addEventListener("click", () => {
    requestAiAnalysis({ applyMove: false, autoRun: false });
  });

  document.querySelector("#ai-move")?.addEventListener("click", () => {
    requestAiAnalysis({ applyMove: true, autoRun: false });
  });

  document.querySelector("#ai-run10")?.addEventListener("click", () => {
    if (state.aiBusy || state.gameOver) {
      return;
    }
    state.aiContinuous = false;
    state.aiAutoRunRemaining = 10;
    runNextAiAutoTurn();
  });

  document.querySelector("#ai-run")?.addEventListener("click", () => {
    if (state.aiBusy || state.gameOver) {
      return;
    }
    state.aiContinuous = true;
    state.aiAutoRunRemaining = 0;
    runNextAiAutoTurn();
  });

  document.querySelector("#ai-stop")?.addEventListener("click", () => {
    stopAiLoop();
  });

  document.querySelector("#ai-export")?.addEventListener("click", () => {
    downloadTextFile(
      createDatasetFilename(),
      serializeAiDataset(state.aiDataset),
      "application/json",
    );
  });

  document.querySelector("#ai-clear-log")?.addEventListener("click", () => {
    clearAiDataset(state);
    clearAiError(state);
    rerender();
  });
}

function initializeAiWorker() {
  if (!("Worker" in window)) {
    return;
  }

  try {
    aiWorker = new Worker(new URL("../worker/ai-worker.js", import.meta.url), {
      type: "module",
    });

    aiWorker.addEventListener("message", (event) => {
      const { type, requestId, analysis, error } = event.data ?? {};
      if (type === "analysis-result") {
        finalizeAiAnalysis(requestId, analysis);
        return;
      }
      if (type === "analysis-error") {
        failAiAnalysis(requestId, error ?? "Unknown AI worker error");
      }
    });

    aiWorker.addEventListener("error", (event) => {
      console.warn("AI worker failed, falling back to main thread.", event);
      aiWorker = null;
    });
  } catch (error) {
    console.warn("Failed to initialize AI worker, falling back to main thread.", error);
    aiWorker = null;
  }
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  try {
    await navigator.serviceWorker.register("./service-worker.js");
  } catch (error) {
    console.warn("Service worker registration failed", error);
  }
}

initializeAiWorker();
registerServiceWorker();
rerender();
