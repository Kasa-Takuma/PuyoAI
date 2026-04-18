import {
  createChainFocusDatasetFilename,
  createSlimDatasetFilename,
  serializeAiDataset,
} from "../ai/dataset.js";
import { renderBatchApp } from "./render.js";

const root = document.querySelector("#app");

const DEFAULT_PARALLEL_COUNT = Math.max(
  1,
  Math.min(
    8,
    (globalThis.navigator?.hardwareConcurrency
      ? globalThis.navigator.hardwareConcurrency - 1
      : 4),
  ),
);

let workerHandles = [];

function normalizeParallelCount(value) {
  return Math.max(1, Math.min(16, Number.parseInt(value, 10) || DEFAULT_PARALLEL_COUNT));
}

function normalizeAiSettings(settings) {
  return {
    depth: Math.max(1, Math.min(4, Number.parseInt(settings.depth, 10) || 3)),
    beamWidth: Math.max(4, Math.min(96, Number.parseInt(settings.beamWidth, 10) || 24)),
  };
}

function createWorkerSnapshot(id) {
  return {
    id,
    status: "idle",
    turn: 0,
    score: 0,
    maxChains: 0,
    totalTurns: 0,
    completedGames: 0,
    currentGame: 0,
    overallBestChain: 0,
    currentSeed: "",
    lastSearchMs: 0,
    error: null,
  };
}

function createBatchState() {
  const parallelCount = DEFAULT_PARALLEL_COUNT;
  return {
    parallelCount,
    seedBase: "batch",
    aiSettings: { depth: 3, beamWidth: 24 },
    running: false,
    stopRequested: false,
    slimDataset: [],
    chainFocusDataset: [],
    workers: Array.from({ length: parallelCount }, (_, index) =>
      createWorkerSnapshot(index + 1),
    ),
  };
}

let state = createBatchState();
let renderPending = false;

function rerender() {
  renderBatchApp(root, state);
  bindEvents();
}

function scheduleRender() {
  if (renderPending) {
    return;
  }
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    rerender();
  });
}

function setWorkerSnapshot(workerId, patch) {
  state.workers = state.workers.map((worker) =>
    worker.id === workerId ? { ...worker, ...patch } : worker,
  );
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

function updateRunningFlag() {
  const hasActiveWorker = state.workers.some((worker) =>
    ["running", "game-over", "stop-requested"].includes(worker.status),
  );

  if (!hasActiveWorker) {
    state.running = false;
    state.stopRequested = false;
  }
}

function handleWorkerMessage(event) {
  const { type, workerId, ...payload } = event.data ?? {};

  if (type === "batch-slim-dataset-chunk") {
    state.slimDataset = state.slimDataset.concat(payload.samples ?? []);
    scheduleRender();
    return;
  }

  if (type === "batch-chain-focus-dataset-chunk") {
    state.chainFocusDataset = state.chainFocusDataset.concat(payload.samples ?? []);
    scheduleRender();
    return;
  }

  if (type !== "batch-update") {
    return;
  }

  setWorkerSnapshot(workerId, payload);
  updateRunningFlag();
  scheduleRender();
}

function handleWorkerError(workerId, error) {
  setWorkerSnapshot(workerId, {
    status: "error",
    error: error instanceof Error ? error.message : String(error),
  });
  updateRunningFlag();
  scheduleRender();
}

function syncWorkers() {
  const targetCount = state.parallelCount;

  while (workerHandles.length < targetCount) {
    const id = workerHandles.length + 1;
    const worker = new Worker(new URL("../worker/batch-worker.js", import.meta.url), {
      type: "module",
    });
    worker.addEventListener("message", handleWorkerMessage);
    worker.addEventListener("error", (event) => {
      handleWorkerError(id, event.message || "Worker failed");
    });
    workerHandles.push({ id, worker });
  }

  while (workerHandles.length > targetCount) {
    const handle = workerHandles.pop();
    handle?.worker.terminate();
  }

  state.workers = Array.from({ length: targetCount }, (_, index) => {
    const previous = state.workers.find((worker) => worker.id === index + 1);
    return previous ?? createWorkerSnapshot(index + 1);
  });
}

function startAllWorkers() {
  syncWorkers();
  state.running = true;
  state.stopRequested = false;
  state.slimDataset = [];
  state.chainFocusDataset = [];
  state.workers = state.workers.map((worker) => createWorkerSnapshot(worker.id));
  rerender();

  for (const handle of workerHandles) {
    handle.worker.postMessage({
      type: "start-batch",
      payload: {
        workerId: handle.id,
        seedBase: state.seedBase,
        aiSettings: state.aiSettings,
        presetId: "sandbox",
      },
    });
  }
}

function stopAllWorkers() {
  state.stopRequested = true;
  state.workers = state.workers.map((worker) =>
    ["running", "game-over"].includes(worker.status)
      ? { ...worker, status: "stop-requested" }
      : worker,
  );
  rerender();

  for (const handle of workerHandles) {
    handle.worker.postMessage({ type: "stop-batch" });
  }
}

function bindEvents() {
  document.querySelector("#parallel-count")?.addEventListener("change", (event) => {
    if (state.running) {
      return;
    }
    state.parallelCount = normalizeParallelCount(event.target.value);
    syncWorkers();
    rerender();
  });

  document.querySelector("#batch-seed")?.addEventListener("change", (event) => {
    state.seedBase = event.target.value.trim() || "batch";
    rerender();
  });

  document.querySelector("#batch-depth")?.addEventListener("change", (event) => {
    state.aiSettings = normalizeAiSettings({
      ...state.aiSettings,
      depth: event.target.value,
    });
    rerender();
  });

  document.querySelector("#batch-beam")?.addEventListener("change", (event) => {
    state.aiSettings = normalizeAiSettings({
      ...state.aiSettings,
      beamWidth: event.target.value,
    });
    rerender();
  });

  document.querySelector("#start-all")?.addEventListener("click", () => {
    if (state.running) {
      return;
    }
    startAllWorkers();
  });

  document.querySelector("#stop-all")?.addEventListener("click", () => {
    if (!state.running) {
      return;
    }
    stopAllWorkers();
  });

  document.querySelector("#export-slim-dataset")?.addEventListener("click", () => {
    if (state.slimDataset.length === 0) {
      return;
    }

    downloadTextFile(
      createSlimDatasetFilename(),
      serializeAiDataset(state.slimDataset),
      "application/json",
    );
  });

  document.querySelector("#export-chain-focus-dataset")?.addEventListener("click", () => {
    if (state.chainFocusDataset.length === 0) {
      return;
    }

    downloadTextFile(
      createChainFocusDatasetFilename(),
      serializeAiDataset(state.chainFocusDataset),
      "application/json",
    );
  });

  document.querySelector("#clear-batch-dataset")?.addEventListener("click", () => {
    state.slimDataset = [];
    state.chainFocusDataset = [];
    rerender();
  });
}

syncWorkers();
window.addEventListener("beforeunload", () => {
  for (const handle of workerHandles) {
    handle.worker.terminate();
  }
});
rerender();
