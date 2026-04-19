import {
  createBenchmarkReportFilename,
  createChainFocusDatasetFilename,
  createSlimDatasetFilename,
  serializeAiDataset,
} from "../ai/dataset.js";
import { DEFAULT_SEARCH_PROFILE_ID } from "../ai/search-profiles.js";
import { renderBatchApp } from "./render.js";

const root = document.querySelector("#app");
const HIGH_CHAIN_THRESHOLD = 7;
const MAX_CHAIN_EVENTS_IN_REPORT = 100_000;

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
    searchProfile:
      typeof settings?.searchProfile === "string" && settings.searchProfile.length > 0
        ? settings.searchProfile
        : DEFAULT_SEARCH_PROFILE_ID,
  };
}

function createWorkerSnapshot(id) {
  return {
    id,
    searchProfile: DEFAULT_SEARCH_PROFILE_ID,
    status: "idle",
    turn: 0,
    score: 0,
    maxChains: 0,
    totalTurns: 0,
    completedGames: 0,
    currentGame: 0,
    overallBestChain: 0,
    currentSeed: "",
    sessionScore: 0,
    chainHistogram: {},
    lastSearchMs: 0,
    error: null,
  };
}

function createWorkerSnapshotWithProfile(id, searchProfile = DEFAULT_SEARCH_PROFILE_ID) {
  return {
    ...createWorkerSnapshot(id),
    searchProfile:
      typeof searchProfile === "string" && searchProfile.length > 0
        ? searchProfile
        : DEFAULT_SEARCH_PROFILE_ID,
  };
}

function createBatchState() {
  const parallelCount = DEFAULT_PARALLEL_COUNT;
  return {
    parallelCount,
    seedBase: "batch",
    aiSettings: { depth: 3, beamWidth: 24, searchProfile: DEFAULT_SEARCH_PROFILE_ID },
    running: false,
    stopRequested: false,
    slimDataset: [],
    chainFocusDataset: [],
    chainEvents: [],
    droppedChainEvents: 0,
    workers: Array.from({ length: parallelCount }, (_, index) =>
      createWorkerSnapshotWithProfile(index + 1),
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

function mergeHistogram(target, source = {}) {
  for (const [chain, count] of Object.entries(source)) {
    target[chain] = (target[chain] ?? 0) + count;
  }
  return target;
}

function histogramAtLeast(histogram, threshold) {
  return Object.entries(histogram).reduce((sum, [chain, count]) => {
    return Number(chain) >= threshold ? sum + count : sum;
  }, 0);
}

function buildProfileSummaries(workers) {
  const summaries = new Map();

  for (const worker of workers) {
    const profile = worker.searchProfile || "unknown";
    const summary =
      summaries.get(profile) ??
      {
        searchProfile: profile,
        workers: 0,
        totalTurns: 0,
        completedGames: 0,
        sessionScore: 0,
        bestChain: 0,
        chainHistogram: {},
      };

    summary.workers += 1;
    summary.totalTurns += worker.totalTurns;
    summary.completedGames += worker.completedGames;
    summary.sessionScore += worker.sessionScore ?? 0;
    summary.bestChain = Math.max(
      summary.bestChain,
      worker.overallBestChain,
      worker.maxChains,
    );
    mergeHistogram(summary.chainHistogram, worker.chainHistogram);
    summaries.set(profile, summary);
  }

  return [...summaries.values()].map((summary) => ({
    ...summary,
    chains7Plus: histogramAtLeast(summary.chainHistogram, HIGH_CHAIN_THRESHOLD),
    chains10Plus: histogramAtLeast(summary.chainHistogram, 10),
  }));
}

function buildBenchmarkReport() {
  const chainHistogram = state.workers.reduce(
    (histogram, worker) => mergeHistogram(histogram, worker.chainHistogram),
    {},
  );
  const totalTurns = state.workers.reduce(
    (sum, worker) => sum + worker.totalTurns,
    0,
  );
  const completedGames = state.workers.reduce(
    (sum, worker) => sum + worker.completedGames,
    0,
  );
  const sessionScore = state.workers.reduce(
    (sum, worker) => sum + (worker.sessionScore ?? 0),
    0,
  );
  const bestChain = state.workers.reduce(
    (best, worker) => Math.max(best, worker.overallBestChain, worker.maxChains),
    0,
  );
  const chainEvents = state.chainEvents.slice(0, MAX_CHAIN_EVENTS_IN_REPORT);

  return {
    kind: "puyoai_batch_benchmark_summary",
    version: 1,
    createdAt: new Date().toISOString(),
    thresholds: {
      highChain: HIGH_CHAIN_THRESHOLD,
      chainFocus: 10,
    },
    settings: {
      seedBase: state.seedBase,
      parallelCount: state.parallelCount,
      depth: state.aiSettings.depth,
      beamWidth: state.aiSettings.beamWidth,
      bulkSearchProfile: state.aiSettings.searchProfile,
      workerProfiles: state.workers.map((worker) => ({
        workerId: worker.id,
        searchProfile: worker.searchProfile,
      })),
    },
    totals: {
      totalTurns,
      completedGames,
      sessionScore,
      bestChain,
      chains7Plus: histogramAtLeast(chainHistogram, HIGH_CHAIN_THRESHOLD),
      chains10Plus: histogramAtLeast(chainHistogram, 10),
      slimSamples: state.slimDataset.length,
      chainFocusSamples: state.chainFocusDataset.length,
      highChainEventCount: state.chainEvents.length,
      exportedHighChainEvents: chainEvents.length,
      droppedChainEvents: state.droppedChainEvents,
      highChainEventsTruncated:
        state.chainEvents.length > MAX_CHAIN_EVENTS_IN_REPORT,
    },
    chainHistogram,
    profileSummaries: buildProfileSummaries(state.workers),
    workers: state.workers.map((worker) => ({
      workerId: worker.id,
      searchProfile: worker.searchProfile,
      status: worker.status,
      currentGame: worker.currentGame,
      completedGames: worker.completedGames,
      currentTurn: worker.turn,
      totalTurns: worker.totalTurns,
      currentScore: worker.score,
      sessionScore: worker.sessionScore ?? 0,
      currentGameMaxChains: worker.maxChains,
      bestChain: Math.max(worker.overallBestChain, worker.maxChains),
      currentSeed: worker.currentSeed,
      chainHistogram: worker.chainHistogram ?? {},
      chains7Plus: histogramAtLeast(worker.chainHistogram, HIGH_CHAIN_THRESHOLD),
      chains10Plus: histogramAtLeast(worker.chainHistogram, 10),
    })),
    highChainEvents: chainEvents,
  };
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

  if (type === "batch-chain-event") {
    if (state.chainEvents.length < MAX_CHAIN_EVENTS_IN_REPORT) {
      state.chainEvents = state.chainEvents.concat(payload.event);
    } else {
      state.droppedChainEvents += 1;
    }
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
    return (
      previous ??
      createWorkerSnapshotWithProfile(index + 1, state.aiSettings.searchProfile)
    );
  });
}

function startAllWorkers() {
  syncWorkers();
  state.running = true;
  state.stopRequested = false;
  state.slimDataset = [];
  state.chainFocusDataset = [];
  state.chainEvents = [];
  state.droppedChainEvents = 0;
  state.workers = state.workers.map((worker) =>
    createWorkerSnapshotWithProfile(worker.id, worker.searchProfile),
  );
  rerender();

  for (const handle of workerHandles) {
    const workerConfig = state.workers.find((worker) => worker.id === handle.id);
    handle.worker.postMessage({
      type: "start-batch",
      payload: {
        workerId: handle.id,
        seedBase: state.seedBase,
        aiSettings: normalizeAiSettings({
          ...state.aiSettings,
          searchProfile: workerConfig?.searchProfile ?? state.aiSettings.searchProfile,
        }),
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

  document.querySelector("#batch-search-profile")?.addEventListener("change", (event) => {
    state.aiSettings = {
      ...state.aiSettings,
      searchProfile: event.target.value,
    };
    rerender();
  });

  document
    .querySelector("#apply-search-profile-to-all")
    ?.addEventListener("click", () => {
      if (state.running) {
        return;
      }
      state.workers = state.workers.map((worker) => ({
        ...worker,
        searchProfile: state.aiSettings.searchProfile,
      }));
      rerender();
    });

  document.querySelectorAll("[data-worker-search-profile]").forEach((element) => {
    element.addEventListener("change", (event) => {
      if (state.running) {
        return;
      }
      const workerId = Number.parseInt(event.target.dataset.workerSearchProfile, 10);
      const worker = state.workers.find((entry) => entry.id === workerId);
      if (!worker) {
        return;
      }
      setWorkerSnapshot(workerId, {
        searchProfile: event.target.value,
      });
      rerender();
    });
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

  document.querySelector("#export-benchmark-report")?.addEventListener("click", () => {
    const report = buildBenchmarkReport();
    if (report.totals.totalTurns === 0) {
      return;
    }

    downloadTextFile(
      createBenchmarkReportFilename(),
      JSON.stringify(report, null, 2),
      "application/json",
    );
  });

  document.querySelector("#clear-batch-dataset")?.addEventListener("click", () => {
    state.slimDataset = [];
    state.chainFocusDataset = [];
    state.chainEvents = [];
    state.droppedChainEvents = 0;
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
