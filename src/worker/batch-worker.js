import { searchBestMove } from "../ai/search.js";
import {
  createAiSnapshot,
  createChainFocusTrainingSample,
  createSlimPolicyTrainingSample,
  createValueTrainingSample,
} from "../ai/dataset.js";
import { extractBoardFeatures } from "../ai/features.js";
import { applyAction, createGameState } from "../app/state.js";

let activeRunId = 0;
let stopRequested = false;
const DATASET_FLUSH_SIZE = 12;
const HIGH_CHAIN_THRESHOLD = 7;
const CHAIN_FOCUS_THRESHOLD = 10;
const CHAIN_FOCUS_WINDOW = 12;
const VALUE_HORIZONS = Object.freeze([12, 24, 48]);
const VALUE_MAX_HORIZON = Math.max(...VALUE_HORIZONS);
const BENCHMARK_FEATURE_KEYS = Object.freeze([
  "stackCells",
  "maxHeight",
  "hiddenCells",
  "dangerCells",
  "surfaceRoughness",
  "steepWalls",
  "valleyPenalty",
  "adjacency",
  "group2Count",
  "group3Count",
  "surfaceExtendableGroup2Count",
  "surfaceReadyGroup3Count",
  "isolatedSingles",
  "colorBalance",
  "columnsUsed",
  "bestVirtualChain",
  "bestVirtualScore",
  "virtualChainCount2Plus",
  "virtualChainCount3Plus",
  "topVirtualChainSum",
  "topVirtualScoreSum",
]);

function nextTick() {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function postWorkerUpdate(workerId, payload) {
  self.postMessage({
    type: "batch-update",
    workerId,
    ...payload,
  });
}

function cloneHistogram(histogram) {
  return Object.fromEntries(Object.entries(histogram));
}

function incrementHistogram(histogram, chains) {
  const key = String(chains);
  histogram[key] = (histogram[key] ?? 0) + 1;
}

function postChainEvent(workerId, event) {
  self.postMessage({
    type: "batch-chain-event",
    workerId,
    event,
  });
}

function pickBenchmarkFeatures(features) {
  return Object.fromEntries(
    BENCHMARK_FEATURE_KEYS.map((key) => [key, features[key] ?? 0]),
  );
}

function topCandidateSummary(candidates, limit = 3) {
  return (candidates ?? []).slice(0, limit).map((candidate) => ({
    actionKey: candidate.actionKey,
    searchScore: Math.round(candidate.searchScore),
    immediateChains: candidate.immediateChains,
    immediateScore: candidate.immediateScore,
    bestDepth: candidate.bestDepth,
  }));
}

function flushDatasetChunk(workerId, type, datasetBuffer) {
  if (datasetBuffer.length === 0) {
    return [];
  }

  self.postMessage({
    type,
    workerId,
    samples: datasetBuffer,
  });
  return [];
}

function createEmptyFutureLabel() {
  return {
    complete: false,
    stepsObserved: 0,
    totalScore: 0,
    maxChains: 0,
    chainEvents: 0,
    chains7Plus: 0,
    chains10Plus: 0,
    chains11Plus: 0,
    chains12Plus: 0,
    chains13Plus: 0,
    topout: false,
    topoutAt: null,
  };
}

function createFutureLabels() {
  return Object.fromEntries(
    VALUE_HORIZONS.map((horizon) => [horizon, createEmptyFutureLabel()]),
  );
}

function updateFutureLabel(label, result, stepOffset) {
  label.stepsObserved = stepOffset;
  label.totalScore += result.totalScore;
  label.maxChains = Math.max(label.maxChains, result.totalChains);
  if (result.totalChains > 0) {
    label.chainEvents += 1;
  }
  if (result.totalChains >= 7) {
    label.chains7Plus += 1;
  }
  if (result.totalChains >= 10) {
    label.chains10Plus += 1;
  }
  if (result.totalChains >= 11) {
    label.chains11Plus += 1;
  }
  if (result.totalChains >= 12) {
    label.chains12Plus += 1;
  }
  if (result.totalChains >= 13) {
    label.chains13Plus += 1;
  }
  if (result.topout && !label.topout) {
    label.topout = true;
    label.topoutAt = stepOffset;
  }
}

function finalizeValueEntry(entry, terminal = false) {
  const future = Object.fromEntries(
    VALUE_HORIZONS.map((horizon) => {
      const label = { ...entry.future[horizon] };
      label.complete = terminal || label.stepsObserved >= horizon;
      return [horizon, label];
    }),
  );

  return createValueTrainingSample({
    snapshot: entry.snapshot,
    analysis: entry.analysis,
    workerId: entry.workerId,
    gameSeed: entry.gameSeed,
    features: entry.features,
    immediate: entry.immediate,
    future,
  });
}

function observeValueFuture(pendingEntries, result, terminal = false) {
  const completed = [];

  for (const entry of pendingEntries) {
    entry.stepsObserved += 1;
    for (const horizon of VALUE_HORIZONS) {
      if (entry.stepsObserved <= horizon) {
        updateFutureLabel(entry.future[horizon], result, entry.stepsObserved);
      }
    }
  }

  while (
    pendingEntries.length > 0 &&
    (terminal || pendingEntries[0].stepsObserved >= VALUE_MAX_HORIZON)
  ) {
    completed.push(finalizeValueEntry(pendingEntries.shift(), terminal));
  }

  return completed;
}

async function runBatchLoop({
  workerId,
  seedBase,
  aiSettings,
  presetId = "sandbox",
}) {
  activeRunId += 1;
  const runId = activeRunId;
  stopRequested = false;

  let completedGames = 0;
  let totalTurns = 0;
  let overallBestChain = 0;
  let currentState = null;
  let currentSeed = "";
  let sessionScore = 0;
  let lastHighChainTotalTurn = null;
  let lastFocusChainTotalTurn = null;
  let chainEventsTotal = 0;
  const chainHistogram = {};
  let slimDatasetBuffer = [];
  let chainFocusBuffer = [];
  let valueDatasetBuffer = [];
  let recentDetailedTurns = [];
  let pendingValueEntries = [];

  while (!stopRequested && runId === activeRunId) {
    currentSeed = `${seedBase}:worker-${workerId}:game-${completedGames + 1}`;
    currentState = createGameState({
      presetId,
      seed: currentSeed,
      aiSettings,
    });

    postWorkerUpdate(workerId, {
      status: "running",
      currentGame: completedGames + 1,
      completedGames,
      totalTurns,
      overallBestChain,
      turn: 0,
      score: 0,
      maxChains: 0,
      currentSeed,
      searchProfile: aiSettings.searchProfile,
      lastSearchMs: 0,
      sessionScore,
      chainEventsTotal,
      chainHistogram: cloneHistogram(chainHistogram),
      error: null,
    });

    while (!stopRequested && runId === activeRunId && !currentState.gameOver) {
      const snapshot = createAiSnapshot(currentState);
      const analysis = searchBestMove({
        board: currentState.board,
        currentPair: currentState.currentPair,
        nextQueue: currentState.nextQueue,
        settings: currentState.aiSettings,
      });
      slimDatasetBuffer.push(createSlimPolicyTrainingSample(snapshot, analysis));
      if (slimDatasetBuffer.length >= DATASET_FLUSH_SIZE) {
        slimDatasetBuffer = flushDatasetChunk(
          workerId,
          "batch-slim-dataset-chunk",
          slimDatasetBuffer,
        );
      }

      const preActionBoard = currentState.board;
      const preActionFeatures = extractBoardFeatures(preActionBoard, {
        includeVirtualChains: true,
      });
      const result = applyAction(currentState, analysis.bestAction, "ai");
      totalTurns += 1;
      sessionScore += result.totalScore;
      overallBestChain = Math.max(overallBestChain, currentState.maxChains);
      if (result.totalChains > 0) {
        chainEventsTotal += 1;
      }

      pendingValueEntries.push({
        workerId,
        gameSeed: currentSeed,
        snapshot,
        analysis,
        features: pickBenchmarkFeatures(preActionFeatures),
        immediate: {
          chains: result.totalChains,
          score: result.totalScore,
          topout: result.topout,
          allClear: result.allClear,
          actionKey: analysis.bestActionKey,
        },
        future: createFutureLabels(),
        stepsObserved: 0,
      });
      valueDatasetBuffer.push(
        ...observeValueFuture(pendingValueEntries, result, currentState.gameOver),
      );
      if (valueDatasetBuffer.length >= DATASET_FLUSH_SIZE) {
        valueDatasetBuffer = flushDatasetChunk(
          workerId,
          "batch-value-dataset-chunk",
          valueDatasetBuffer,
        );
      }

      if (result.totalChains >= HIGH_CHAIN_THRESHOLD) {
        const benchmarkFeatures = extractBoardFeatures(preActionBoard, {
          includeVirtualChains: true,
        });
        const topCandidates = topCandidateSummary(analysis.candidates);
        const searchMargin =
          topCandidates.length >= 2
            ? topCandidates[0].searchScore - topCandidates[1].searchScore
            : null;
        const turnsSincePrevious7Plus =
          lastHighChainTotalTurn === null
            ? null
            : totalTurns - lastHighChainTotalTurn;
        const turnsSincePrevious10Plus =
          lastFocusChainTotalTurn === null
            ? null
            : totalTurns - lastFocusChainTotalTurn;

        incrementHistogram(chainHistogram, result.totalChains);
        postChainEvent(workerId, {
          workerId,
          searchProfile: aiSettings.searchProfile,
          seed: currentSeed,
          game: completedGames + 1,
          turn: currentState.turn - 1,
          totalTurns,
          chains: result.totalChains,
          score: result.totalScore,
          totalScore: currentState.totalScore,
          maxChains: currentState.maxChains,
          actionKey: analysis.bestActionKey,
          currentPair: snapshot.currentPair,
          nextQueueHead: snapshot.nextQueue.slice(0, 3),
          topCandidates,
          searchMargin,
          turnsSincePrevious7Plus,
          turnsSincePrevious10Plus,
          featureBucket:
            result.totalChains >= CHAIN_FOCUS_THRESHOLD
              ? "10+"
              : String(result.totalChains),
          features: pickBenchmarkFeatures(benchmarkFeatures),
        });
        lastHighChainTotalTurn = totalTurns;
        if (result.totalChains >= CHAIN_FOCUS_THRESHOLD) {
          lastFocusChainTotalTurn = totalTurns;
        }
      }

      recentDetailedTurns.push({
        snapshot,
        analysis,
        result: {
          totalChains: result.totalChains,
          totalScore: result.totalScore,
        },
      });
      if (recentDetailedTurns.length > CHAIN_FOCUS_WINDOW) {
        recentDetailedTurns.shift();
      }

      if (result.totalChains >= CHAIN_FOCUS_THRESHOLD) {
        const triggerTurn = snapshot.turn;
        chainFocusBuffer.push(
          ...recentDetailedTurns.map((entry) =>
            createChainFocusTrainingSample(entry.snapshot, entry.analysis, {
              workerId,
              gameSeed: currentSeed,
              triggerTurn,
              triggerChains: result.totalChains,
              triggerScore: result.totalScore,
              thresholdChains: CHAIN_FOCUS_THRESHOLD,
              offsetFromTrigger: entry.snapshot.turn - triggerTurn,
            }),
          ),
        );

        if (chainFocusBuffer.length >= DATASET_FLUSH_SIZE) {
          chainFocusBuffer = flushDatasetChunk(
            workerId,
            "batch-chain-focus-dataset-chunk",
            chainFocusBuffer,
          );
        }
      }

      postWorkerUpdate(workerId, {
        status: currentState.gameOver ? "game-over" : "running",
        currentGame: completedGames + 1,
        completedGames,
        totalTurns,
        overallBestChain,
        turn: currentState.turn - 1,
        score: currentState.totalScore,
        maxChains: currentState.maxChains,
        currentSeed,
        searchProfile: aiSettings.searchProfile,
        lastSearchMs: analysis.elapsedMs,
        sessionScore,
        chainEventsTotal,
        chainHistogram: cloneHistogram(chainHistogram),
        error: null,
      });

      await nextTick();
    }

    if (runId !== activeRunId || stopRequested) {
      break;
    }

    completedGames += 1;
    overallBestChain = Math.max(overallBestChain, currentState.maxChains);
    slimDatasetBuffer = flushDatasetChunk(
      workerId,
      "batch-slim-dataset-chunk",
      slimDatasetBuffer,
    );
    chainFocusBuffer = flushDatasetChunk(
      workerId,
      "batch-chain-focus-dataset-chunk",
      chainFocusBuffer,
    );
    valueDatasetBuffer.push(...pendingValueEntries.map((entry) => finalizeValueEntry(entry, true)));
    pendingValueEntries = [];
    valueDatasetBuffer = flushDatasetChunk(
      workerId,
      "batch-value-dataset-chunk",
      valueDatasetBuffer,
    );
    recentDetailedTurns = [];

    postWorkerUpdate(workerId, {
      status: "running",
      currentGame: completedGames + 1,
      completedGames,
      totalTurns,
      overallBestChain,
      turn: 0,
      score: 0,
      maxChains: 0,
      currentSeed: `${seedBase}:worker-${workerId}:game-${completedGames + 1}`,
      searchProfile: aiSettings.searchProfile,
      lastSearchMs: 0,
      sessionScore,
      chainEventsTotal,
      chainHistogram: cloneHistogram(chainHistogram),
      error: null,
    });

    await nextTick();
  }

  slimDatasetBuffer = flushDatasetChunk(
    workerId,
    "batch-slim-dataset-chunk",
    slimDatasetBuffer,
  );
  chainFocusBuffer = flushDatasetChunk(
    workerId,
    "batch-chain-focus-dataset-chunk",
    chainFocusBuffer,
  );
  valueDatasetBuffer = flushDatasetChunk(
    workerId,
    "batch-value-dataset-chunk",
    valueDatasetBuffer,
  );

  postWorkerUpdate(workerId, {
    status: "stopped",
    currentGame: currentState ? completedGames + 1 : 0,
    completedGames,
    totalTurns,
    overallBestChain,
    turn: currentState ? currentState.turn - 1 : 0,
    score: currentState ? currentState.totalScore : 0,
    maxChains: currentState ? currentState.maxChains : 0,
    currentSeed,
    searchProfile: aiSettings.searchProfile,
    lastSearchMs: 0,
    sessionScore,
    chainEventsTotal,
    chainHistogram: cloneHistogram(chainHistogram),
    error: null,
  });
}

self.addEventListener("message", (event) => {
  const { type, payload } = event.data ?? {};

  if (type === "start-batch") {
    runBatchLoop(payload).catch((error) => {
      const workerId = payload?.workerId ?? 0;
      postWorkerUpdate(workerId, {
        status: "error",
        currentGame: 0,
        completedGames: 0,
        totalTurns: 0,
        overallBestChain: 0,
        turn: 0,
        score: 0,
        maxChains: 0,
        currentSeed: "",
        searchProfile: payload?.aiSettings?.searchProfile ?? "",
        lastSearchMs: 0,
        sessionScore: 0,
        chainEventsTotal: 0,
        chainHistogram: {},
        error: error instanceof Error ? error.message : String(error),
      });
    });
    return;
  }

  if (type === "stop-batch") {
    stopRequested = true;
  }
});
