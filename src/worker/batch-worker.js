import { searchBestMove } from "../ai/search.js";
import {
  createAiSnapshot,
  createChainFocusTrainingSample,
  createSlimPolicyTrainingSample,
} from "../ai/dataset.js";
import { applyAction, createGameState } from "../app/state.js";

let activeRunId = 0;
let stopRequested = false;
const DATASET_FLUSH_SIZE = 12;
const CHAIN_FOCUS_THRESHOLD = 6;
const CHAIN_FOCUS_WINDOW = 6;

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
  let slimDatasetBuffer = [];
  let chainFocusBuffer = [];
  let recentDetailedTurns = [];

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
      lastSearchMs: 0,
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

      const result = applyAction(currentState, analysis.bestAction, "ai");
      totalTurns += 1;
      overallBestChain = Math.max(overallBestChain, currentState.maxChains);

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
        lastSearchMs: analysis.elapsedMs,
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
      lastSearchMs: 0,
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
    lastSearchMs: 0,
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
        lastSearchMs: 0,
        error: error instanceof Error ? error.message : String(error),
      });
    });
    return;
  }

  if (type === "stop-batch") {
    stopRequested = true;
  }
});
