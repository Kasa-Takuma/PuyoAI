import { searchBestMove } from "../ai/search.js";
import { applyAction, createGameState } from "../app/state.js";

let activeRunId = 0;
let stopRequested = false;

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
      const analysis = searchBestMove({
        board: currentState.board,
        currentPair: currentState.currentPair,
        nextQueue: currentState.nextQueue,
        settings: currentState.aiSettings,
      });

      applyAction(currentState, analysis.bestAction, "ai");
      totalTurns += 1;
      overallBestChain = Math.max(overallBestChain, currentState.maxChains);

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
