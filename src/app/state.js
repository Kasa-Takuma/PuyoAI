import {
  decodeAction,
  encodeAction,
  enumerateLegalActions,
  cloneBoard,
} from "../core/board.js";
import { resolveTurn } from "../core/engine.js";
import { PRESETS } from "../core/presets.js";
import { NEXT_PREVIEW_COUNT } from "../core/constants.js";
import { createRng, fillQueue } from "../core/randomizer.js";
import {
  createAiSnapshot,
  createPolicyTrainingSample,
} from "../ai/dataset.js";

function clonePair(pair) {
  return { axis: pair.axis, child: pair.child };
}

function cloneQueue(queue) {
  return queue.map((pair) => clonePair(pair));
}

function cloneAction(action) {
  return {
    column: action.column,
    orientation: action.orientation,
  };
}

function normalizeSeed(seedText) {
  const trimmed = seedText.trim();
  return trimmed.length > 0 ? trimmed : "puyoai";
}

function normalizeAiSettings(aiSettings) {
  return {
    depth: Math.max(1, Math.min(4, Number.parseInt(aiSettings?.depth, 10) || 3)),
    beamWidth: Math.max(
      4,
      Math.min(96, Number.parseInt(aiSettings?.beamWidth, 10) || 24),
    ),
  };
}

function normalizeAiMode(aiMode) {
  return aiMode === "learned" ? "learned" : "search";
}

export function createGameState({
  presetId = "sandbox",
  seed = "puyoai",
  existingAiDataset = [],
  aiSettings = { depth: 3, beamWidth: 24 },
  aiMode = "search",
} = {}) {
  const preset = PRESETS[presetId] ?? PRESETS.sandbox;
  const normalizedSeed = normalizeSeed(seed);
  const rng = createRng(`${presetId}:${normalizedSeed}`);

  const currentPair = clonePair(preset.currentPair);
  const nextQueue = cloneQueue(preset.nextQueue);
  fillQueue(rng, nextQueue, 8);

  const legalActions = enumerateLegalActions(preset.board, currentPair);
  const selectedAction = preset.suggestedAction
    ? encodeAction(preset.suggestedAction)
    : encodeAction(legalActions[0]);

  return {
    presetId,
    seed: normalizedSeed,
    rng,
    board: cloneBoard(preset.board),
    currentPair,
    nextQueue,
    turn: 1,
    totalScore: 0,
    maxChains: 0,
    gameOver: false,
    lastResult: null,
    lastAction: null,
    replayIndex: -1,
    replayTimer: null,
    history: [],
    selectedAction,
    aiMode: normalizeAiMode(aiMode),
    aiSettings: normalizeAiSettings(aiSettings),
    aiBusy: false,
    aiStatus: "idle",
    aiAnalysis: null,
    aiDataset: [...existingAiDataset],
    aiAutoRunRemaining: 0,
    aiContinuous: false,
    aiLastError: null,
  };
}

export function getPreset(presetId) {
  return PRESETS[presetId] ?? PRESETS.sandbox;
}

export function getLegalActions(state) {
  return enumerateLegalActions(state.board, state.currentPair);
}

export function setSelectedAction(state, actionValue) {
  state.selectedAction = actionValue;
}

export function setAiSetting(state, key, value) {
  const next = {
    ...state.aiSettings,
    [key]: value,
  };
  state.aiSettings = normalizeAiSettings(next);
}

export function setAiMode(state, aiMode) {
  state.aiMode = normalizeAiMode(aiMode);
  state.aiAnalysis = null;
  state.aiLastError = null;
}

export function getReplayEvents(state) {
  return state.lastResult?.events ?? [];
}

export function getDisplayedBoard(state) {
  const events = getReplayEvents(state);
  if (state.replayIndex >= 0 && state.replayIndex < events.length) {
    return events[state.replayIndex].board;
  }
  return state.board;
}

export function getDisplayedEvent(state) {
  const events = getReplayEvents(state);
  if (state.replayIndex >= 0 && state.replayIndex < events.length) {
    return events[state.replayIndex];
  }
  return null;
}

export function canReplay(state) {
  return getReplayEvents(state).length > 0;
}

export function stepReplay(state, delta) {
  const events = getReplayEvents(state);
  if (events.length === 0) {
    state.replayIndex = -1;
    return;
  }

  state.replayIndex = Math.max(
    0,
    Math.min(events.length - 1, state.replayIndex + delta),
  );
}

export function stopReplay(state) {
  if (state.replayTimer !== null) {
    window.clearInterval(state.replayTimer);
    state.replayTimer = null;
  }
}

export function startReplay(state, onTick) {
  stopReplay(state);
  const events = getReplayEvents(state);
  if (events.length === 0) {
    return;
  }

  state.replayIndex = 0;
  state.replayTimer = window.setInterval(() => {
    if (state.replayIndex >= events.length - 1) {
      stopReplay(state);
      onTick();
      return;
    }
    state.replayIndex += 1;
    onTick();
  }, 480);
}

export function resetReplayToLatest(state) {
  const events = getReplayEvents(state);
  state.replayIndex = events.length > 0 ? events.length - 1 : -1;
}

export function createAiRequestPayload(state) {
  return {
    mode: state.aiMode,
    board: cloneBoard(state.board),
    currentPair: clonePair(state.currentPair),
    nextQueue: cloneQueue(state.nextQueue),
    settings: { ...state.aiSettings },
  };
}

export function createAiTrainingSnapshot(state) {
  return createAiSnapshot(state);
}

export function recordAiAnalysis(state, snapshot, analysis) {
  state.aiAnalysis = {
    snapshot,
    ...analysis,
  };
  state.aiLastError = null;
  if (analysis.kind !== "learned") {
    state.aiDataset.unshift(createPolicyTrainingSample(snapshot, analysis));
  }
}

export function clearAiDataset(state) {
  state.aiDataset = [];
}

export function setAiStatus(state, status, busy = false) {
  state.aiStatus = status;
  state.aiBusy = busy;
}

export function setAiError(state, message) {
  state.aiStatus = "error";
  state.aiBusy = false;
  state.aiLastError = message;
}

export function clearAiError(state) {
  state.aiLastError = null;
}

export function applyAction(state, action, source = "manual") {
  if (state.gameOver) {
    return null;
  }

  const normalizedAction =
    typeof action === "string" ? decodeAction(action) : cloneAction(action);
  const currentPair = clonePair(state.currentPair);
  const result = resolveTurn(state.board, currentPair, normalizedAction);

  state.board = cloneBoard(result.finalBoard);
  state.totalScore += result.totalScore;
  state.maxChains = Math.max(state.maxChains, result.totalChains);
  state.lastResult = result;
  state.lastAction = normalizedAction;
  state.history.unshift({
    turn: state.turn,
    pair: currentPair,
    action: normalizedAction,
    score: result.totalScore,
    chains: result.totalChains,
    topout: result.topout,
    source,
  });
  state.turn += 1;
  state.gameOver = result.topout;
  state.currentPair = state.nextQueue.shift();
  fillQueue(state.rng, state.nextQueue, 8);
  resetReplayToLatest(state);

  if (!state.gameOver) {
    const legalActions = getLegalActions(state);
    state.selectedAction = encodeAction(legalActions[0]);
  }

  return result;
}

export function applySelectedAction(state) {
  return applyAction(state, state.selectedAction, "manual");
}

export function applyRandomAction(state) {
  const legalActions = getLegalActions(state);
  const randomAction = legalActions[state.rng.nextInt(legalActions.length)];
  state.selectedAction = encodeAction(randomAction);
  return applyAction(state, randomAction, "random");
}

export function previewQueue(state) {
  return state.nextQueue.slice(0, NEXT_PREVIEW_COUNT);
}
