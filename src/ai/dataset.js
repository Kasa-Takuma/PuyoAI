import { boardToRows, encodeAction } from "../core/board.js";

function clonePair(pair) {
  return {
    axis: pair.axis,
    child: pair.child,
  };
}

function cloneAction(action) {
  return {
    column: action.column,
    orientation: action.orientation,
  };
}

function cloneTopCandidates(candidates, limit = 3) {
  return candidates.slice(0, limit).map((candidate) => ({
    actionKey: candidate.actionKey,
    searchScore: candidate.searchScore,
  }));
}

export function createAiSnapshot(state) {
  return {
    presetId: state.presetId,
    seed: state.seed,
    turn: state.turn,
    totalScore: state.totalScore,
    boardRows: boardToRows(state.board),
    currentPair: clonePair(state.currentPair),
    nextQueue: state.nextQueue.slice(0, 5).map((pair) => clonePair(pair)),
  };
}

export function createPolicyTrainingSample(snapshot, analysis) {
  return {
    kind: "search_policy",
    version: 1,
    createdAt: new Date().toISOString(),
    state: snapshot,
    search: {
      objective: analysis.objective,
      settings: analysis.settings,
      expandedNodeCount: analysis.expandedNodeCount,
      candidateCount: analysis.candidateCount,
      elapsedMs: analysis.elapsedMs,
    },
    bestAction: cloneAction(analysis.bestAction),
    bestActionKey: analysis.bestActionKey,
    bestScore: analysis.bestScore,
    candidates: analysis.candidates.map((candidate) => ({
      action: cloneAction(candidate.action),
      actionKey: candidate.actionKey,
      searchScore: candidate.searchScore,
      cumulativeValue: candidate.cumulativeValue,
      heuristicScore: candidate.heuristicScore,
      immediateScore: candidate.immediateScore,
      immediateChains: candidate.immediateChains,
      immediateTopout: candidate.immediateTopout,
      immediateAllClear: candidate.immediateAllClear,
      bestDepth: candidate.bestDepth,
      featureVector: [...candidate.featureVector],
      line: candidate.line.map((action) => cloneAction(action)),
    })),
  };
}

export function createSlimPolicyTrainingSample(snapshot, analysis) {
  return {
    kind: "search_policy_slim",
    version: 1,
    state: {
      boardRows: [...snapshot.boardRows],
      currentPair: clonePair(snapshot.currentPair),
      nextQueue: snapshot.nextQueue.map((pair) => clonePair(pair)),
    },
    search: {
      objective: analysis.objective,
      settings: analysis.settings,
    },
    bestActionKey: analysis.bestActionKey,
    topCandidates: cloneTopCandidates(analysis.candidates ?? []),
  };
}

export function createChainFocusTrainingSample(
  snapshot,
  analysis,
  {
    workerId,
    gameSeed,
    triggerTurn,
    triggerChains,
    triggerScore,
    offsetFromTrigger,
    thresholdChains = 10,
  },
) {
  return {
    ...createPolicyTrainingSample(snapshot, analysis),
    kind: "search_policy_chain_focus",
    focus: {
      workerId,
      gameSeed,
      triggerTurn,
      triggerChains,
      triggerScore,
      thresholdChains,
      offsetFromTrigger,
    },
  };
}

export function createValueTrainingSample({
  snapshot,
  analysis,
  workerId,
  gameSeed,
  features,
  immediate,
  future,
}) {
  return {
    kind: "search_value",
    version: 1,
    state: {
      boardRows: [...snapshot.boardRows],
      currentPair: clonePair(snapshot.currentPair),
      nextQueue: snapshot.nextQueue.map((pair) => clonePair(pair)),
      turn: snapshot.turn,
      totalScore: snapshot.totalScore,
    },
    context: {
      workerId,
      gameSeed,
      searchProfile: analysis.objective,
    },
    features,
    search: {
      objective: analysis.objective,
      settings: analysis.settings,
      bestActionKey: analysis.bestActionKey,
      bestScore: Math.round(analysis.bestScore),
      candidateCount: analysis.candidateCount,
      topCandidates: cloneTopCandidates(analysis.candidates ?? [], 5),
    },
    immediate,
    future,
  };
}

export function serializeAiDataset(dataset) {
  return JSON.stringify(dataset, null, 2);
}

export function createDatasetFilename() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return `puyoai-search-dataset-${iso}.json`;
}

export function createSlimDatasetFilename() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return `puyoai-search-slim-${iso}.json`;
}

export function createChainFocusDatasetFilename() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return `puyoai-search-chain-focus-${iso}.json`;
}

export function createValueDatasetFilename() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return `puyoai-search-value-${iso}.json`;
}

export function createBenchmarkReportFilename() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return `puyoai-benchmark-summary-${iso}.json`;
}

export function summarizeBestAction(analysis) {
  if (!analysis?.bestAction) {
    return "No action";
  }

  return `${analysis.bestAction.orientation}:${analysis.bestAction.column + 1}`;
}

export function candidateActionKey(candidate) {
  return candidate.actionKey ?? encodeAction(candidate.action);
}
