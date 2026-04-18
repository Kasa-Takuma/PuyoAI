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

export function serializeAiDataset(dataset) {
  return JSON.stringify(dataset, null, 2);
}

export function createDatasetFilename() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return `puyoai-search-dataset-${iso}.json`;
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
