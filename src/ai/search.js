import { encodeAction, enumerateLegalActions } from "../core/board.js";
import { resolveTurn } from "../core/engine.js";
import {
  extractBoardFeatures,
  featuresToVector,
  scoreBoardFeatures,
} from "./features.js";

export const SEARCH_OBJECTIVE = "chain_builder_v3";

function clampDepth(depth) {
  return Math.max(1, Math.min(4, Number.parseInt(depth, 10) || 1));
}

function clampBeamWidth(beamWidth) {
  return Math.max(4, Math.min(96, Number.parseInt(beamWidth, 10) || 24));
}

function normalizeSettings(settings = {}) {
  return {
    depth: clampDepth(settings.depth ?? 3),
    beamWidth: clampBeamWidth(settings.beamWidth ?? 24),
  };
}

function cloneAction(action) {
  return {
    column: action.column,
    orientation: action.orientation,
  };
}

function rememberCandidateNode(candidatePools, node, maxPerRoot = 3) {
  const nodes = candidatePools.get(node.rootKey) ?? [];
  nodes.push(node);
  nodes.sort((left, right) => right.searchScore - left.searchScore);
  if (nodes.length > maxPerRoot) {
    nodes.length = maxPerRoot;
  }
  candidatePools.set(node.rootKey, nodes);
}

function scoreTurnResult(result) {
  if (result.topout) {
    return -5_000_000;
  }

  if (result.totalChains === 0) {
    return 0;
  }

  if (result.totalChains === 1) {
    return -240 + result.totalScore * 0.15;
  }

  const chainValue = 1_100 * result.totalChains ** 3;
  const scoreValue = result.totalScore * 0.7;
  const allClearBonus = result.allClear ? 180 : 0;
  return chainValue + scoreValue + allClearBonus;
}

function createExpandedNode(node, pair, action, layerIndex) {
  const result = resolveTurn(node.board, pair, action);
  const features = extractBoardFeatures(result.finalBoard, {
    includeVirtualChains: false,
  });
  const heuristicScore = scoreBoardFeatures(features);
  const turnValue = scoreTurnResult(result);
  const cumulativeValue = node.cumulativeValue + turnValue;
  const searchScore = cumulativeValue + heuristicScore;
  const rootAction = node.rootAction ?? cloneAction(action);
  const rootKey = node.rootKey ?? encodeAction(action);
  const path = node.path.concat(cloneAction(action));
  const rootTurn = node.rootTurn ?? {
    score: result.totalScore,
    chains: result.totalChains,
    topout: result.topout,
    allClear: result.allClear,
  };

  return {
    board: result.finalBoard,
    rootAction,
    rootKey,
    rootTurn,
    path,
    cumulativeValue,
    searchScore,
    heuristicScore,
    bestDepth: layerIndex + 1,
    expandedNodes: node.expandedNodes + 1,
    lastResult: result,
    lastFeatures: features,
  };
}

function createCandidate(node) {
  const refinedFeatures = extractBoardFeatures(node.board, {
    includeVirtualChains: true,
  });
  const heuristicScore = scoreBoardFeatures(refinedFeatures);
  const searchScore = node.cumulativeValue + heuristicScore;

  return {
    action: cloneAction(node.rootAction),
    actionKey: node.rootKey,
    searchScore,
    cumulativeValue: node.cumulativeValue,
    heuristicScore,
    immediateScore: node.rootTurn.score,
    immediateChains: node.rootTurn.chains,
    immediateTopout: node.rootTurn.topout,
    immediateAllClear: node.rootTurn.allClear,
    bestDepth: node.bestDepth,
    line: node.path.map((action) => cloneAction(action)),
    leafFeatures: refinedFeatures,
    featureVector: featuresToVector(refinedFeatures),
    leafResult: {
      totalScore: node.lastResult.totalScore,
      totalChains: node.lastResult.totalChains,
      topout: node.lastResult.topout,
      allClear: node.lastResult.allClear,
    },
  };
}

export function searchBestMove({ board, currentPair, nextQueue = [], settings = {} }) {
  const startedAt = performance.now();
  const normalizedSettings = normalizeSettings(settings);
  const candidatePools = new Map();
  const rootActions = enumerateLegalActions(board, currentPair);

  let expandedNodeCount = 0;
  let frontier = rootActions.map((action) => {
    const node = createExpandedNode(
      {
        board,
        rootAction: null,
        rootKey: null,
        rootTurn: null,
        path: [],
        cumulativeValue: 0,
        expandedNodes: 0,
      },
      currentPair,
      action,
      0,
    );
    expandedNodeCount += 1;
    rememberCandidateNode(candidatePools, node);
    return node;
  });

  frontier.sort((left, right) => right.searchScore - left.searchScore);
  frontier = frontier.slice(0, normalizedSettings.beamWidth);

  for (let depthIndex = 1; depthIndex < normalizedSettings.depth; depthIndex += 1) {
    const pair = nextQueue[depthIndex - 1];
    if (!pair) {
      break;
    }

    const expanded = [];

    for (const node of frontier) {
      const actions = enumerateLegalActions(node.board, pair);
      for (const action of actions) {
        const child = createExpandedNode(node, pair, action, depthIndex);
        expandedNodeCount += 1;
        rememberCandidateNode(candidatePools, child);
        expanded.push(child);
      }
    }

    if (expanded.length === 0) {
      break;
    }

    expanded.sort((left, right) => right.searchScore - left.searchScore);
    frontier = expanded.slice(0, normalizedSettings.beamWidth);
  }

  const candidates = [...candidatePools.values()]
    .map((nodes) =>
      nodes
        .map((node) => createCandidate(node))
        .sort((left, right) => right.searchScore - left.searchScore)[0],
    )
    .sort(
    (left, right) => right.searchScore - left.searchScore,
  );
  const bestAction = candidates[0]?.action ?? cloneAction(rootActions[0]);
  const bestScore = candidates[0]?.searchScore ?? -Infinity;
  const elapsedMs = performance.now() - startedAt;

  return {
    objective: SEARCH_OBJECTIVE,
    settings: normalizedSettings,
    bestAction,
    bestActionKey: bestAction ? encodeAction(bestAction) : null,
    bestScore,
    candidates,
    expandedNodeCount,
    candidateCount: candidates.length,
    elapsedMs,
  };
}
