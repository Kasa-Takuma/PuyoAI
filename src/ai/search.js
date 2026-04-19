import { encodeAction, enumerateLegalActions } from "../core/board.js";
import { resolveTurn } from "../core/engine.js";
import {
  extractBoardFeatures,
  featuresToVector,
  scoreBoardFeatures,
} from "./features.js";
import {
  DEFAULT_SEARCH_PROFILE_ID,
  getSearchProfile,
  SEARCH_PROFILES,
} from "./search-profiles.js";

export { SEARCH_PROFILES, DEFAULT_SEARCH_PROFILE_ID };

function clampDepth(depth) {
  return Math.max(1, Math.min(4, Number.parseInt(depth, 10) || 1));
}

function clampBeamWidth(beamWidth) {
  return Math.max(4, Math.min(96, Number.parseInt(beamWidth, 10) || 24));
}

function normalizeSettings(settings = {}) {
  const normalizedProfile = getSearchProfile(settings.searchProfile ?? settings.profileId);
  return {
    depth: clampDepth(settings.depth ?? 3),
    beamWidth: clampBeamWidth(settings.beamWidth ?? 24),
    searchProfile: normalizedProfile.id,
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

const CHAIN_BUILDER_V3_TURN_WEIGHTS = Object.freeze({
  topoutPenalty: -5_000_000,
  singleChainPenalty: -240,
  singleScoreScale: 0.15,
  chainValueBase: 1100,
  chainExponent: 3,
  scoreScale: 0.7,
  allClearBonus: 180,
});

const TURN_RESULT_PROFILE_WEIGHTS = Object.freeze({
  chain_builder_v3: CHAIN_BUILDER_V3_TURN_WEIGHTS,
  chain_builder_v4: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 1200,
    scoreScale: 0.76,
    singleChainPenalty: -300,
    singleScoreScale: 0.1,
  }),
  chain_builder_v5: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 850,
    chainExponent: 3.35,
    scoreScale: 0.82,
    singleChainPenalty: -340,
    singleScoreScale: 0.08,
  }),
  chain_builder_v6: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 1050,
    chainExponent: 3.1,
    scoreScale: 0.8,
    singleChainPenalty: -310,
    singleScoreScale: 0.09,
  }),
  chain_builder_v7: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 950,
    chainExponent: 3.2,
    scoreScale: 0.82,
    singleChainPenalty: -320,
    singleScoreScale: 0.085,
  }),
  chain_builder_v7a: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 860,
    chainExponent: 3.28,
    scoreScale: 0.84,
    singleChainPenalty: -360,
    singleScoreScale: 0.07,
    midChainPenalty: -55_000,
    tenPlusBonus: 135_000,
    elevenPlusBonus: 100_000,
    twelvePlusBonus: 140_000,
  }),
  chain_builder_v8: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 820,
    chainExponent: 3.32,
    scoreScale: 0.86,
    singleChainPenalty: -16_000,
    singleScoreScale: 0.04,
    smallChainPenaltyStep: -24_000,
    midChainPenalty: -75_000,
    sevenChainPenalty: -30_000,
    eightChainPenalty: -45_000,
    nineChainPenalty: -15_000,
    tenPlusBonus: 170_000,
    elevenPlusBonus: 130_000,
    twelvePlusBonus: 180_000,
  }),
  chain_builder_v9: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 780,
    chainExponent: 3.35,
    scoreScale: 0.88,
    singleChainPenalty: -22_000,
    singleScoreScale: 0.03,
    smallChainPenaltyStep: -32_000,
    midChainPenalty: -110_000,
    sevenChainPenalty: -45_000,
    eightChainPenalty: -70_000,
    nineChainPenalty: -45_000,
    tenPlusBonus: 120_000,
    elevenPlusBonus: 270_000,
    twelvePlusBonus: 320_000,
  }),
  chain_builder_v9a: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 740,
    chainExponent: 3.4,
    scoreScale: 0.9,
    singleChainPenalty: -26_000,
    singleScoreScale: 0.025,
    smallChainPenaltyStep: -45_000,
    midChainPenalty: -125_000,
    sevenChainPenalty: -65_000,
    eightChainPenalty: -100_000,
    nineChainPenalty: -120_000,
    tenPlusBonus: 30_000,
    elevenPlusBonus: 450_000,
    twelvePlusBonus: 540_000,
  }),
});

function scoreTurnResult(result, profileId = DEFAULT_SEARCH_PROFILE_ID) {
  const weights =
    TURN_RESULT_PROFILE_WEIGHTS[profileId] ??
    TURN_RESULT_PROFILE_WEIGHTS[DEFAULT_SEARCH_PROFILE_ID];
  if (result.topout) {
    return weights.topoutPenalty;
  }

  if (result.totalChains === 0) {
    return 0;
  }

  if (result.totalChains === 1) {
    return weights.singleChainPenalty + result.totalScore * weights.singleScoreScale;
  }

  const chainValue =
    weights.chainValueBase * result.totalChains ** weights.chainExponent;
  const scoreValue = result.totalScore * weights.scoreScale;
  const allClearBonus = result.allClear ? weights.allClearBonus : 0;
  const smallChainPenalty =
    result.totalChains >= 2 && result.totalChains <= 6
      ? (weights.smallChainPenaltyStep ?? 0) * (7 - result.totalChains)
      : 0;
  const midChainPenalty =
    result.totalChains >= 7 && result.totalChains <= 9
      ? weights.midChainPenalty ?? 0
      : 0;
  const sevenChainPenalty =
    result.totalChains === 7 ? weights.sevenChainPenalty ?? 0 : 0;
  const eightChainPenalty =
    result.totalChains === 8 ? weights.eightChainPenalty ?? 0 : 0;
  const nineChainPenalty =
    result.totalChains === 9 ? weights.nineChainPenalty ?? 0 : 0;
  const tenPlusBonus = result.totalChains >= 10 ? weights.tenPlusBonus ?? 0 : 0;
  const elevenPlusBonus =
    result.totalChains >= 11 ? weights.elevenPlusBonus ?? 0 : 0;
  const twelvePlusBonus =
    result.totalChains >= 12 ? weights.twelvePlusBonus ?? 0 : 0;
  return (
    chainValue +
    scoreValue +
    allClearBonus +
    smallChainPenalty +
    midChainPenalty +
    sevenChainPenalty +
    eightChainPenalty +
    nineChainPenalty +
    tenPlusBonus +
    elevenPlusBonus +
    twelvePlusBonus
  );
}

function createExpandedNode(node, pair, action, layerIndex, profileId) {
  const result = resolveTurn(node.board, pair, action);
  const features = extractBoardFeatures(result.finalBoard, {
    includeVirtualChains: false,
  });
  const heuristicScore = scoreBoardFeatures(features, profileId);
  const turnValue = scoreTurnResult(result, profileId);
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
    profileId,
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
  const heuristicScore = scoreBoardFeatures(refinedFeatures, node.profileId);
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
  const profile = getSearchProfile(normalizedSettings.searchProfile);
  const candidatePools = new Map();
  const rootActions = enumerateLegalActions(board, currentPair);

  let expandedNodeCount = 0;
  let frontier = rootActions.map((action) => {
    const node = createExpandedNode(
      {
        board,
        profileId: profile.id,
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
      profile.id,
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
        const child = createExpandedNode(node, pair, action, depthIndex, profile.id);
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
    objective: profile.id,
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
