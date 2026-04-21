import { encodeAction, enumerateLegalActions } from "../core/board.js";
import { resolveTurn } from "../core/engine.js";
import {
  extractBoardFeatures,
  featuresToVector,
  scoreBoardFeatures,
} from "./features.js";
import { evaluateValueModel, normalizeValueAssistSettings } from "./value.js";
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

function normalizeProfileConfig(profileConfig, fallbackProfileId) {
  if (!profileConfig || typeof profileConfig !== "object") {
    return null;
  }

  const baseProfile = getSearchProfile(profileConfig.baseProfileId ?? fallbackProfileId);
  const id =
    typeof profileConfig.id === "string" && profileConfig.id.length > 0
      ? profileConfig.id
      : `${baseProfile.id}_tuned`;

  return {
    id,
    label:
      typeof profileConfig.label === "string" && profileConfig.label.length > 0
        ? profileConfig.label
        : id,
    baseProfileId: baseProfile.id,
    turnWeights:
      profileConfig.turnWeights && typeof profileConfig.turnWeights === "object"
        ? profileConfig.turnWeights
        : null,
    boardWeights:
      profileConfig.boardWeights && typeof profileConfig.boardWeights === "object"
        ? profileConfig.boardWeights
        : null,
    bonusScales:
      profileConfig.bonusScales && typeof profileConfig.bonusScales === "object"
        ? profileConfig.bonusScales
        : null,
  };
}

function normalizeSettings(settings = {}) {
  const requestedProfileId = settings.searchProfile ?? settings.profileId;
  const profileConfig = normalizeProfileConfig(
    settings.profileConfig,
    requestedProfileId,
  );
  const normalizedProfile = getSearchProfile(
    profileConfig?.baseProfileId ?? requestedProfileId,
  );
  const valueAssist = normalizeValueAssistSettings(settings);

  const normalizedSettings = {
    depth: clampDepth(settings.depth ?? 3),
    beamWidth: clampBeamWidth(settings.beamWidth ?? 24),
    searchProfile: profileConfig?.id ?? normalizedProfile.id,
    useValueModel: valueAssist.useValueModel,
    valueWeight: valueAssist.valueWeight,
  };
  if (profileConfig) {
    normalizedSettings.baseSearchProfile = normalizedProfile.id;
    normalizedSettings.profileConfig = profileConfig;
  }
  return normalizedSettings;
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
  chain_builder_v9b: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 770,
    chainExponent: 3.36,
    scoreScale: 0.89,
    singleChainPenalty: -23_000,
    singleScoreScale: 0.03,
    smallChainPenaltyStep: -34_000,
    midChainPenalty: -115_000,
    sevenChainPenalty: -48_000,
    eightChainPenalty: -74_000,
    nineChainPenalty: -55_000,
    tenPlusBonus: 95_000,
    elevenPlusBonus: 320_000,
    twelvePlusBonus: 380_000,
  }),
  chain_builder_v10: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 800,
    chainExponent: 3.34,
    scoreScale: 0.875,
    singleChainPenalty: -20_000,
    singleScoreScale: 0.035,
    smallChainPenaltyStep: -30_000,
    midChainPenalty: -90_000,
    sevenChainPenalty: -38_000,
    eightChainPenalty: -58_000,
    nineChainPenalty: -35_000,
    tenPlusBonus: 130_000,
    elevenPlusBonus: 245_000,
    twelvePlusBonus: 300_000,
  }),
  chain_builder_v11: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 750,
    chainExponent: 3.30539,
    scoreScale: 0.93973,
    singleChainPenalty: -23_000,
    singleScoreScale: 0.03,
    smallChainPenaltyStep: -35_190,
    midChainPenalty: -128_117,
    sevenChainPenalty: -41_134,
    eightChainPenalty: -71_521,
    nineChainPenalty: -60_178,
    tenPlusBonus: 97_509,
    elevenPlusBonus: 320_358,
    twelvePlusBonus: 349_342,
  }),
  chain_builder_v12: Object.freeze({
    ...CHAIN_BUILDER_V3_TURN_WEIGHTS,
    chainValueBase: 738,
    chainExponent: 3.25039,
    scoreScale: 0.89989,
    singleChainPenalty: -23_000,
    singleScoreScale: 0.03,
    smallChainPenaltyStep: -42_148,
    midChainPenalty: -150_194,
    sevenChainPenalty: -49_504,
    eightChainPenalty: -70_464,
    nineChainPenalty: -53_129,
    tenPlusBonus: 105_948,
    elevenPlusBonus: 202_440,
    twelvePlusBonus: 512_657,
  }),
});

export function getTurnResultProfileWeights(profileId = DEFAULT_SEARCH_PROFILE_ID) {
  const weights =
    TURN_RESULT_PROFILE_WEIGHTS[profileId] ??
    TURN_RESULT_PROFILE_WEIGHTS[DEFAULT_SEARCH_PROFILE_ID];
  return { ...weights };
}

function resolveTurnResultProfileWeights(profileId, profileConfig) {
  const baseProfileId = profileConfig?.baseProfileId ?? profileId;
  return {
    ...getTurnResultProfileWeights(baseProfileId),
    ...(profileConfig?.turnWeights ?? {}),
  };
}

function scoreContextualTurnAdjustment(
  result,
  profileId,
  context = {},
  profileConfig = null,
) {
  const effectiveProfileId = profileConfig?.baseProfileId ?? profileId;
  if (
    effectiveProfileId !== "chain_builder_v10" ||
    result.totalChains < 7 ||
    result.totalChains > 9
  ) {
    return 0;
  }

  const beforeFeatures = context.beforeFeatures;
  if (!beforeFeatures) {
    return 0;
  }

  const isSafeBeforeFire =
    beforeFeatures.hiddenCells === 0 &&
    beforeFeatures.dangerCells <= 2 &&
    beforeFeatures.maxHeight <= 11;
  if (!isSafeBeforeFire) {
    return 0;
  }

  const hasMoreRoomToGrow =
    beforeFeatures.bestVirtualChain >= 8 ||
    beforeFeatures.topVirtualChainSum >= 22 ||
    beforeFeatures.topVirtualScoreSum >= 80_000 ||
    beforeFeatures.stackCells >= 38;
  if (!hasMoreRoomToGrow) {
    return 0;
  }

  const safety = Math.max(0, 12 - beforeFeatures.maxHeight) * 11_000 +
    Math.max(0, 3 - beforeFeatures.dangerCells) * 18_000;
  const potential = Math.max(0, beforeFeatures.bestVirtualChain - 7) * 20_000 +
    Math.max(0, beforeFeatures.topVirtualChainSum - 22) * 5000 +
    Math.max(0, beforeFeatures.topVirtualScoreSum - 90_000) * 0.12;
  const chainFactor = 10 - result.totalChains;
  const penalty = chainFactor * (45_000 + safety + potential);

  return -Math.min(260_000, penalty);
}

function scoreTurnResult(
  result,
  profileId = DEFAULT_SEARCH_PROFILE_ID,
  context = {},
  profileConfig = null,
) {
  const weights = resolveTurnResultProfileWeights(profileId, profileConfig);
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
  const contextualAdjustment = scoreContextualTurnAdjustment(
    result,
    profileId,
    context,
    profileConfig,
  );
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
    twelvePlusBonus +
    contextualAdjustment
  );
}

function createExpandedNode(node, pair, action, layerIndex, profileId, profileConfig) {
  const result = resolveTurn(node.board, pair, action);
  const features = extractBoardFeatures(result.finalBoard, {
    includeVirtualChains: false,
  });
  const heuristicScore = scoreBoardFeatures(features, profileId, profileConfig);
  const beforeFeatures =
    (profileConfig?.baseProfileId ?? profileId) === "chain_builder_v10" &&
    result.totalChains >= 7 &&
    result.totalChains <= 9
      ? extractBoardFeatures(node.board, { includeVirtualChains: true })
      : null;
  const turnValue = scoreTurnResult(
    result,
    profileId,
    { beforeFeatures },
    profileConfig,
  );
  const cumulativeValue = node.cumulativeValue + turnValue;
  const projectedScore = node.projectedScore + result.totalScore;
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
    profileConfig,
    rootAction,
    rootKey,
    rootTurn,
    path,
    cumulativeValue,
    projectedScore,
    searchScore,
    heuristicScore,
    bestDepth: layerIndex + 1,
    expandedNodes: node.expandedNodes + 1,
    lastResult: result,
    lastFeatures: features,
  };
}

function createLeafState({ node, currentPair, nextQueue, turn, totalScore }) {
  const depth = Math.max(1, node.bestDepth);
  const nextCurrentPair = nextQueue[depth - 1] ?? currentPair;
  const nextQueueAfter = nextQueue.slice(depth);

  return {
    board: node.board,
    currentPair: nextCurrentPair,
    nextQueue: nextQueueAfter,
    turn: turn + depth,
    totalScore: totalScore + node.projectedScore,
  };
}

function createCandidate(node, rootContext) {
  const refinedFeatures = extractBoardFeatures(node.board, {
    includeVirtualChains: true,
  });
  const heuristicScore = scoreBoardFeatures(
    refinedFeatures,
    node.profileId,
    node.profileConfig,
  );
  const valuePrediction = rootContext.valueModel
    ? evaluateValueModel({
        model: rootContext.valueModel,
        ...createLeafState({
          node,
          currentPair: rootContext.currentPair,
          nextQueue: rootContext.nextQueue,
          turn: rootContext.turn,
          totalScore: rootContext.totalScore,
        }),
        features: refinedFeatures,
      })
    : null;
  const valueScore = valuePrediction
    ? valuePrediction.objective * rootContext.valueWeight
    : 0;
  const searchScore = node.cumulativeValue + heuristicScore + valueScore;

  return {
    action: cloneAction(node.rootAction),
    actionKey: node.rootKey,
    searchScore,
    cumulativeValue: node.cumulativeValue,
    heuristicScore,
    valueScore,
    valuePrediction,
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

export function searchBestMove({
  board,
  currentPair,
  nextQueue = [],
  settings = {},
  turn = 1,
  totalScore = 0,
  valueModel = null,
}) {
  const startedAt = performance.now();
  const normalizedSettings = normalizeSettings(settings);
  const profileId = normalizedSettings.searchProfile;
  const profileConfig = normalizedSettings.profileConfig;
  const activeValueModel = normalizedSettings.useValueModel ? valueModel : null;
  const candidatePools = new Map();
  const rootActions = enumerateLegalActions(board, currentPair);

  let expandedNodeCount = 0;
  let frontier = rootActions.map((action) => {
    const node = createExpandedNode(
      {
        board,
        profileId,
        profileConfig,
        rootAction: null,
        rootKey: null,
        rootTurn: null,
        path: [],
        cumulativeValue: 0,
        projectedScore: 0,
        expandedNodes: 0,
      },
      currentPair,
      action,
      0,
      profileId,
      profileConfig,
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
        const child = createExpandedNode(
          node,
          pair,
          action,
          depthIndex,
          profileId,
          profileConfig,
        );
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
        .map((node) =>
          createCandidate(node, {
            currentPair,
            nextQueue,
            turn,
            totalScore,
            valueModel: activeValueModel,
            valueWeight: normalizedSettings.valueWeight,
          }),
        )
        .sort((left, right) => right.searchScore - left.searchScore)[0],
    )
    .sort(
    (left, right) => right.searchScore - left.searchScore,
  );
  const bestAction = candidates[0]?.action ?? cloneAction(rootActions[0]);
  const bestScore = candidates[0]?.searchScore ?? -Infinity;
  const elapsedMs = performance.now() - startedAt;

  return {
    kind: activeValueModel ? "search_value_assisted" : "search",
    objective: profileId,
    settings: normalizedSettings,
    valueAssist: activeValueModel
      ? {
          modelName: activeValueModel.name ?? "value_mlp",
          targetHorizon: activeValueModel.targetHorizon ?? 48,
          weight: normalizedSettings.valueWeight,
        }
      : null,
    bestAction,
    bestActionKey: bestAction ? encodeAction(bestAction) : null,
    bestScore,
    candidates,
    expandedNodeCount,
    candidateCount: candidates.length,
    elapsedMs,
  };
}
