import { ACTION_INDEX } from "./action-vocab.js";
import { extractBoardFeaturesFast } from "./features-fast.js";
import { evaluateSoloValueFast } from "./solo-value.js";
import { encodeAction } from "../core/board.js";
import {
  fastBoardHash,
  fastColumnHeights,
  fastEnumerateLegalActions,
  fastResolveTurn,
  fromLegacyBoard,
  pairToCodes,
} from "../core/fast-board.js";

export const SOLO_SEARCH_DEFAULTS = Object.freeze({
  depth: 3,
  beamWidth: 22,
  discount: 0.99,
  preserveRootActions: true,
  valueAtLeafOnly: false,
  dedupe: true,
});

// A learned value alone is too smooth to identify the narrow trigger shapes
// that lead to a large chain.  Keep the learned value as the long-horizon
// term, but add a small structural prior at each leaf.  The prior is only a
// search-time tie breaker; training targets and the reported chain utility
// remain the fixed real-event reward.
const SOLO_STRUCTURE_SCALE = 1000;
const SOLO_MODEL_VALUE_WEIGHT = 30;
const SOLO_MODEL_DEATH_WEIGHT = 30;

// A placement reward is kept separate from the board prior.  The board prior
// answers "is this position promising?" while this term answers "was this
// placement a premature small fire?".  These values are scaled down with the
// structural score so the new search can combine both signals without using
// the legacy search implementation at runtime.
const SOLO_TURN_WEIGHTS = Object.freeze({
  chainValueBase: 825,
  chainExponent: 3.10028,
  scoreScale: 0.90442,
  singleChainPenalty: -23_000,
  singleScoreScale: 0.03,
  smallChainPenaltyStep: -71_545,
  midChainPenalty: -150_121,
  sevenChainPenalty: -46_388,
  eightChainPenalty: -69_074,
  nineChainPenalty: -64_412,
  tenPlusBonus: 95_631,
  elevenPlusBonus: 203_425,
  twelvePlusBonus: 478_153,
});

function soloTurnPrior(result) {
  if (result.topout) return -5_000;
  if (result.totalChains === 0) return result.allClear ? 0.18 : 0;
  if (result.totalChains === 1) {
    return (
      SOLO_TURN_WEIGHTS.singleChainPenalty +
      result.totalScore * SOLO_TURN_WEIGHTS.singleScoreScale
    ) / SOLO_STRUCTURE_SCALE;
  }
  const chains = result.totalChains;
  const chainValue =
    SOLO_TURN_WEIGHTS.chainValueBase * chains ** SOLO_TURN_WEIGHTS.chainExponent;
  const smallChainPenalty =
    chains >= 2 && chains <= 6
      ? SOLO_TURN_WEIGHTS.smallChainPenaltyStep * (7 - chains)
      : 0;
  const midChainPenalty =
    chains >= 7 && chains <= 9 ? SOLO_TURN_WEIGHTS.midChainPenalty : 0;
  const sevenChainPenalty = chains === 7 ? SOLO_TURN_WEIGHTS.sevenChainPenalty : 0;
  const eightChainPenalty = chains === 8 ? SOLO_TURN_WEIGHTS.eightChainPenalty : 0;
  const nineChainPenalty = chains === 9 ? SOLO_TURN_WEIGHTS.nineChainPenalty : 0;
  const tenPlusBonus = chains >= 10 ? SOLO_TURN_WEIGHTS.tenPlusBonus : 0;
  const elevenPlusBonus = chains >= 11 ? SOLO_TURN_WEIGHTS.elevenPlusBonus : 0;
  const twelvePlusBonus = chains >= 12 ? SOLO_TURN_WEIGHTS.twelvePlusBonus : 0;
  return (
    chainValue +
    result.totalScore * SOLO_TURN_WEIGHTS.scoreScale +
    smallChainPenalty +
    midChainPenalty +
    sevenChainPenalty +
    eightChainPenalty +
    nineChainPenalty +
    tenPlusBonus +
    elevenPlusBonus +
    twelvePlusBonus
  ) / SOLO_STRUCTURE_SCALE;
}

const SOLO_STRUCTURE_WEIGHTS = Object.freeze({
  bestVirtualChain: 1051,
  topVirtualChainSum: 376,
  virtualChainCount2Plus: 58,
  virtualChainCount3Plus: 215,
  bestVirtualScore: 0.48661,
  topVirtualScoreSum: 0.13754,
  surfaceReadyGroup3Count: 209,
  surfaceExtendableGroup2Count: 64,
  group3Count: 62,
  group2Count: 18,
  adjacency: 12,
  staircaseLinks: 20,
  colorBalance: 140,
  stackCells: 12,
  columnsUsed: 14,
  hiddenCells: -5000,
  dangerCells: -241,
  surfaceRoughness: -15,
  steepWalls: -67,
  valleyPenalty: -41,
  isolatedSingles: -39,
});

function scoreSoloStructureFeatures(features) {
  const virtualChainCount2Plus = Math.min(features.virtualChainCount2Plus, 6);
  const virtualChainCount3Plus = Math.min(features.virtualChainCount3Plus, 3);
  const weightedFeatures = {
    ...features,
    virtualChainCount2Plus,
    virtualChainCount3Plus,
  };
  const base = Object.entries(SOLO_STRUCTURE_WEIGHTS).reduce(
    (sum, [key, weight]) =>
      sum +
      (key === "bestVirtualChain"
        ? (weightedFeatures[key] ?? 0) ** 3
        : weightedFeatures[key] ?? 0) *
        weight,
    0,
  );
  const best = features.bestVirtualChain;
  const topSum = features.topVirtualChainSum;
  const largeChainPrior =
    Math.max(0, best - 5) ** 3 * 460 +
    Math.max(0, topSum - 15) * 2400 +
    (best >= 10 ? 90_000 : 0) -
    Math.max(0, features.maxHeight - 9) * Math.max(0, 6 - best) * 1400;
  const matureChainPrior =
    Math.max(0, best - 8) ** 3 * 1450 +
    Math.max(0, best - 10) ** 3 * 5800 +
    Math.max(0, topSum - 25) * 2500 +
    Math.max(0, topSum - 29) * 8800 +
    Math.max(0, features.topVirtualScoreSum - 115_000) * 0.2 +
    Math.max(0, features.topVirtualScoreSum - 160_000) * 0.34 +
    Math.min(features.virtualChainCount3Plus, 10) *
      (best >= 10 ? 2200 : 0) +
    (best >= 11 ? 460_000 : 0) +
    (best >= 12 ? 400_000 : 0) +
    (best >= 11 && features.stackCells >= 52
      ? Math.min(features.stackCells - 51, 10) * 4200
      : 0) -
    (best >= 7 && best <= 9
      ? Math.max(0, 10 - best) * Math.max(0, features.stackCells - 36) * 4200
      : 0) -
    (best === 10
      ? Math.max(0, 28 - topSum) * 14_000 +
        Math.max(0, 135_000 - features.topVirtualScoreSum) * 0.08
      : 0) -
    Math.max(0, features.stackCells - 51) *
      Math.max(0, 11 - best) *
      13_500 -
    Math.max(0, features.maxHeight - 11) * Math.max(0, 10 - best) * 6200 -
    Math.max(0, features.dangerCells - 3) * Math.max(0, 11 - best) * 3700 -
    Math.max(0, features.surfaceRoughness - 16) * 1600 -
    Math.max(0, features.steepWalls - 9) * 2400 -
    Math.max(0, features.hiddenCells) * 14_000;
  return base + largeChainPrior + matureChainPrior;
}

function soloStructureValue(board, includeVirtualChains) {
  return (
    scoreSoloStructureFeatures(
      extractBoardFeaturesFast(board, { includeVirtualChains }),
    ) / SOLO_STRUCTURE_SCALE
  );
}

export function soloChainUtility(chains) {
  if (chains < 10) return 0;
  if (chains === 10) return 1;
  if (chains === 11) return 1.25;
  if (chains === 12) return 1.5;
  if (chains === 13) return 1.75;
  return 2;
}

function normalizeSettings(settings = {}) {
  const depth = Number.parseInt(settings.depth, 10);
  const beamWidth = Number.parseInt(settings.beamWidth, 10);
  return {
    depth: Math.max(1, Math.min(3, Number.isFinite(depth) ? depth : 3)),
    beamWidth: Math.max(
      22,
      Math.min(
        64,
        Number.isFinite(beamWidth) ? beamWidth : SOLO_SEARCH_DEFAULTS.beamWidth,
      ),
    ),
    discount: SOLO_SEARCH_DEFAULTS.discount,
    preserveRootActions: settings.preserveRootActions !== false,
    valueAtLeafOnly: settings.valueAtLeafOnly === true,
    dedupe: settings.dedupe !== false,
  };
}

function cloneAction(action) {
  return { column: action.column, orientation: action.orientation };
}

function actionId(action) {
  return ACTION_INDEX[encodeAction(action)] ?? Number.MAX_SAFE_INTEGER;
}

function compareNodes(left, right) {
  const scoreDifference = right.searchScore - left.searchScore;
  if (scoreDifference !== 0) return scoreDifference;
  const rootDifference = left.rootActionId - right.rootActionId;
  if (rootDifference !== 0) return rootDifference;
  const pathLength = Math.min(left.pathIds.length, right.pathIds.length);
  for (let index = 0; index < pathLength; index += 1) {
    if (left.pathIds[index] !== right.pathIds[index]) {
      return left.pathIds[index] - right.pathIds[index];
    }
  }
  return left.pathIds.length - right.pathIds.length;
}

function boardsEqual(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function simpleIntermediateValue(board) {
  const heights = fastColumnHeights(board);
  let cells = 0;
  let roughness = 0;
  let danger = 0;
  for (let x = 0; x < heights.length; x += 1) {
    cells += heights[x];
    if (x > 0) roughness += Math.abs(heights[x] - heights[x - 1]);
    danger += Math.max(0, heights[x] - 9);
  }
  return cells * 0.002 - roughness * 0.01 - danger * 0.08;
}

function remainingNext(nextQueue, depth) {
  return nextQueue.slice(Math.max(0, depth - 1), 2);
}

function nodeValue(node, nextQueue, settings, model, isFinalDepth) {
  if (node.topout) {
    return {
      searchScore: node.cumulativeReward + node.cumulativePrior,
      prediction: null,
    };
  }
  const useModel = isFinalDepth || !settings.valueAtLeafOnly;
  const prediction = useModel
    ? evaluateSoloValueFast({
        model,
        fastBoard: node.board,
        nextQueue: remainingNext(nextQueue, node.depth),
      })
    : null;
  const future = prediction?.value ?? simpleIntermediateValue(node.board);
  const structural = soloStructureValue(node.board, isFinalDepth);
  const learned = prediction
    ? prediction.value * SOLO_MODEL_VALUE_WEIGHT -
      prediction.deathProbability * SOLO_MODEL_DEATH_WEIGHT
    : 0;
  return {
    searchScore:
      node.cumulativeReward +
      node.cumulativePrior +
      structural +
      settings.discount ** node.depth *
        (prediction ? learned : future),
    prediction,
  };
}

function makeChild(node, pairCodes, action, depth, nextQueue, settings, model, isFinalDepth) {
  const result = fastResolveTurn(node.board, pairCodes.axis, pairCodes.child, action);
  const reward = result.topout ? -4 : soloChainUtility(result.totalChains);
  const actionKey = encodeAction(action);
  const id = ACTION_INDEX[actionKey] ?? Number.MAX_SAFE_INTEGER;
  const child = {
    board: result.board,
    depth,
    rootAction: node.rootAction ?? cloneAction(action),
    rootActionKey: node.rootActionKey ?? actionKey,
    rootActionId: node.rootActionId ?? id,
    rootResult: node.rootResult ?? result,
    path: [...node.path, cloneAction(action)],
    pathIds: [...node.pathIds, id],
    turnPrior: soloTurnPrior(result),
    cumulativeReward:
      node.cumulativeReward + settings.discount ** (depth - 1) * reward,
    cumulativePrior:
      node.cumulativePrior + settings.discount ** (depth - 1) * soloTurnPrior(result),
    topout: result.topout,
    lastResult: result,
    searchScore: -Infinity,
    prediction: null,
  };
  const scored = nodeValue(
    child,
    nextQueue,
    settings,
    model,
    isFinalDepth,
  );
  child.searchScore = scored.searchScore;
  child.prediction = scored.prediction;
  return child;
}

function dedupeNodes(nodes) {
  const buckets = new Map();
  for (const node of nodes) {
    const key = `${node.rootActionId}:${fastBoardHash(node.board)}`;
    const bucket = buckets.get(key) ?? [];
    const matchingIndex = bucket.findIndex((existing) =>
      boardsEqual(existing.board, node.board),
    );
    if (matchingIndex < 0) {
      bucket.push(node);
      buckets.set(key, bucket);
    } else if (compareNodes(node, bucket[matchingIndex]) < 0) {
      bucket[matchingIndex] = node;
    }
  }
  return [...buckets.values()].flat();
}

function selectBeam(nodes, beamWidth, preserveRootActions) {
  const sorted = [...nodes].sort(compareNodes);
  if (!preserveRootActions || sorted.length <= beamWidth) {
    return sorted.slice(0, beamWidth);
  }

  const selected = [];
  const selectedNodes = new Set();
  const seenRoots = new Set();
  for (const node of sorted) {
    if (seenRoots.has(node.rootActionKey)) continue;
    seenRoots.add(node.rootActionKey);
    selected.push(node);
    selectedNodes.add(node);
  }
  for (const node of sorted) {
    if (selected.length >= beamWidth) break;
    if (!selectedNodes.has(node)) selected.push(node);
  }
  return selected.sort(compareNodes).slice(0, beamWidth);
}

function finalCandidates(frontier) {
  const bestByRoot = new Map();
  for (const node of frontier) {
    const scored = {
      ...node,
      searchScore: node.searchScore,
    };
    const existing = bestByRoot.get(node.rootActionKey);
    if (!existing || compareNodes(scored, existing) < 0) {
      bestByRoot.set(node.rootActionKey, scored);
    }
  }
  return [...bestByRoot.values()].sort(compareNodes).map((node) => ({
    action: cloneAction(node.rootAction),
    actionKey: node.rootActionKey,
    searchScore: node.searchScore,
    cumulativeReward: node.cumulativeReward,
    predictedValue: node.prediction?.value ?? 0,
    deathProbability: node.prediction?.deathProbability ?? (node.topout ? 1 : 0),
    immediateChains: node.rootResult.totalChains,
    immediateTopout: node.rootResult.topout,
    bestDepth: node.depth,
    line: node.path.map(cloneAction),
  }));
}

export function searchSoloMove({
  board,
  currentPair,
  nextQueue = [],
  settings = {},
  model,
}) {
  if (!model) throw new Error("searchSoloMove requires a solo value model.");
  const startedAt = performance.now();
  const normalized = normalizeSettings(settings);
  const pairs = [currentPair, ...nextQueue.slice(0, 2)];
  const effectiveDepth = Math.min(normalized.depth, pairs.length);
  const rootBoard = fromLegacyBoard(board);
  let frontier = [
    {
      board: rootBoard,
      depth: 0,
      rootAction: null,
      rootActionKey: null,
      rootActionId: null,
      rootResult: null,
      path: [],
      pathIds: [],
      cumulativeReward: 0,
      cumulativePrior: 0,
      topout: false,
    },
  ];
  let expandedNodeCount = 0;

  for (let depthIndex = 0; depthIndex < effectiveDepth; depthIndex += 1) {
    const pairCodes = pairToCodes(pairs[depthIndex]);
    const isFinalDepth = depthIndex + 1 === effectiveDepth;
    const expanded = [];
    for (const node of frontier) {
      if (node.topout) continue;
      const actions = fastEnumerateLegalActions(
        node.board,
        pairCodes.axis,
        pairCodes.child,
      );
      for (const action of actions) {
        expanded.push(
          makeChild(
            node,
            pairCodes,
            action,
            depthIndex + 1,
            nextQueue,
            normalized,
            model,
            isFinalDepth,
          ),
        );
        expandedNodeCount += 1;
      }
    }
    if (expanded.length === 0) break;
    const survivors = normalized.dedupe ? dedupeNodes(expanded) : expanded;
    const living = survivors.filter((node) => !node.topout);
    frontier = selectBeam(
      living.length > 0 ? living : survivors,
      normalized.beamWidth,
      normalized.preserveRootActions,
    );
  }

  const candidates = finalCandidates(frontier);
  const best = candidates[0] ?? null;
  return {
    kind: "solo-search",
    objective: "discounted_large_chain_utility_128",
    settings: { ...normalized, depth: effectiveDepth },
    modelName: model.name ?? "solo_value",
    modelHash: model.sha256 ?? null,
    bestAction: best?.action ?? null,
    bestActionKey: best?.actionKey ?? null,
    bestScore: best?.searchScore ?? -Infinity,
    candidates,
    expandedNodeCount,
    elapsedMs: performance.now() - startedAt,
  };
}
