import {
  BOARD_WIDTH,
  COLORS,
  ORIENTATIONS,
  PLAYABLE_COLORS,
  STORAGE_HEIGHT,
  VISIBLE_HEIGHT,
} from "../core/constants.js";
import { enumerateLegalActions } from "../core/board.js";
import { resolveTurn } from "../core/engine.js";
import { DEFAULT_SEARCH_PROFILE_ID } from "./search-profiles.js";

const NEIGHBOR_OFFSETS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

const FEATURE_CACHE = new Map();
const MAX_FEATURE_CACHE_SIZE = 4096;
const SAFE_VISIBLE_HEIGHT = VISIBLE_HEIGHT - 2;

const PROBE_PAIRS = PLAYABLE_COLORS.flatMap((axis) =>
  PLAYABLE_COLORS.map((child) => ({ axis, child })),
);

function isFeatureColor(color) {
  return color !== COLORS.EMPTY && color !== COLORS.GARBAGE;
}

export const FEATURE_KEYS = [
  "stackCells",
  "occupiedCells",
  "heightSum",
  "maxHeight",
  "hiddenCells",
  "dangerCells",
  "surfaceRoughness",
  "staircaseLinks",
  "steepWalls",
  "valleyPenalty",
  "adjacency",
  "group2Count",
  "group3Count",
  "extendableGroup2Count",
  "surfaceExtendableGroup2Count",
  "readyGroup3Count",
  "surfaceReadyGroup3Count",
  "isolatedSingles",
  "colorBalance",
  "columnsUsed",
  "bestVirtualChain",
  "bestVirtualScore",
  "virtualChainCount",
  "virtualChainCount2Plus",
  "virtualChainCount3Plus",
  "topVirtualChainSum",
  "topVirtualScoreSum",
];

function boardKey(board) {
  return board.map((row) => row.join("")).join("/");
}

function countOccupiedCells(board) {
  let count = 0;

  for (let y = 0; y < STORAGE_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      if (board[y][x] !== COLORS.EMPTY) {
        count += 1;
      }
    }
  }

  return count;
}

function collectReachableLiberties(board, heights, cells) {
  const liberties = new Set();
  const surfaceLiberties = new Set();

  for (const cell of cells) {
    for (const [dx, dy] of NEIGHBOR_OFFSETS) {
      const nx = cell.x + dx;
      const ny = cell.y + dy;
      if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= VISIBLE_HEIGHT) {
        continue;
      }
      if (board[ny][nx] !== COLORS.EMPTY) {
        continue;
      }

      const key = `${nx}:${ny}`;
      liberties.add(key);
      if (ny === heights[nx]) {
        surfaceLiberties.add(key);
      }
    }
  }

  return {
    libertyCount: liberties.size,
    surfaceLibertyCount: surfaceLiberties.size,
  };
}

function collectGroups(board) {
  const visited = Array.from({ length: VISIBLE_HEIGHT }, () =>
    Array.from({ length: BOARD_WIDTH }, () => false),
  );
  const groups = [];

  for (let y = 0; y < VISIBLE_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const color = board[y][x];
      if (!isFeatureColor(color) || visited[y][x]) {
        continue;
      }

      const stack = [{ x, y }];
      const cells = [];
      visited[y][x] = true;

      while (stack.length > 0) {
        const current = stack.pop();
        cells.push(current);

        for (const [dx, dy] of NEIGHBOR_OFFSETS) {
          const nx = current.x + dx;
          const ny = current.y + dy;
          if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= VISIBLE_HEIGHT) {
            continue;
          }
          if (visited[ny][nx] || board[ny][nx] !== color) {
            continue;
          }
          visited[ny][nx] = true;
          stack.push({ x: nx, y: ny });
        }
      }

      groups.push({ color, cells });
    }
  }

  return groups;
}

function getColumnHeights(board) {
  const heights = [];

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    let height = 0;
    for (let y = STORAGE_HEIGHT - 1; y >= 0; y -= 1) {
      if (board[y][x] !== COLORS.EMPTY) {
        height = y + 1;
        break;
      }
    }
    heights.push(height);
  }

  return heights;
}

function analyzeVirtualPairChains(board) {
  const chainResults = [];

  for (const pair of PROBE_PAIRS) {
    const actions = enumerateLegalActions(board, pair).filter(
      (action) =>
        action.orientation === ORIENTATIONS.UP ||
        action.orientation === ORIENTATIONS.RIGHT,
    );
    for (const action of actions) {
      const result = resolveTurn(board, pair, action);
      if (result.topout || result.totalChains === 0) {
        continue;
      }
      chainResults.push({
        chains: result.totalChains,
        score: result.totalScore,
      });
    }
  }

  chainResults.sort((left, right) => {
    if (right.chains !== left.chains) {
      return right.chains - left.chains;
    }
    return right.score - left.score;
  });

  const topResults = chainResults.slice(0, 3);

  return {
    bestVirtualChain: chainResults[0]?.chains ?? 0,
    bestVirtualScore: chainResults[0]?.score ?? 0,
    virtualChainCount: chainResults.length,
    virtualChainCount2Plus: chainResults.filter((entry) => entry.chains >= 2).length,
    virtualChainCount3Plus: chainResults.filter((entry) => entry.chains >= 3).length,
    topVirtualChainSum: topResults.reduce((sum, entry) => sum + entry.chains, 0),
    topVirtualScoreSum: topResults.reduce((sum, entry) => sum + entry.score, 0),
  };
}

export function extractBoardFeatures(board, { includeVirtualChains = true } = {}) {
  const key = `${includeVirtualChains ? "full" : "base"}:${boardKey(board)}`;
  const cached = FEATURE_CACHE.get(key);
  if (cached) {
    return cached;
  }

  const heights = getColumnHeights(board);
  const groups = collectGroups(board);
  const stackCells = countOccupiedCells(board);
  const hiddenCells = heights.reduce(
    (sum, height) => sum + Math.max(0, height - VISIBLE_HEIGHT),
    0,
  );
  const dangerCells = heights.reduce(
    (sum, height) => sum + Math.max(0, height - SAFE_VISIBLE_HEIGHT),
    0,
  );
  const surfaceRoughness = heights
    .slice(1)
    .reduce((sum, height, index) => sum + Math.abs(height - heights[index]), 0);
  const staircaseLinks = heights.slice(1).reduce((sum, height, index) => {
    const diff = Math.abs(height - heights[index]);
    if (diff === 1) {
      return sum + 2;
    }
    if (diff === 2) {
      return sum + 1;
    }
    return sum;
  }, 0);
  const steepWalls = heights.slice(1).reduce((sum, height, index) => {
    return sum + Math.max(0, Math.abs(height - heights[index]) - 2);
  }, 0);
  const valleyPenalty = heights.reduce((sum, height, index) => {
    if (index === 0 || index === heights.length - 1) {
      return sum;
    }
    const neighborFloor = Math.min(heights[index - 1], heights[index + 1]);
    return sum + Math.max(0, neighborFloor - height - 1);
  }, 0);
  const maxHeight = Math.max(...heights);
  const columnsUsed = heights.filter((height) => height > 0).length;
  const adjacency = groups.reduce(
    (sum, group) => sum + Math.max(0, group.cells.length - 1),
    0,
  );

  let group2Count = 0;
  let group3Count = 0;
  let extendableGroup2Count = 0;
  let surfaceExtendableGroup2Count = 0;
  let readyGroup3Count = 0;
  let surfaceReadyGroup3Count = 0;
  let isolatedSingles = 0;

  for (const group of groups) {
    const { libertyCount, surfaceLibertyCount } = collectReachableLiberties(
      board,
      heights,
      group.cells,
    );

    if (group.cells.length === 1) {
      isolatedSingles += 1;
    }
    if (group.cells.length === 2) {
      group2Count += 1;
      if (libertyCount >= 2) {
        extendableGroup2Count += 1;
      }
      if (surfaceLibertyCount >= 1) {
        surfaceExtendableGroup2Count += 1;
      }
    }
    if (group.cells.length === 3) {
      group3Count += 1;
      if (libertyCount >= 1) {
        readyGroup3Count += 1;
      }
      if (surfaceLibertyCount >= 1) {
        surfaceReadyGroup3Count += 1;
      }
    }
  }

  const colorCounts = new Map();
  for (const group of groups) {
    colorCounts.set(
      group.color,
      (colorCounts.get(group.color) ?? 0) + group.cells.length,
    );
  }
  const countValues = [...colorCounts.values()];
  const colorBalance =
    countValues.length <= 1 || stackCells === 0
      ? 0
      : 1 - (Math.max(...countValues) - Math.min(...countValues)) / stackCells;

  const shouldAnalyzeVirtualChains =
    stackCells >= 6 &&
    (group3Count > 0 || surfaceExtendableGroup2Count >= 2 || maxHeight >= 4);

  const virtualChains =
    includeVirtualChains && shouldAnalyzeVirtualChains
    ? analyzeVirtualPairChains(board)
    : {
        bestVirtualChain: 0,
        bestVirtualScore: 0,
        virtualChainCount: 0,
        virtualChainCount2Plus: 0,
        virtualChainCount3Plus: 0,
        topVirtualChainSum: 0,
        topVirtualScoreSum: 0,
      };

  const features = {
    stackCells,
    occupiedCells: stackCells,
    heightSum: heights.reduce((sum, height) => sum + height, 0),
    maxHeight,
    hiddenCells,
    dangerCells,
    surfaceRoughness,
    staircaseLinks,
    steepWalls,
    valleyPenalty,
    adjacency,
    group2Count,
    group3Count,
    extendableGroup2Count,
    surfaceExtendableGroup2Count,
    readyGroup3Count,
    surfaceReadyGroup3Count,
    isolatedSingles,
    colorBalance,
    columnsUsed,
    ...virtualChains,
    columnHeights: heights,
  };

  if (FEATURE_CACHE.size >= MAX_FEATURE_CACHE_SIZE) {
    FEATURE_CACHE.clear();
  }
  FEATURE_CACHE.set(key, features);
  return features;
}

export function featuresToVector(features) {
  return FEATURE_KEYS.map((key) => features[key]);
}

const CHAIN_BUILDER_V3_BOARD_WEIGHTS = Object.freeze({
  bestVirtualChain: 620,
  topVirtualChainSum: 160,
  virtualChainCount2Plus: 110,
  virtualChainCount3Plus: 260,
  bestVirtualScore: 0.42,
  topVirtualScoreSum: 0.05,
  surfaceReadyGroup3Count: 240,
  surfaceExtendableGroup2Count: 120,
  readyGroup3Count: 0,
  extendableGroup2Count: 0,
  group3Count: 80,
  group2Count: 28,
  adjacency: 12,
  staircaseLinks: 20,
  colorBalance: 140,
  stackCells: 12,
  columnsUsed: 14,
  hiddenCells: -5000,
  dangerCells: -220,
  surfaceRoughness: -10,
  steepWalls: -70,
  valleyPenalty: -32,
  isolatedSingles: -38,
});

const BOARD_PROFILE_WEIGHTS = Object.freeze({
  chain_builder_v3: CHAIN_BUILDER_V3_BOARD_WEIGHTS,
  chain_builder_v4: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 720,
    topVirtualChainSum: 220,
    virtualChainCount3Plus: 320,
    bestVirtualScore: 0.5,
    topVirtualScoreSum: 0.075,
    dangerCells: -185,
    surfaceRoughness: -8,
    steepWalls: -58,
    valleyPenalty: -28,
    isolatedSingles: -28,
  }),
  chain_builder_v5: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 780,
    topVirtualChainSum: 250,
    virtualChainCount2Plus: 90,
    virtualChainCount3Plus: 240,
    bestVirtualScore: 0.58,
    topVirtualScoreSum: 0.095,
    surfaceReadyGroup3Count: 190,
    surfaceExtendableGroup2Count: 105,
    group3Count: 70,
    group2Count: 24,
    dangerCells: -195,
    surfaceRoughness: -9,
    steepWalls: -62,
    valleyPenalty: -30,
    isolatedSingles: -30,
  }),
  chain_builder_v6: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 750,
    topVirtualChainSum: 235,
    virtualChainCount3Plus: 330,
    bestVirtualScore: 0.54,
    topVirtualScoreSum: 0.09,
    surfaceReadyGroup3Count: 230,
    surfaceExtendableGroup2Count: 118,
    group3Count: 78,
    group2Count: 27,
    dangerCells: -190,
    surfaceRoughness: -8,
    steepWalls: -60,
    valleyPenalty: -29,
    isolatedSingles: -30,
  }),
  chain_builder_v7: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 765,
    topVirtualChainSum: 245,
    virtualChainCount3Plus: 305,
    bestVirtualScore: 0.55,
    topVirtualScoreSum: 0.095,
    surfaceReadyGroup3Count: 225,
    surfaceExtendableGroup2Count: 116,
    group3Count: 78,
    group2Count: 27,
    dangerCells: -195,
    surfaceRoughness: -9,
    steepWalls: -63,
    valleyPenalty: -30,
    isolatedSingles: -31,
  }),
  chain_builder_v7a: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 820,
    topVirtualChainSum: 275,
    virtualChainCount2Plus: 90,
    virtualChainCount3Plus: 290,
    bestVirtualScore: 0.6,
    topVirtualScoreSum: 0.12,
    surfaceReadyGroup3Count: 220,
    surfaceExtendableGroup2Count: 108,
    group3Count: 74,
    group2Count: 24,
    dangerCells: -220,
    surfaceRoughness: -11,
    steepWalls: -75,
    valleyPenalty: -35,
    isolatedSingles: -34,
  }),
  chain_builder_v8: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 850,
    topVirtualChainSum: 305,
    virtualChainCount2Plus: 70,
    virtualChainCount3Plus: 250,
    bestVirtualScore: 0.64,
    topVirtualScoreSum: 0.14,
    surfaceReadyGroup3Count: 210,
    surfaceExtendableGroup2Count: 100,
    group3Count: 70,
    group2Count: 22,
    dangerCells: -235,
    surfaceRoughness: -12,
    steepWalls: -82,
    valleyPenalty: -38,
    isolatedSingles: -36,
  }),
  chain_builder_v9: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 900,
    topVirtualChainSum: 335,
    virtualChainCount2Plus: 60,
    virtualChainCount3Plus: 220,
    bestVirtualScore: 0.7,
    topVirtualScoreSum: 0.17,
    surfaceReadyGroup3Count: 195,
    surfaceExtendableGroup2Count: 92,
    group3Count: 64,
    group2Count: 19,
    dangerCells: -255,
    surfaceRoughness: -13,
    steepWalls: -88,
    valleyPenalty: -40,
    isolatedSingles: -38,
  }),
  chain_builder_v9a: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 930,
    topVirtualChainSum: 360,
    virtualChainCount2Plus: 55,
    virtualChainCount3Plus: 205,
    bestVirtualScore: 0.76,
    topVirtualScoreSum: 0.2,
    surfaceReadyGroup3Count: 185,
    surfaceExtendableGroup2Count: 86,
    group3Count: 60,
    group2Count: 18,
    dangerCells: -270,
    surfaceRoughness: -14,
    steepWalls: -92,
    valleyPenalty: -42,
    isolatedSingles: -40,
  }),
  chain_builder_v9b: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 910,
    topVirtualChainSum: 345,
    virtualChainCount2Plus: 58,
    virtualChainCount3Plus: 215,
    bestVirtualScore: 0.72,
    topVirtualScoreSum: 0.18,
    surfaceReadyGroup3Count: 190,
    surfaceExtendableGroup2Count: 89,
    group3Count: 62,
    group2Count: 18,
    dangerCells: -260,
    surfaceRoughness: -13,
    steepWalls: -90,
    valleyPenalty: -41,
    isolatedSingles: -39,
  }),
  chain_builder_v10: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 880,
    topVirtualChainSum: 325,
    virtualChainCount2Plus: 66,
    virtualChainCount3Plus: 238,
    bestVirtualScore: 0.68,
    topVirtualScoreSum: 0.16,
    surfaceReadyGroup3Count: 202,
    surfaceExtendableGroup2Count: 96,
    group3Count: 67,
    group2Count: 20,
    dangerCells: -248,
    surfaceRoughness: -12,
    steepWalls: -86,
    valleyPenalty: -39,
    isolatedSingles: -37,
  }),
  chain_builder_v11: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 958,
    topVirtualChainSum: 332,
    virtualChainCount2Plus: 58,
    virtualChainCount3Plus: 200,
    bestVirtualScore: 0.70125,
    topVirtualScoreSum: 0.16504,
    surfaceReadyGroup3Count: 204,
    surfaceExtendableGroup2Count: 82,
    group3Count: 62,
    group2Count: 18,
    dangerCells: -263,
    surfaceRoughness: -14,
    steepWalls: -81,
    valleyPenalty: -41,
    isolatedSingles: -39,
  }),
  chain_builder_v12: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 942,
    topVirtualChainSum: 403,
    virtualChainCount2Plus: 58,
    virtualChainCount3Plus: 189,
    bestVirtualScore: 0.73346,
    topVirtualScoreSum: 0.12942,
    surfaceReadyGroup3Count: 198,
    surfaceExtendableGroup2Count: 70,
    group3Count: 62,
    group2Count: 18,
    dangerCells: -221,
    surfaceRoughness: -14,
    steepWalls: -77,
    valleyPenalty: -41,
    isolatedSingles: -39,
  }),
  chain_builder_v12_ac: Object.freeze({
    ...CHAIN_BUILDER_V3_BOARD_WEIGHTS,
    bestVirtualChain: 942,
    topVirtualChainSum: 403,
    virtualChainCount2Plus: 58,
    virtualChainCount3Plus: 189,
    bestVirtualScore: 0.73346,
    topVirtualScoreSum: 0.12942,
    surfaceReadyGroup3Count: 198,
    surfaceExtendableGroup2Count: 70,
    group3Count: 62,
    group2Count: 18,
    dangerCells: -221,
    surfaceRoughness: -14,
    steepWalls: -77,
    valleyPenalty: -41,
    isolatedSingles: -39,
  }),
});

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

export function getBoardProfileWeights(profileId = DEFAULT_SEARCH_PROFILE_ID) {
  const weights =
    BOARD_PROFILE_WEIGHTS[profileId] ??
    BOARD_PROFILE_WEIGHTS[DEFAULT_SEARCH_PROFILE_ID];
  return { ...weights };
}

function bonusScale(profileConfig, key) {
  const value = profileConfig?.bonusScales?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 1;
}

export function scoreBoardFeatures(
  features,
  profileId = DEFAULT_SEARCH_PROFILE_ID,
  profileConfig = null,
) {
  const virtualChainCount2Plus = Math.min(features.virtualChainCount2Plus, 6);
  const virtualChainCount3Plus = Math.min(features.virtualChainCount3Plus, 3);
  const effectiveProfileId = profileConfig?.baseProfileId ?? profileId;
  const weights = {
    ...getBoardProfileWeights(effectiveProfileId),
    ...(profileConfig?.boardWeights ?? {}),
  };
  const v4PlusLargeChainBonus =
    effectiveProfileId === "chain_builder_v4" ||
    effectiveProfileId === "chain_builder_v5" ||
    effectiveProfileId === "chain_builder_v6" ||
    effectiveProfileId === "chain_builder_v7" ||
    effectiveProfileId === "chain_builder_v7a" ||
    effectiveProfileId === "chain_builder_v8" ||
    effectiveProfileId === "chain_builder_v9" ||
    effectiveProfileId === "chain_builder_v9a" ||
    effectiveProfileId === "chain_builder_v9b" ||
    effectiveProfileId === "chain_builder_v10" ||
    effectiveProfileId === "chain_builder_v11" ||
    effectiveProfileId === "chain_builder_v12" ||
    effectiveProfileId === "chain_builder_v12_ac"
      ? bonusScale(profileConfig, "largeChain") *
        (Math.max(0, features.bestVirtualChain - 5) ** 3 * 460 +
          Math.max(0, features.topVirtualChainSum - 15) * 2400 +
          (features.bestVirtualChain >= 10 ? 90_000 : 0) -
          Math.max(0, features.maxHeight - 9) *
            Math.max(0, 6 - features.bestVirtualChain) *
            1400)
      : 0;
  const v5TenPlusBonus =
    effectiveProfileId === "chain_builder_v5"
      ? Math.max(0, features.bestVirtualChain - 8) ** 3 * 2100 +
        Math.max(0, features.topVirtualChainSum - 24) * 5200 +
        Math.max(0, features.topVirtualScoreSum - 100_000) * 0.22 +
        (features.bestVirtualChain >= 10 ? 85_000 : 0) -
        Math.max(0, 10 - features.bestVirtualChain) *
          (Math.max(0, features.maxHeight - 10) * 2400 +
            features.dangerCells * 900 +
          Math.max(0, features.steepWalls - 6) * 1000)
      : 0;
  const v6TenPlusBonus =
    effectiveProfileId === "chain_builder_v6"
      ? Math.max(0, features.bestVirtualChain - 8) ** 3 * 850 +
        Math.max(0, features.topVirtualChainSum - 24) * 2200 +
        Math.max(0, features.topVirtualScoreSum - 100_000) * 0.08 +
        (features.bestVirtualChain >= 10 ? 40_000 : 0) -
        Math.max(0, features.maxHeight - 10) *
          Math.max(0, 7 - features.bestVirtualChain) *
          900
      : 0;
  const v7StretchBonus =
    effectiveProfileId === "chain_builder_v7"
      ? Math.max(0, features.bestVirtualChain - 8) ** 3 * 700 +
        Math.max(0, features.bestVirtualChain - 10) ** 3 * 900 +
        Math.max(0, features.topVirtualChainSum - 24) * 1800 +
        Math.max(0, features.topVirtualChainSum - 28) * 1800 +
        Math.max(0, features.topVirtualScoreSum - 115_000) * 0.1 +
        (features.bestVirtualChain >= 10 ? 35_000 : 0) +
        (features.bestVirtualChain >= 11 ? 45_000 : 0) +
        (features.bestVirtualChain >= 10 && features.stackCells >= 50
          ? Math.min(features.stackCells - 49, 10) * 1200
          : 0) -
        Math.max(0, features.surfaceRoughness - 18) * 600 -
        Math.max(0, features.steepWalls - 10) * 900 -
        Math.max(0, features.maxHeight - 12) * 3000
      : 0;
  const v7aStableFrequencyBonus =
    effectiveProfileId === "chain_builder_v7a"
      ? Math.max(0, features.bestVirtualChain - 7) ** 3 * 1250 +
        Math.max(0, features.bestVirtualChain - 9) ** 3 * 2400 +
        Math.max(0, features.topVirtualChainSum - 22) * 3600 +
        Math.max(0, features.topVirtualChainSum - 28) * 5200 +
        Math.max(0, features.topVirtualScoreSum - 85_000) * 0.2 +
        Math.max(0, features.topVirtualScoreSum - 130_000) * 0.18 +
        Math.min(features.virtualChainCount3Plus, 14) *
          (features.bestVirtualChain >= 8 ? 2600 : 600) +
        (features.bestVirtualChain >= 10 ? 125_000 : 0) +
        (features.bestVirtualChain >= 11 ? 95_000 : 0) +
        (features.bestVirtualChain >= 10 && features.stackCells >= 48
          ? Math.min(features.stackCells - 47, 12) * 2600
          : 0) -
        Math.max(0, features.stackCells - 50) *
          Math.max(0, 10 - features.bestVirtualChain) *
          9000 -
        Math.max(0, features.maxHeight - 10) *
          Math.max(0, 9 - features.bestVirtualChain) *
          4200 -
        Math.max(0, features.dangerCells - 2) *
          Math.max(0, 10 - features.bestVirtualChain) *
          2600 -
        Math.max(0, features.surfaceRoughness - 17) * 1200 -
        Math.max(0, features.steepWalls - 9) * 1800 -
        Math.max(0, features.hiddenCells) * 10_000
      : 0;
  const v8AntiSmallFireBonus =
    effectiveProfileId === "chain_builder_v8"
      ? Math.max(0, features.bestVirtualChain - 7) ** 3 * 1000 +
        Math.max(0, features.bestVirtualChain - 9) ** 3 * 3200 +
        Math.max(0, features.topVirtualChainSum - 23) * 3200 +
        Math.max(0, features.topVirtualChainSum - 28) * 6800 +
        Math.max(0, features.topVirtualScoreSum - 95_000) * 0.24 +
        Math.max(0, features.topVirtualScoreSum - 145_000) * 0.22 +
        Math.min(features.virtualChainCount3Plus, 12) *
          (features.bestVirtualChain >= 9 ? 2400 : 250) +
        (features.bestVirtualChain >= 10 ? 170_000 : 0) +
        (features.bestVirtualChain >= 11 ? 125_000 : 0) +
        (features.bestVirtualChain >= 10 && features.stackCells >= 48
          ? Math.min(features.stackCells - 47, 12) * 3200
          : 0) -
        Math.max(0, features.stackCells - 48) *
          Math.max(0, 10 - features.bestVirtualChain) *
          11_500 -
        Math.max(0, features.maxHeight - 10) *
          Math.max(0, 9 - features.bestVirtualChain) *
          5000 -
        Math.max(0, features.dangerCells - 2) *
          Math.max(0, 10 - features.bestVirtualChain) *
          3200 -
        Math.max(0, features.surfaceRoughness - 16) * 1400 -
        Math.max(0, features.steepWalls - 9) * 2200 -
        Math.max(0, features.hiddenCells) * 12_000
      : 0;
  const v9StableLargeChainBonus =
    effectiveProfileId === "chain_builder_v9"
      ? Math.max(0, features.bestVirtualChain - 8) ** 3 * 1400 +
        Math.max(0, features.bestVirtualChain - 10) ** 3 * 5200 +
        Math.max(0, features.topVirtualChainSum - 25) * 2600 +
        Math.max(0, features.topVirtualChainSum - 29) * 8200 +
        Math.max(0, features.topVirtualScoreSum - 115_000) * 0.2 +
        Math.max(0, features.topVirtualScoreSum - 165_000) * 0.3 +
        Math.min(features.virtualChainCount3Plus, 10) *
          (features.bestVirtualChain >= 10 ? 2200 : 0) +
        (features.bestVirtualChain >= 11 ? 420_000 : 0) +
        (features.bestVirtualChain >= 12 ? 360_000 : 0) +
        (features.bestVirtualChain >= 11 && features.stackCells >= 52
          ? Math.min(features.stackCells - 51, 10) * 3600
          : 0) -
        (features.bestVirtualChain >= 7 && features.bestVirtualChain <= 9
          ? Math.max(0, 10 - features.bestVirtualChain) *
            Math.max(0, features.stackCells - 36) *
            3500
          : 0) -
        Math.max(0, features.bestVirtualChain - 9) *
          Math.max(0, 10 - features.topVirtualChainSum) *
          12_000 -
        Math.max(0, features.stackCells - 50) *
          Math.max(0, 11 - features.bestVirtualChain) *
          13_500 -
        Math.max(0, features.maxHeight - 11) *
          Math.max(0, 10 - features.bestVirtualChain) *
          6000 -
        Math.max(0, features.dangerCells - 3) *
          Math.max(0, 11 - features.bestVirtualChain) *
          3600 -
        Math.max(0, features.surfaceRoughness - 16) * 1600 -
        Math.max(0, features.steepWalls - 9) * 2400 -
        Math.max(0, features.hiddenCells) * 14_000
      : 0;
  const v9aFocusElevenPlusBonus =
    effectiveProfileId === "chain_builder_v9a"
      ? Math.max(0, features.bestVirtualChain - 8) ** 3 * 1600 +
        Math.max(0, features.bestVirtualChain - 10) ** 3 * 7200 +
        Math.max(0, features.topVirtualChainSum - 25) * 2400 +
        Math.max(0, features.topVirtualChainSum - 29) * 10_500 +
        Math.max(0, features.topVirtualScoreSum - 105_000) * 0.18 +
        Math.max(0, features.topVirtualScoreSum - 145_000) * 0.42 +
        Math.max(0, features.bestVirtualScore - 50_000) * 0.62 +
        (features.bestVirtualChain >= 11 ? 520_000 : 0) +
        (features.bestVirtualChain >= 12 ? 520_000 : 0) +
        (features.bestVirtualChain >= 10 && features.stackCells >= 52
          ? Math.min(features.stackCells - 51, 9) * 5200
          : 0) +
        (features.bestVirtualChain >= 11 && features.maxHeight >= 12
          ? Math.min(features.maxHeight - 11, 3) * 28_000
          : 0) -
        (features.bestVirtualChain >= 7 && features.bestVirtualChain <= 9
          ? Math.max(0, 10 - features.bestVirtualChain) *
            Math.max(0, features.stackCells - 34) *
            5200
          : 0) -
        (features.bestVirtualChain === 10
          ? Math.max(0, 29 - features.topVirtualChainSum) * 24_000 +
            Math.max(0, 145_000 - features.topVirtualScoreSum) * 0.18
          : 0) -
        Math.max(0, features.stackCells - 52) *
          Math.max(0, 11 - features.bestVirtualChain) *
          15_000 -
        Math.max(0, features.maxHeight - 11) *
          Math.max(0, 10 - features.bestVirtualChain) *
          7000 -
        Math.max(0, features.dangerCells - 4) *
          Math.max(0, 11 - features.bestVirtualChain) *
          4200 -
        Math.max(0, features.surfaceRoughness - 16) * 1700 -
        Math.max(0, features.steepWalls - 9) * 2600 -
        Math.max(0, features.hiddenCells) * 16_000
      : 0;
  const v9bBalancedElevenPlusBonus =
    effectiveProfileId === "chain_builder_v9b" ||
    effectiveProfileId === "chain_builder_v11" ||
    effectiveProfileId === "chain_builder_v12" ||
    effectiveProfileId === "chain_builder_v12_ac"
      ? bonusScale(profileConfig, "v9b") *
        (Math.max(0, features.bestVirtualChain - 8) ** 3 * 1450 +
          Math.max(0, features.bestVirtualChain - 10) ** 3 * 5800 +
          Math.max(0, features.topVirtualChainSum - 25) * 2500 +
          Math.max(0, features.topVirtualChainSum - 29) * 8800 +
          Math.max(0, features.topVirtualScoreSum - 115_000) * 0.2 +
          Math.max(0, features.topVirtualScoreSum - 160_000) * 0.34 +
          Math.min(features.virtualChainCount3Plus, 10) *
            (features.bestVirtualChain >= 10 ? 2200 : 0) +
          (features.bestVirtualChain >= 11 ? 460_000 : 0) +
          (features.bestVirtualChain >= 12 ? 400_000 : 0) +
          (features.bestVirtualChain >= 11 && features.stackCells >= 52
            ? Math.min(features.stackCells - 51, 10) * 4200
            : 0) -
          (features.bestVirtualChain >= 7 && features.bestVirtualChain <= 9
            ? Math.max(0, 10 - features.bestVirtualChain) *
              Math.max(0, features.stackCells - 36) *
              4200
            : 0) -
          (features.bestVirtualChain === 10
            ? Math.max(0, 28 - features.topVirtualChainSum) * 14_000 +
              Math.max(0, 135_000 - features.topVirtualScoreSum) * 0.08
            : 0) -
          Math.max(0, features.stackCells - 51) *
            Math.max(0, 11 - features.bestVirtualChain) *
            13_500 -
          Math.max(0, features.maxHeight - 11) *
            Math.max(0, 10 - features.bestVirtualChain) *
            6200 -
          Math.max(0, features.dangerCells - 3) *
            Math.max(0, 11 - features.bestVirtualChain) *
            3700 -
          Math.max(0, features.surfaceRoughness - 16) * 1600 -
          Math.max(0, features.steepWalls - 9) * 2400 -
          Math.max(0, features.hiddenCells) * 14_000)
      : 0;
  const v10Maturity =
    effectiveProfileId === "chain_builder_v10"
      ? clamp01(
          clamp01((features.stackCells - 38) / 18) * 0.42 +
            clamp01((features.bestVirtualChain - 8) / 3) * 0.34 +
            clamp01((features.topVirtualChainSum - 23) / 8) * 0.24,
        )
      : 0;
  const v10EarlyBuildBonus =
    effectiveProfileId === "chain_builder_v10"
      ? Math.max(0, features.bestVirtualChain - 7) ** 3 * 950 +
        Math.max(0, features.bestVirtualChain - 9) ** 3 * 2800 +
        Math.max(0, features.topVirtualChainSum - 23) * 3100 +
        Math.max(0, features.topVirtualChainSum - 28) * 5900 +
        Math.max(0, features.topVirtualScoreSum - 95_000) * 0.23 +
        Math.max(0, features.topVirtualScoreSum - 145_000) * 0.21 +
        Math.min(features.virtualChainCount3Plus, 12) *
          (features.bestVirtualChain >= 9 ? 2400 : 350) +
        (features.bestVirtualChain >= 10 ? 155_000 : 0) +
        (features.bestVirtualChain >= 11 ? 115_000 : 0) +
        (features.bestVirtualChain >= 10 && features.stackCells >= 48
          ? Math.min(features.stackCells - 47, 12) * 3000
          : 0) -
        Math.max(0, features.stackCells - 49) *
          Math.max(0, 10 - features.bestVirtualChain) *
          10_800 -
        Math.max(0, features.maxHeight - 10) *
          Math.max(0, 9 - features.bestVirtualChain) *
          4800 -
        Math.max(0, features.dangerCells - 2) *
          Math.max(0, 10 - features.bestVirtualChain) *
          3100 -
        Math.max(0, features.surfaceRoughness - 16) * 1400 -
        Math.max(0, features.steepWalls - 9) * 2200 -
        Math.max(0, features.hiddenCells) * 12_000
      : 0;
  const v10MatureElevenPlusBonus =
    effectiveProfileId === "chain_builder_v10"
      ? Math.max(0, features.bestVirtualChain - 8) ** 3 * 1500 +
        Math.max(0, features.bestVirtualChain - 10) ** 3 * 6000 +
        Math.max(0, features.topVirtualChainSum - 25) * 2500 +
        Math.max(0, features.topVirtualChainSum - 29) * 9000 +
        Math.max(0, features.topVirtualScoreSum - 115_000) * 0.21 +
        Math.max(0, features.topVirtualScoreSum - 160_000) * 0.35 +
        Math.min(features.virtualChainCount3Plus, 10) *
          (features.bestVirtualChain >= 10 ? 2200 : 0) +
        (features.bestVirtualChain >= 11 ? 470_000 : 0) +
        (features.bestVirtualChain >= 12 ? 410_000 : 0) +
        (features.bestVirtualChain >= 11 && features.stackCells >= 52
          ? Math.min(features.stackCells - 51, 10) * 4300
          : 0) -
        (features.bestVirtualChain >= 7 && features.bestVirtualChain <= 9
          ? Math.max(0, 10 - features.bestVirtualChain) *
            Math.max(0, features.stackCells - 36) *
            4200
          : 0) -
        (features.bestVirtualChain === 10
          ? Math.max(0, 28 - features.topVirtualChainSum) * 15_000 +
            Math.max(0, 135_000 - features.topVirtualScoreSum) * 0.08
          : 0) -
        Math.max(0, features.stackCells - 51) *
          Math.max(0, 11 - features.bestVirtualChain) *
          13_800 -
        Math.max(0, features.maxHeight - 11) *
          Math.max(0, 10 - features.bestVirtualChain) *
          6200 -
        Math.max(0, features.dangerCells - 3) *
          Math.max(0, 11 - features.bestVirtualChain) *
          3700 -
        Math.max(0, features.surfaceRoughness - 16) * 1600 -
        Math.max(0, features.steepWalls - 9) * 2400 -
        Math.max(0, features.hiddenCells) * 14_000
      : 0;
  const v10HybridChainBonus =
    effectiveProfileId === "chain_builder_v10"
      ? v10EarlyBuildBonus * (1 - v10Maturity) +
        v10MatureElevenPlusBonus * v10Maturity
      : 0;

  return (
    features.bestVirtualChain ** 3 * weights.bestVirtualChain +
    features.topVirtualChainSum * weights.topVirtualChainSum +
    virtualChainCount2Plus * weights.virtualChainCount2Plus +
    virtualChainCount3Plus * weights.virtualChainCount3Plus +
    features.bestVirtualScore * weights.bestVirtualScore +
    features.topVirtualScoreSum * weights.topVirtualScoreSum +
    features.surfaceReadyGroup3Count * weights.surfaceReadyGroup3Count +
    features.surfaceExtendableGroup2Count * weights.surfaceExtendableGroup2Count +
    features.readyGroup3Count * weights.readyGroup3Count +
    features.extendableGroup2Count * weights.extendableGroup2Count +
    features.group3Count * weights.group3Count +
    features.group2Count * weights.group2Count +
    features.adjacency * weights.adjacency +
    features.staircaseLinks * weights.staircaseLinks +
    features.colorBalance * weights.colorBalance +
    features.stackCells * weights.stackCells +
    features.columnsUsed * weights.columnsUsed +
    features.hiddenCells * weights.hiddenCells +
    features.dangerCells * weights.dangerCells +
    features.surfaceRoughness * weights.surfaceRoughness +
    features.steepWalls * weights.steepWalls +
    features.valleyPenalty * weights.valleyPenalty +
    features.isolatedSingles * weights.isolatedSingles +
    v4PlusLargeChainBonus +
    v5TenPlusBonus +
    v6TenPlusBonus +
    v7StretchBonus +
    v7aStableFrequencyBonus +
    v8AntiSmallFireBonus +
    v9StableLargeChainBonus +
    v9aFocusElevenPlusBonus +
    v9bBalancedElevenPlusBonus +
    v10HybridChainBonus
  );
}
