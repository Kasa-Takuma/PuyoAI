import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLORS,
  ORIENTATIONS,
  PLAYABLE_COLORS,
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

  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
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
      if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_HEIGHT) {
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
  const visited = Array.from({ length: BOARD_HEIGHT }, () =>
    Array.from({ length: BOARD_WIDTH }, () => false),
  );
  const groups = [];

  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const color = board[y][x];
      if (color === COLORS.EMPTY || visited[y][x]) {
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
          if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_HEIGHT) {
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
    for (let y = BOARD_HEIGHT - 1; y >= 0; y -= 1) {
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
});

export function scoreBoardFeatures(features, profileId = DEFAULT_SEARCH_PROFILE_ID) {
  const virtualChainCount2Plus = Math.min(features.virtualChainCount2Plus, 6);
  const virtualChainCount3Plus = Math.min(features.virtualChainCount3Plus, 3);
  const weights = BOARD_PROFILE_WEIGHTS[profileId] ?? BOARD_PROFILE_WEIGHTS[DEFAULT_SEARCH_PROFILE_ID];
  const v4PlusLargeChainBonus =
    profileId === "chain_builder_v4" || profileId === "chain_builder_v5"
      ? Math.max(0, features.bestVirtualChain - 5) ** 3 * 460 +
        Math.max(0, features.topVirtualChainSum - 15) * 2400 +
        (features.bestVirtualChain >= 10 ? 90_000 : 0) -
        Math.max(0, features.maxHeight - 9) *
          Math.max(0, 6 - features.bestVirtualChain) *
          1400
      : 0;
  const v5TenPlusBonus =
    profileId === "chain_builder_v5"
      ? Math.max(0, features.bestVirtualChain - 8) ** 3 * 2100 +
        Math.max(0, features.topVirtualChainSum - 24) * 5200 +
        Math.max(0, features.topVirtualScoreSum - 100_000) * 0.22 +
        (features.bestVirtualChain >= 10 ? 85_000 : 0) -
        Math.max(0, 10 - features.bestVirtualChain) *
          (Math.max(0, features.maxHeight - 10) * 2400 +
            features.dangerCells * 900 +
            Math.max(0, features.steepWalls - 6) * 1000)
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
    v5TenPlusBonus
  );
}
