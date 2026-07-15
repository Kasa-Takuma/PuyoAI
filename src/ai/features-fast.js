// Fast-path feature extraction mirroring src/ai/features.js exactly, but
// operating on the column-major fast-board representation (src/core/
// fast-board.js) to avoid the array-of-arrays overhead in search hot paths
// (in particular the 16-pair x 11-action virtual chain probe). See
// tests/features-fast.test.js for the exhaustive equivalence tests against
// the legacy implementation.
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  ORIENTATIONS,
  PLAYABLE_COLORS,
  STORAGE_HEIGHT,
  VISIBLE_HEIGHT,
} from "../core/constants.js";
import {
  FAST_COLORS,
  fastBoardKey,
  fastColumnHeights,
  fastEnumerateLegalActions,
  fastResolveTurn,
  pairToCodes,
} from "../core/fast-board.js";

const CELL_COUNT = BOARD_WIDTH * BOARD_HEIGHT;

const FAST_NEIGHBOR_OFFSETS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

const FAST_FEATURE_CACHE = new Map();
const MAX_FAST_FEATURE_CACHE_SIZE = 4096;
const SAFE_VISIBLE_HEIGHT = VISIBLE_HEIGHT - 2;

const PROBE_PAIR_CODES = PLAYABLE_COLORS.flatMap((axis) =>
  PLAYABLE_COLORS.map((child) => pairToCodes({ axis, child })),
);

// Reused scratch buffers for the flood-fill group detection, mirroring the
// approach in fast-board.js's findMatchedGroups.
const groupVisited = new Uint8Array(CELL_COUNT);
const groupStack = new Int16Array(CELL_COUNT);

function isFeatureColorCode(code) {
  return code !== FAST_COLORS.EMPTY && code !== FAST_COLORS.GARBAGE;
}

function countOccupiedCellsFast(fastBoard) {
  let count = 0;

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < STORAGE_HEIGHT; y += 1) {
      if (fastBoard[base + y] !== FAST_COLORS.EMPTY) {
        count += 1;
      }
    }
  }

  return count;
}

function collectReachableLibertiesFast(fastBoard, heights, cells) {
  const liberties = new Set();
  const surfaceLiberties = new Set();

  for (const index of cells) {
    const x = (index / BOARD_HEIGHT) | 0;
    const y = index % BOARD_HEIGHT;

    for (const [dx, dy] of FAST_NEIGHBOR_OFFSETS) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= VISIBLE_HEIGHT) {
        continue;
      }

      const neighborIndex = nx * BOARD_HEIGHT + ny;
      if (fastBoard[neighborIndex] !== FAST_COLORS.EMPTY) {
        continue;
      }

      liberties.add(neighborIndex);
      if (ny === heights[nx]) {
        surfaceLiberties.add(neighborIndex);
      }
    }
  }

  return {
    libertyCount: liberties.size,
    surfaceLibertyCount: surfaceLiberties.size,
  };
}

function collectGroupsFast(fastBoard) {
  groupVisited.fill(0);
  const groups = [];

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < VISIBLE_HEIGHT; y += 1) {
      const index = base + y;
      const color = fastBoard[index];
      if (!isFeatureColorCode(color) || groupVisited[index]) {
        continue;
      }

      let stackSize = 0;
      groupStack[stackSize] = index;
      stackSize += 1;
      groupVisited[index] = 1;
      const cells = [index];

      while (stackSize > 0) {
        stackSize -= 1;
        const current = groupStack[stackSize];
        const cx = (current / BOARD_HEIGHT) | 0;
        const cy = current % BOARD_HEIGHT;

        if (cx + 1 < BOARD_WIDTH) {
          const neighbor = current + BOARD_HEIGHT;
          if (!groupVisited[neighbor] && fastBoard[neighbor] === color) {
            groupVisited[neighbor] = 1;
            groupStack[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
        if (cx - 1 >= 0) {
          const neighbor = current - BOARD_HEIGHT;
          if (!groupVisited[neighbor] && fastBoard[neighbor] === color) {
            groupVisited[neighbor] = 1;
            groupStack[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
        if (cy + 1 < VISIBLE_HEIGHT) {
          const neighbor = current + 1;
          if (!groupVisited[neighbor] && fastBoard[neighbor] === color) {
            groupVisited[neighbor] = 1;
            groupStack[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
        if (cy - 1 >= 0) {
          const neighbor = current - 1;
          if (!groupVisited[neighbor] && fastBoard[neighbor] === color) {
            groupVisited[neighbor] = 1;
            groupStack[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
      }

      groups.push({ color, cells });
    }
  }

  return groups;
}

function analyzeVirtualPairChainsFast(fastBoard) {
  const chainResults = [];

  for (const pair of PROBE_PAIR_CODES) {
    const actions = fastEnumerateLegalActions(fastBoard, pair.axis, pair.child).filter(
      (action) =>
        action.orientation === ORIENTATIONS.UP ||
        action.orientation === ORIENTATIONS.RIGHT,
    );
    for (const action of actions) {
      const result = fastResolveTurn(fastBoard, pair.axis, pair.child, action);
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

export function extractBoardFeaturesFast(
  fastBoard,
  { includeVirtualChains = true } = {},
) {
  const key = `${includeVirtualChains ? "full" : "base"}:${fastBoardKey(fastBoard)}`;
  const cached = FAST_FEATURE_CACHE.get(key);
  if (cached) {
    return cached;
  }

  const heights = fastColumnHeights(fastBoard);
  const groups = collectGroupsFast(fastBoard);
  const stackCells = countOccupiedCellsFast(fastBoard);
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
    const { libertyCount, surfaceLibertyCount } = collectReachableLibertiesFast(
      fastBoard,
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
    ? analyzeVirtualPairChainsFast(fastBoard)
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

  if (FAST_FEATURE_CACHE.size >= MAX_FAST_FEATURE_CACHE_SIZE) {
    FAST_FEATURE_CACHE.clear();
  }
  FAST_FEATURE_CACHE.set(key, features);
  return features;
}
