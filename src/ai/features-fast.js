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
  BB_WORDS,
  bbDilate,
  bbPopcount,
  buildColorMasks,
  COLOR_MASKS,
  FAST_COLORS,
  fastBoardHash,
  fastColumnHeights,
  fastEnumerateLegalActions,
  fastResolveTurn,
  LANE_BITS,
  pairToCodes,
  popcount32,
  VISIBLE_MASK,
} from "../core/fast-board.js";

const CELL_COUNT = BOARD_WIDTH * BOARD_HEIGHT;

const FAST_FEATURE_CACHE = new Map();
const MAX_FAST_FEATURE_CACHE_SIZE = 4096;
const SAFE_VISIBLE_HEIGHT = VISIBLE_HEIGHT - 2;

const PROBE_PAIR_CODES = PLAYABLE_COLORS.flatMap((axis) =>
  PLAYABLE_COLORS.map((child) => pairToCodes({ axis, child })),
);

// Reused scratch buffers for the bitboard flood-fill group detection below,
// mirroring the approach in fast-board.js's findMatchedGroups. Kept as
// features-fast.js's own buffers (distinct from fast-board.js's private
// scratch arrays) so nothing here depends on fast-board.js's internal call
// sequencing beyond buildColorMasks/COLOR_MASKS.
const colorVisibleScratch = new Uint32Array(BB_WORDS);
const residualScratch = new Uint32Array(BB_WORDS);
const seedScratch = new Uint32Array(BB_WORDS);
const compScratch = new Uint32Array(BB_WORDS);
const nextScratch = new Uint32Array(BB_WORDS);
const dilateScratch = new Uint32Array(BB_WORDS);
const emptyMaskScratch = new Uint32Array(BB_WORDS);
const surfaceMaskScratch = new Uint32Array(BB_WORDS);

function fastBoardsEqual(a, b) {
  for (let i = 0; i < CELL_COUNT; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
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

// Sets bit (x, heights[x]) for every column whose surface sits within the
// visible board, matching the `ny === heights[nx]` check in the legacy
// collectReachableLiberties. Columns already taller than the visible board
// contribute no surface liberties, since no visible row can equal their
// height.
function buildSurfaceMask(heights, out) {
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const height = heights[x];
    if (height < VISIBLE_HEIGHT) {
      const word = x >> 1;
      const laneShift = (x & 1) * LANE_BITS;
      out[word] |= 1 << (laneShift + height);
    }
  }
}

// Bitboard equivalent of collectGroups: for every playable color, extract
// all visible-board connected components (any size, not just size >= 4 --
// unlike fast-board.js's findMatchedGroups, which only cares about erasable
// groups). Also captures the visible empty-cell mask alongside the groups,
// since both are derived from the same buildColorMasks call and the caller
// needs it for liberty computation.
function collectGroupsFast(fastBoard) {
  buildColorMasks(fastBoard);

  const rawEmptyMask = COLOR_MASKS[FAST_COLORS.EMPTY];
  emptyMaskScratch[0] = rawEmptyMask[0] & VISIBLE_MASK;
  emptyMaskScratch[1] = rawEmptyMask[1] & VISIBLE_MASK;
  emptyMaskScratch[2] = rawEmptyMask[2] & VISIBLE_MASK;

  const groups = [];

  for (let color = FAST_COLORS.RED; color <= FAST_COLORS.YELLOW; color += 1) {
    const colorMask = COLOR_MASKS[color];
    colorVisibleScratch[0] = colorMask[0] & VISIBLE_MASK;
    colorVisibleScratch[1] = colorMask[1] & VISIBLE_MASK;
    colorVisibleScratch[2] = colorMask[2] & VISIBLE_MASK;
    residualScratch[0] = colorVisibleScratch[0];
    residualScratch[1] = colorVisibleScratch[1];
    residualScratch[2] = colorVisibleScratch[2];

    while (
      residualScratch[0] !== 0 ||
      residualScratch[1] !== 0 ||
      residualScratch[2] !== 0
    ) {
      let seedWord = 0;
      while (residualScratch[seedWord] === 0) {
        seedWord += 1;
      }
      seedScratch[0] = 0;
      seedScratch[1] = 0;
      seedScratch[2] = 0;
      const w = residualScratch[seedWord];
      seedScratch[seedWord] = w & -w;

      compScratch[0] = seedScratch[0];
      compScratch[1] = seedScratch[1];
      compScratch[2] = seedScratch[2];
      for (;;) {
        bbDilate(nextScratch, compScratch);
        nextScratch[0] &= colorVisibleScratch[0];
        nextScratch[1] &= colorVisibleScratch[1];
        nextScratch[2] &= colorVisibleScratch[2];
        if (
          nextScratch[0] === compScratch[0] &&
          nextScratch[1] === compScratch[1] &&
          nextScratch[2] === compScratch[2]
        ) {
          break;
        }
        compScratch[0] = nextScratch[0];
        compScratch[1] = nextScratch[1];
        compScratch[2] = nextScratch[2];
      }

      groups.push({
        colorCode: color,
        size: bbPopcount(compScratch),
        mask: Uint32Array.of(compScratch[0], compScratch[1], compScratch[2]),
      });

      residualScratch[0] &= ~compScratch[0];
      residualScratch[1] &= ~compScratch[1];
      residualScratch[2] &= ~compScratch[2];
    }
  }

  return { groups, emptyMask: emptyMaskScratch };
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
  const key = fastBoardHash(fastBoard) * 2 + (includeVirtualChains ? 1 : 0);
  const cached = FAST_FEATURE_CACHE.get(key);
  if (cached && fastBoardsEqual(cached.board, fastBoard)) {
    return cached.features;
  }

  const heights = fastColumnHeights(fastBoard);
  const { groups, emptyMask } = collectGroupsFast(fastBoard);
  buildSurfaceMask(heights, surfaceMaskScratch);

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

  let adjacency = 0;
  let group2Count = 0;
  let group3Count = 0;
  let extendableGroup2Count = 0;
  let surfaceExtendableGroup2Count = 0;
  let readyGroup3Count = 0;
  let surfaceReadyGroup3Count = 0;
  let isolatedSingles = 0;
  const colorCounts = new Map();

  for (const group of groups) {
    const { size } = group;
    adjacency += Math.max(0, size - 1);
    colorCounts.set(group.colorCode, (colorCounts.get(group.colorCode) ?? 0) + size);

    if (size === 1) {
      isolatedSingles += 1;
      continue;
    }
    if (size !== 2 && size !== 3) {
      continue;
    }

    bbDilate(dilateScratch, group.mask);
    const n0 = dilateScratch[0] & emptyMask[0];
    const n1 = dilateScratch[1] & emptyMask[1];
    const n2 = dilateScratch[2] & emptyMask[2];
    const libertyCount = popcount32(n0) + popcount32(n1) + popcount32(n2);
    const surfaceLibertyCount =
      popcount32(n0 & surfaceMaskScratch[0]) +
      popcount32(n1 & surfaceMaskScratch[1]) +
      popcount32(n2 & surfaceMaskScratch[2]);

    if (size === 2) {
      group2Count += 1;
      if (libertyCount >= 2) {
        extendableGroup2Count += 1;
      }
      if (surfaceLibertyCount >= 1) {
        surfaceExtendableGroup2Count += 1;
      }
    } else {
      group3Count += 1;
      if (libertyCount >= 1) {
        readyGroup3Count += 1;
      }
      if (surfaceLibertyCount >= 1) {
        surfaceReadyGroup3Count += 1;
      }
    }
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
  FAST_FEATURE_CACHE.set(key, { board: fastBoard.slice(), features });
  return features;
}
