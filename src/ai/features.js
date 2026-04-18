import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLORS,
  VISIBLE_HEIGHT,
} from "../core/constants.js";

const NEIGHBOR_OFFSETS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

export const FEATURE_KEYS = [
  "occupiedCells",
  "heightSum",
  "maxHeight",
  "hiddenCells",
  "surfaceRoughness",
  "staircaseScore",
  "valleyDepth",
  "adjacency",
  "group2Count",
  "group3Count",
  "extendableGroup2Count",
  "readyGroup3Count",
  "largeGroupCount",
  "isolatedSingles",
  "colorBalance",
  "centerPreference",
  "columnsUsed",
];

function collectGroupLiberties(board, cells) {
  const liberties = new Set();

  for (const cell of cells) {
    for (const [dx, dy] of NEIGHBOR_OFFSETS) {
      const nx = cell.x + dx;
      const ny = cell.y + dy;
      if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_HEIGHT) {
        continue;
      }
      if (board[ny][nx] === COLORS.EMPTY) {
        liberties.add(`${nx}:${ny}`);
      }
    }
  }

  return liberties.size;
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

export function extractBoardFeatures(board) {
  const heights = getColumnHeights(board);
  const groups = collectGroups(board);
  const occupiedCells = heights.reduce((sum, height) => sum + height, 0);
  const hiddenCells = heights.reduce(
    (sum, height) => sum + Math.max(0, height - VISIBLE_HEIGHT),
    0,
  );
  const surfaceRoughness = heights
    .slice(1)
    .reduce((sum, height, index) => sum + Math.abs(height - heights[index]), 0);
  const staircaseScore = heights.slice(1).reduce((sum, height, index) => {
    const diff = Math.abs(height - heights[index]);
    if (diff === 1) {
      return sum + 3;
    }
    if (diff === 2) {
      return sum + 1;
    }
    if (diff >= 4) {
      return sum - (diff - 3);
    }
    return sum;
  }, 0);
  const valleyDepth = heights.reduce((sum, height, index) => {
    if (index === 0 || index === heights.length - 1) {
      return sum;
    }
    const neighborFloor = Math.min(heights[index - 1], heights[index + 1]);
    return sum + Math.max(0, neighborFloor - height);
  }, 0);
  const maxHeight = Math.max(...heights);
  const columnsUsed = heights.filter((height) => height > 0).length;
  const adjacency = groups.reduce((sum, group) => sum + Math.max(0, group.cells.length - 1), 0);
  const group2Count = groups.filter((group) => group.cells.length === 2).length;
  const group3Count = groups.filter((group) => group.cells.length === 3).length;
  const extendableGroup2Count = groups.filter((group) => {
    return group.cells.length === 2 && collectGroupLiberties(board, group.cells) >= 2;
  }).length;
  const readyGroup3Count = groups.filter((group) => {
    return group.cells.length === 3 && collectGroupLiberties(board, group.cells) >= 1;
  }).length;
  const largeGroupCount = groups.filter((group) => group.cells.length >= 4).length;
  const isolatedSingles = groups.filter((group) => group.cells.length === 1).length;
  const colorCounts = new Map();

  for (const group of groups) {
    colorCounts.set(
      group.color,
      (colorCounts.get(group.color) ?? 0) + group.cells.length,
    );
  }

  const countValues = [...colorCounts.values()];
  const colorBalance =
    countValues.length <= 1
      ? 0
      : 1 - (Math.max(...countValues) - Math.min(...countValues)) / occupiedCells;

  const centerPreference = groups.reduce((sum, group) => {
    return (
      sum +
      group.cells.reduce((groupSum, cell) => {
        const columnBias = 2.5 - Math.abs(cell.x - 2.5);
        return groupSum + columnBias;
      }, 0)
    );
  }, 0);

  return {
    occupiedCells,
    heightSum: heights.reduce((sum, height) => sum + height, 0),
    maxHeight,
    hiddenCells,
    surfaceRoughness,
    staircaseScore,
    valleyDepth,
    adjacency,
    group2Count,
    group3Count,
    extendableGroup2Count,
    readyGroup3Count,
    largeGroupCount,
    isolatedSingles,
    colorBalance,
    centerPreference,
    columnsUsed,
    columnHeights: heights,
  };
}

export function featuresToVector(features) {
  return FEATURE_KEYS.map((key) => features[key]);
}

export function scoreBoardFeatures(features) {
  return (
    features.readyGroup3Count * 260 +
    features.extendableGroup2Count * 110 +
    features.group3Count * 120 +
    features.group2Count * 36 +
    features.staircaseScore * 24 +
    features.valleyDepth * 32 +
    features.adjacency * 8 +
    features.colorBalance * 72 +
    features.centerPreference * 4 -
    features.maxHeight * 56 -
    features.hiddenCells * 3000 -
    features.surfaceRoughness * 8 -
    features.isolatedSingles * 24 -
    features.heightSum * 4 +
    features.columnsUsed * 6
  );
}
