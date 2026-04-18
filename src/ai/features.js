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
  "adjacency",
  "group2Count",
  "group3Count",
  "largeGroupCount",
  "isolatedSingles",
  "colorBalance",
  "centerPreference",
  "columnsUsed",
];

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
  const maxHeight = Math.max(...heights);
  const columnsUsed = heights.filter((height) => height > 0).length;
  const adjacency = groups.reduce((sum, group) => sum + Math.max(0, group.cells.length - 1), 0);
  const group2Count = groups.filter((group) => group.cells.length === 2).length;
  const group3Count = groups.filter((group) => group.cells.length === 3).length;
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
    adjacency,
    group2Count,
    group3Count,
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
    features.group3Count * 180 +
    features.group2Count * 54 +
    features.adjacency * 10 +
    features.colorBalance * 90 +
    features.centerPreference * 5 -
    features.maxHeight * 44 -
    features.hiddenCells * 2600 -
    features.surfaceRoughness * 16 -
    features.isolatedSingles * 18 -
    features.heightSum * 3 +
    features.columnsUsed * 8
  );
}
