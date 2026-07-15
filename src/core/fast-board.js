// High-performance column-major board representation for search hot paths.
//
// This module is a faithful re-implementation of the semantics found in
// board.js / engine.js, using a flat Uint8Array instead of the
// array-of-arrays representation used by the legacy engine. It exists purely
// for speed (search inner loops); the legacy engine remains the source of
// truth and is kept around for the app. See tests/fast-board.test.js for the
// exhaustive equivalence tests against the legacy implementation.
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLORS,
  ORIENTATIONS,
  STORAGE_HEIGHT,
  TOP_OUT_COLUMN,
  TOP_OUT_ROW,
  VISIBLE_HEIGHT,
} from "./constants.js";

const CELL_COUNT = BOARD_WIDTH * BOARD_HEIGHT;

export const FAST_COLORS = Object.freeze({
  EMPTY: 0,
  RED: 1,
  GREEN: 2,
  BLUE: 3,
  YELLOW: 4,
  GARBAGE: 5,
});

const CHAR_TO_CODE = {
  [COLORS.EMPTY]: FAST_COLORS.EMPTY,
  [COLORS.RED]: FAST_COLORS.RED,
  [COLORS.GREEN]: FAST_COLORS.GREEN,
  [COLORS.BLUE]: FAST_COLORS.BLUE,
  [COLORS.YELLOW]: FAST_COLORS.YELLOW,
  [COLORS.GARBAGE]: FAST_COLORS.GARBAGE,
};

const CODE_TO_CHAR = [
  COLORS.EMPTY,
  COLORS.RED,
  COLORS.GREEN,
  COLORS.BLUE,
  COLORS.YELLOW,
  COLORS.GARBAGE,
];

export function colorToCode(char) {
  const code = CHAR_TO_CODE[char];
  if (code === undefined) {
    throw new Error(`Unknown color: ${char}`);
  }
  return code;
}

export function codeToColor(code) {
  const char = CODE_TO_CHAR[code];
  if (char === undefined) {
    throw new Error(`Unknown color code: ${code}`);
  }
  return char;
}

export function pairToCodes(pair) {
  return { axis: colorToCode(pair.axis), child: colorToCode(pair.child) };
}

// index(x, y) = x * BOARD_HEIGHT + y, y = 0 at the bottom (same orientation
// as the legacy board[y][x] representation, just transposed for column
// locality).
export function fromLegacyBoard(board) {
  const fastBoard = new Uint8Array(CELL_COUNT);
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < BOARD_HEIGHT; y += 1) {
      fastBoard[base + y] = colorToCode(board[y][x]);
    }
  }
  return fastBoard;
}

export function toLegacyBoard(fastBoard) {
  const board = [];
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    board.push(new Array(BOARD_WIDTH).fill(COLORS.EMPTY));
  }
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < BOARD_HEIGHT; y += 1) {
      board[y][x] = codeToColor(fastBoard[base + y]);
    }
  }
  return board;
}

function fastColumnHeightAt(fastBoard, x) {
  const base = x * BOARD_HEIGHT;
  for (let y = STORAGE_HEIGHT - 1; y >= 0; y -= 1) {
    if (fastBoard[base + y] !== FAST_COLORS.EMPTY) {
      return y + 1;
    }
  }
  return 0;
}

export function fastColumnHeights(fastBoard) {
  const heights = new Array(BOARD_WIDTH);
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    heights[x] = fastColumnHeightAt(fastBoard, x);
  }
  return heights;
}

export function fastBoardKey(fastBoard) {
  return String.fromCharCode.apply(null, fastBoard);
}

function enumerateDifferentColorActions() {
  const actions = [];

  for (let column = 0; column < BOARD_WIDTH; column += 1) {
    actions.push({ column, orientation: ORIENTATIONS.UP });
    actions.push({ column, orientation: ORIENTATIONS.DOWN });
  }

  for (let column = 0; column < BOARD_WIDTH - 1; column += 1) {
    actions.push({ column, orientation: ORIENTATIONS.RIGHT });
  }

  for (let column = 1; column < BOARD_WIDTH; column += 1) {
    actions.push({ column, orientation: ORIENTATIONS.LEFT });
  }

  return actions;
}

// When axis === child, the legacy enumerateLegalActions dedupes actions by
// the resulting (x, y, color) cell set. Because axis and child share the
// same color, DOWN is always a duplicate of UP for the same column, and
// LEFT(k+1) is always a duplicate of RIGHT(k) for the same column pair,
// regardless of board heights (the cell coordinate sets are identical). The
// surviving order is therefore always UP columns 0..5 followed by RIGHT
// columns 0..4, independent of the actual board contents.
const DIFFERENT_COLOR_ACTIONS = enumerateDifferentColorActions();
const SAME_COLOR_ACTIONS = [
  { column: 0, orientation: ORIENTATIONS.UP },
  { column: 1, orientation: ORIENTATIONS.UP },
  { column: 2, orientation: ORIENTATIONS.UP },
  { column: 3, orientation: ORIENTATIONS.UP },
  { column: 4, orientation: ORIENTATIONS.UP },
  { column: 5, orientation: ORIENTATIONS.UP },
  { column: 0, orientation: ORIENTATIONS.RIGHT },
  { column: 1, orientation: ORIENTATIONS.RIGHT },
  { column: 2, orientation: ORIENTATIONS.RIGHT },
  { column: 3, orientation: ORIENTATIONS.RIGHT },
  { column: 4, orientation: ORIENTATIONS.RIGHT },
];

export function fastEnumerateLegalActions(fastBoard, axisCode, childCode) {
  return axisCode === childCode
    ? SAME_COLOR_ACTIONS.slice()
    : DIFFERENT_COLOR_ACTIONS.slice();
}

function fastComputePlacementCells(fastBoard, axisCode, childCode, action) {
  const { column, orientation } = action;

  if (orientation === ORIENTATIONS.UP || orientation === ORIENTATIONS.DOWN) {
    if (column < 0 || column >= BOARD_WIDTH) {
      throw new Error(`Column out of range: ${column}`);
    }

    const height = fastColumnHeightAt(fastBoard, column);
    const lowerY = height;
    const upperY = height + 1;

    if (orientation === ORIENTATIONS.UP) {
      return [
        { x: column, y: lowerY, color: axisCode },
        { x: column, y: upperY, color: childCode },
      ];
    }

    return [
      { x: column, y: upperY, color: axisCode },
      { x: column, y: lowerY, color: childCode },
    ];
  }

  if (orientation === ORIENTATIONS.RIGHT) {
    if (column < 0 || column >= BOARD_WIDTH - 1) {
      throw new Error(`Column out of range for RIGHT: ${column}`);
    }

    const axisHeight = fastColumnHeightAt(fastBoard, column);
    const childHeight = fastColumnHeightAt(fastBoard, column + 1);
    return [
      { x: column, y: axisHeight, color: axisCode },
      { x: column + 1, y: childHeight, color: childCode },
    ];
  }

  if (orientation === ORIENTATIONS.LEFT) {
    if (column <= 0 || column >= BOARD_WIDTH) {
      throw new Error(`Column out of range for LEFT: ${column}`);
    }

    const axisHeight = fastColumnHeightAt(fastBoard, column);
    const childHeight = fastColumnHeightAt(fastBoard, column - 1);
    return [
      { x: column, y: axisHeight, color: axisCode },
      { x: column - 1, y: childHeight, color: childCode },
    ];
  }

  throw new Error(`Unknown orientation: ${orientation}`);
}

const CHAIN_BONUS = [
  0,
  8,
  16,
  32,
  64,
  96,
  128,
  160,
  192,
  224,
  256,
  288,
  320,
  352,
  384,
  416,
  448,
  480,
  512,
];

const COLOR_BONUS = {
  1: 0,
  2: 3,
  3: 6,
  4: 12,
  5: 24,
};

const GROUP_BONUS = {
  4: 0,
  5: 2,
  6: 3,
  7: 4,
  8: 5,
  9: 6,
  10: 7,
};

function chainBonusFor(chain) {
  return CHAIN_BONUS[Math.min(chain - 1, CHAIN_BONUS.length - 1)];
}

function groupBonusFor(size) {
  if (size >= 11) {
    return 10;
  }
  return GROUP_BONUS[size] ?? 0;
}

function colorBonusFor(count) {
  return COLOR_BONUS[count] ?? 0;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

// Reused scratch buffers for the flood-fill group detection. Cleared with
// fill(0) at the start of every scan for correctness (a generation-counter
// scheme would need to handle Uint8Array wraparound, which is not worth the
// risk here).
const visitedBuf = new Uint8Array(CELL_COUNT);
const stackBuf = new Int16Array(CELL_COUNT);
const garbageFlags = new Uint8Array(CELL_COUNT);

function isClearableColor(color) {
  return color !== FAST_COLORS.EMPTY && color !== FAST_COLORS.GARBAGE;
}

function findMatchedGroups(working) {
  visitedBuf.fill(0);
  const groups = [];

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < VISIBLE_HEIGHT; y += 1) {
      const index = base + y;
      const color = working[index];
      if (!isClearableColor(color) || visitedBuf[index]) {
        continue;
      }

      let stackSize = 0;
      stackBuf[stackSize] = index;
      stackSize += 1;
      visitedBuf[index] = 1;
      const cells = [index];

      while (stackSize > 0) {
        stackSize -= 1;
        const current = stackBuf[stackSize];
        const cx = (current / BOARD_HEIGHT) | 0;
        const cy = current % BOARD_HEIGHT;

        if (cx + 1 < BOARD_WIDTH) {
          const neighbor = current + BOARD_HEIGHT;
          if (!visitedBuf[neighbor] && working[neighbor] === color) {
            visitedBuf[neighbor] = 1;
            stackBuf[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
        if (cx - 1 >= 0) {
          const neighbor = current - BOARD_HEIGHT;
          if (!visitedBuf[neighbor] && working[neighbor] === color) {
            visitedBuf[neighbor] = 1;
            stackBuf[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
        if (cy + 1 < VISIBLE_HEIGHT) {
          const neighbor = current + 1;
          if (!visitedBuf[neighbor] && working[neighbor] === color) {
            visitedBuf[neighbor] = 1;
            stackBuf[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
        if (cy - 1 >= 0) {
          const neighbor = current - 1;
          if (!visitedBuf[neighbor] && working[neighbor] === color) {
            visitedBuf[neighbor] = 1;
            stackBuf[stackSize] = neighbor;
            stackSize += 1;
            cells.push(neighbor);
          }
        }
      }

      if (cells.length >= 4) {
        groups.push({ color, cells });
      }
    }
  }

  return groups;
}

// Boundary is the full board (BOARD_HEIGHT rows), matching engine.js's
// clearAdjacentGarbage which relies on isInsideBoard rather than
// VISIBLE_HEIGHT, so garbage sitting in the hidden rows can also be cleared.
function markAdjacentGarbage(working, index) {
  const x = (index / BOARD_HEIGHT) | 0;
  const y = index % BOARD_HEIGHT;

  if (x + 1 < BOARD_WIDTH) {
    const neighbor = index + BOARD_HEIGHT;
    if (working[neighbor] === FAST_COLORS.GARBAGE) {
      garbageFlags[neighbor] = 1;
    }
  }
  if (x - 1 >= 0) {
    const neighbor = index - BOARD_HEIGHT;
    if (working[neighbor] === FAST_COLORS.GARBAGE) {
      garbageFlags[neighbor] = 1;
    }
  }
  if (y + 1 < BOARD_HEIGHT) {
    const neighbor = index + 1;
    if (working[neighbor] === FAST_COLORS.GARBAGE) {
      garbageFlags[neighbor] = 1;
    }
  }
  if (y - 1 >= 0) {
    const neighbor = index - 1;
    if (working[neighbor] === FAST_COLORS.GARBAGE) {
      garbageFlags[neighbor] = 1;
    }
  }
}

function applyFastGravity(working) {
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    let writePos = 0;

    for (let y = 0; y < STORAGE_HEIGHT; y += 1) {
      const value = working[base + y];
      if (value !== FAST_COLORS.EMPTY) {
        working[base + writePos] = value;
        writePos += 1;
      }
    }

    for (let y = writePos; y < BOARD_HEIGHT; y += 1) {
      working[base + y] = FAST_COLORS.EMPTY;
    }
  }
}

function fastIsBoardEmpty(working) {
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < STORAGE_HEIGHT; y += 1) {
      if (working[base + y] !== FAST_COLORS.EMPTY) {
        return false;
      }
    }
  }
  return true;
}

function resolveFastBoard(working) {
  let chain = 0;
  let totalScore = 0;

  while (true) {
    const matchedGroups = findMatchedGroups(working);
    if (matchedGroups.length === 0) {
      break;
    }

    chain += 1;
    const colorSet = new Set();
    let erasedCount = 0;
    let groupBonusSum = 0;
    for (const group of matchedGroups) {
      colorSet.add(group.color);
      erasedCount += group.cells.length;
      groupBonusSum += groupBonusFor(group.cells.length);
    }

    const multiplier = clamp(
      chainBonusFor(chain) + colorBonusFor(colorSet.size) + groupBonusSum,
      1,
      999,
    );
    totalScore += 10 * erasedCount * multiplier;

    for (const group of matchedGroups) {
      for (const index of group.cells) {
        working[index] = FAST_COLORS.EMPTY;
      }
    }

    garbageFlags.fill(0);
    for (const group of matchedGroups) {
      for (const index of group.cells) {
        markAdjacentGarbage(working, index);
      }
    }
    for (let index = 0; index < CELL_COUNT; index += 1) {
      if (garbageFlags[index]) {
        working[index] = FAST_COLORS.EMPTY;
      }
    }

    applyFastGravity(working);
  }

  return {
    totalChains: chain,
    totalScore,
    allClear: fastIsBoardEmpty(working),
  };
}

export function fastResolveTurn(fastBoard, axisCode, childCode, action) {
  const working = new Uint8Array(fastBoard);
  const cells = fastComputePlacementCells(fastBoard, axisCode, childCode, action);

  let topout = false;
  for (const cell of cells) {
    if (cell.y < STORAGE_HEIGHT) {
      working[cell.x * BOARD_HEIGHT + cell.y] = cell.color;
    }
    if (cell.x === TOP_OUT_COLUMN && cell.y === TOP_OUT_ROW) {
      topout = true;
    }
  }

  if (topout) {
    return {
      board: working,
      topout: true,
      totalChains: 0,
      totalScore: 0,
      allClear: false,
    };
  }

  const resolved = resolveFastBoard(working);

  return {
    board: working,
    topout: false,
    totalChains: resolved.totalChains,
    totalScore: resolved.totalScore,
    allClear: resolved.allClear,
  };
}
