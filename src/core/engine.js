import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLORS,
  EVENT_TYPES,
  STORAGE_HEIGHT,
  TOP_OUT_COLUMN,
  TOP_OUT_ROW,
  VISIBLE_HEIGHT,
} from "./constants.js";
import { applyPlacement, cloneBoard, isBoardEmpty, isInsideBoard } from "./board.js";

const NEIGHBOR_OFFSETS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

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

function makeEvent(type, board, meta = {}) {
  return {
    type,
    board: cloneBoard(board),
    ...meta,
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
          if (!isInsideBoard(nx, ny) || ny >= VISIBLE_HEIGHT) {
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

function applyGravity(board) {
  let changed = false;

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const column = [];
    for (let y = 0; y < STORAGE_HEIGHT; y += 1) {
      if (board[y][x] !== COLORS.EMPTY) {
        column.push(board[y][x]);
      }
    }

    for (let y = 0; y < BOARD_HEIGHT; y += 1) {
      const nextColor = column[y] ?? COLORS.EMPTY;
      if (board[y][x] !== nextColor) {
        board[y][x] = nextColor;
        changed = true;
      }
    }
  }

  return changed;
}

function resolveBoardState(workingBoard, events = null) {
  let chain = 0;
  let totalScore = 0;
  const stepScores = [];

  while (true) {
    const matchedGroups = collectGroups(workingBoard).filter(
      (group) => group.cells.length >= 4,
    );

    if (matchedGroups.length === 0) {
      break;
    }

    chain += 1;
    const colorsCleared = new Set(matchedGroups.map((group) => group.color)).size;
    const erasedCount = matchedGroups.reduce(
      (sum, group) => sum + group.cells.length,
      0,
    );
    const groupBonus = matchedGroups.reduce(
      (sum, group) => sum + groupBonusFor(group.cells.length),
      0,
    );
    const multiplier = Math.min(
      Math.max(
        1,
        chainBonusFor(chain) + colorBonusFor(colorsCleared) + groupBonus,
      ),
      999,
    );
    const stepScore = 10 * erasedCount * multiplier;
    totalScore += stepScore;
    stepScores.push(stepScore);

    for (const group of matchedGroups) {
      for (const cell of group.cells) {
        workingBoard[cell.y][cell.x] = COLORS.EMPTY;
      }
    }

    if (events) {
      events.push(
        makeEvent(EVENT_TYPES.CLEAR, workingBoard, {
          chain,
          stepScore,
          totalScore,
          erasedCount,
          groups: matchedGroups.map((group) => ({
            color: group.color,
            size: group.cells.length,
            cells: group.cells.map((cell) => ({ ...cell })),
          })),
        }),
      );
    }

    applyGravity(workingBoard);

    if (events) {
      events.push(
        makeEvent(EVENT_TYPES.GRAVITY, workingBoard, {
          chain,
          totalScore,
        }),
      );
    }
  }

  return {
    finalBoard: cloneBoard(workingBoard),
    totalChains: chain,
    totalScore,
    stepScores,
    allClear: isBoardEmpty(workingBoard),
  };
}

export function resolveBoard(board) {
  const workingBoard = cloneBoard(board);
  return resolveBoardState(workingBoard);
}

export function resolveTurn(board, pair, action) {
  const placement = applyPlacement(board, pair, action);
  const workingBoard = placement.board;
  const events = [
    makeEvent(EVENT_TYPES.PLACE, workingBoard, {
      action,
      pair,
      placedCells: placement.cells,
    }),
  ];

  const topout = placement.cells.some(
    (cell) => cell.x === TOP_OUT_COLUMN && cell.y === TOP_OUT_ROW,
  );
  if (topout) {
    events.push(
      makeEvent(EVENT_TYPES.SETTLE, workingBoard, {
        topout: true,
        chain: 0,
        stepScore: 0,
        totalScore: 0,
      }),
    );
    return {
      finalBoard: cloneBoard(workingBoard),
      topout: true,
      totalChains: 0,
      totalScore: 0,
      stepScores: [],
      allClear: false,
      events,
    };
  }
  const resolved = resolveBoardState(workingBoard, events);
  events.push(
    makeEvent(EVENT_TYPES.SETTLE, workingBoard, {
      topout: false,
      chain: resolved.totalChains,
      totalScore: resolved.totalScore,
      stepScore: resolved.stepScores.at(-1) ?? 0,
      allClear: resolved.allClear,
    }),
  );

  return {
    finalBoard: resolved.finalBoard,
    topout: false,
    totalChains: resolved.totalChains,
    totalScore: resolved.totalScore,
    stepScores: resolved.stepScores,
    allClear: resolved.allClear,
    events,
  };
}
