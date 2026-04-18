import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLORS,
  ORIENTATIONS,
  ORIENTATION_ORDER,
} from "./constants.js";

export function createEmptyBoard() {
  return Array.from({ length: BOARD_HEIGHT }, () =>
    Array.from({ length: BOARD_WIDTH }, () => COLORS.EMPTY),
  );
}

export function cloneBoard(board) {
  return board.map((row) => [...row]);
}

export function isInsideBoard(x, y) {
  return x >= 0 && x < BOARD_WIDTH && y >= 0 && y < BOARD_HEIGHT;
}

export function boardFromRows(rowsTopToBottom) {
  const board = createEmptyBoard();
  const clampedRows = rowsTopToBottom.slice(-BOARD_HEIGHT);
  const topOffset = BOARD_HEIGHT - clampedRows.length;

  for (let rowIndex = 0; rowIndex < clampedRows.length; rowIndex += 1) {
    const sourceRow = clampedRows[rowIndex].padEnd(BOARD_WIDTH, COLORS.EMPTY);
    const y = BOARD_HEIGHT - 1 - (rowIndex + topOffset);
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      board[y][x] = sourceRow[x] ?? COLORS.EMPTY;
    }
  }

  return board;
}

export function boardToRows(board) {
  const rows = [];
  for (let y = BOARD_HEIGHT - 1; y >= 0; y -= 1) {
    rows.push(board[y].join(""));
  }
  return rows;
}

export function getColumnHeight(board, x) {
  for (let y = BOARD_HEIGHT - 1; y >= 0; y -= 1) {
    if (board[y][x] !== COLORS.EMPTY) {
      return y + 1;
    }
  }

  return 0;
}

export function isBoardEmpty(board) {
  return board.every((row) => row.every((cell) => cell === COLORS.EMPTY));
}

function normalizePlacementCells(cells) {
  return [...cells]
    .map((cell) => `${cell.x}:${cell.y}:${cell.color}`)
    .sort()
    .join("|");
}

export function computePlacementCells(board, pair, action) {
  const { axis, child } = pair;
  const { column, orientation } = action;

  if (!ORIENTATION_ORDER.includes(orientation)) {
    throw new Error(`Unknown orientation: ${orientation}`);
  }

  if (orientation === ORIENTATIONS.UP || orientation === ORIENTATIONS.DOWN) {
    if (column < 0 || column >= BOARD_WIDTH) {
      throw new Error(`Column out of range: ${column}`);
    }

    const height = getColumnHeight(board, column);
    const lowerY = height;
    const upperY = height + 1;

    if (orientation === ORIENTATIONS.UP) {
      return [
        { x: column, y: lowerY, color: axis, role: "axis" },
        { x: column, y: upperY, color: child, role: "child" },
      ];
    }

    return [
      { x: column, y: upperY, color: axis, role: "axis" },
      { x: column, y: lowerY, color: child, role: "child" },
    ];
  }

  if (orientation === ORIENTATIONS.RIGHT) {
    if (column < 0 || column >= BOARD_WIDTH - 1) {
      throw new Error(`Column out of range for RIGHT: ${column}`);
    }

    const axisHeight = getColumnHeight(board, column);
    const childHeight = getColumnHeight(board, column + 1);
    return [
      { x: column, y: axisHeight, color: axis, role: "axis" },
      { x: column + 1, y: childHeight, color: child, role: "child" },
    ];
  }

  if (column <= 0 || column >= BOARD_WIDTH) {
    throw new Error(`Column out of range for LEFT: ${column}`);
  }

  const axisHeight = getColumnHeight(board, column);
  const childHeight = getColumnHeight(board, column - 1);
  return [
    { x: column, y: axisHeight, color: axis, role: "axis" },
    { x: column - 1, y: childHeight, color: child, role: "child" },
  ];
}

export function applyPlacement(board, pair, action) {
  const nextBoard = cloneBoard(board);
  const cells = computePlacementCells(board, pair, action);

  for (const cell of cells) {
    if (isInsideBoard(cell.x, cell.y)) {
      nextBoard[cell.y][cell.x] = cell.color;
    }
  }

  return {
    board: nextBoard,
    cells,
  };
}

function enumerateRawActions() {
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

export function enumerateLegalActions(board, pair) {
  const rawActions = enumerateRawActions();

  if (pair.axis !== pair.child) {
    return rawActions;
  }

  const unique = [];
  const seen = new Set();

  for (const action of rawActions) {
    const key = normalizePlacementCells(computePlacementCells(board, pair, action));
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(action);
  }

  return unique;
}

export function encodeAction(action) {
  return `${action.orientation}:${action.column}`;
}

export function decodeAction(value) {
  const [orientation, columnText] = value.split(":");
  return {
    orientation,
    column: Number.parseInt(columnText, 10),
  };
}
