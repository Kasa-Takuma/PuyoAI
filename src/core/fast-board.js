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

// Two independent 32bit FNV-1a variants (different offset basis / prime),
// combined into a single 44bit integer safely within Number's 53bit exact
// range: the low 22 bits of h1 become the high bits, and the high 22 bits of
// h2 become the low bits. This avoids the string allocation + hashing cost
// of fastBoardKey on search hot paths where the key only needs to dedupe
// board states (a false-positive collision merely drops one candidate; it
// cannot corrupt correctness of callers that also verify board equality).
const FNV_OFFSET_1 = 0x811c9dc5;
const FNV_PRIME_1 = 0x01000193;
const FNV_OFFSET_2 = 0x9e3779b9;
const FNV_PRIME_2 = 0x85ebca6b;

export function fastBoardHash(fastBoard) {
  let h1 = FNV_OFFSET_1;
  let h2 = FNV_OFFSET_2;
  for (let i = 0; i < CELL_COUNT; i += 1) {
    const byte = fastBoard[i];
    h1 = Math.imul(h1 ^ byte, FNV_PRIME_1);
    h2 = Math.imul(h2 ^ byte, FNV_PRIME_2);
  }
  h1 >>>= 0;
  h2 >>>= 0;
  return (h1 & 0x3fffff) * 4194304 + (h2 >>> 10);
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

// --- Bitboard group detection ------------------------------------------
//
// The 6x14 board is packed into 3 words (BB_WORDS). Word `w` holds columns
// `2*w` ("lane A", bits 0-15) and `2*w+1` ("lane B", bits 16-31); within a
// lane, bit `y` is row y. Only bits 0-13 of each lane are ever populated
// (BOARD_HEIGHT = 14); bits 14-15 stay 0 so a vertical shift can't silently
// smear into the next row. Vertical (y) shifts operate within a lane and
// leak one bit across the lane boundary, which must be masked off: shifting
// up (`<< 1`) pushes lane A's bit 15 into lane B's bit 0 (word bit 16), and
// shifting down (`>>> 1`) pushes lane B's bit 16 into lane A's bit 15.
// Horizontal (x) shifts move whole 16-bit lanes between/within words and
// never leak, since lanes are already word-aligned.
const BB_WORDS = 3;
const LANE_BITS = 16;
const UP_LEAK_MASK = 0xfffeffff; // clears bit16 (lane A bit15 -> lane B bit0 leak)
const DOWN_LEAK_MASK = 0xffff7fff; // clears bit15 (lane B bit16 -> lane A bit15 leak)
const VISIBLE_MASK = 0x0fff0fff; // bits 0-11 of each lane (VISIBLE_HEIGHT = 12)
const BOARD_MASK = 0x3fff3fff; // bits 0-13 of each lane (BOARD_HEIGHT = 14)

// bbUp/bbDown/bbLeft/bbRight are the single-direction building blocks of
// bbDilate below, kept as standalone functions (rather than inlined
// everywhere) so tests/fast-board.test.js can exercise each shift direction
// against known bit patterns in isolation.
export function bbUp(out, w) {
  for (let i = 0; i < BB_WORDS; i += 1) {
    out[i] = (w[i] << 1) & UP_LEAK_MASK;
  }
}

export function bbDown(out, w) {
  for (let i = 0; i < BB_WORDS; i += 1) {
    out[i] = (w[i] >>> 1) & DOWN_LEAK_MASK;
  }
}

export function bbRight(out, w) {
  const w0 = w[0];
  const w1 = w[1];
  const w2 = w[2];
  out[2] = (w2 << 16) | (w1 >>> 16);
  out[1] = (w1 << 16) | (w0 >>> 16);
  out[0] = w0 << 16;
}

export function bbLeft(out, w) {
  const w0 = w[0];
  const w1 = w[1];
  const w2 = w[2];
  out[0] = (w0 >>> 16) | (w1 << 16);
  out[1] = (w1 >>> 16) | (w2 << 16);
  out[2] = w2 >>> 16;
}

// dilate(mask) = mask | up(mask) | down(mask) | left(mask) | right(mask).
// Callers are responsible for masking the result to the region they care
// about (visible rows for group growth, full board for garbage adjacency).
// This is the innermost loop of group extraction, so the four directional
// shifts are computed inline (matching bbUp/bbDown/bbLeft/bbRight exactly)
// instead of calling those helpers through scratch buffers, to avoid extra
// typed-array round-trips per iteration.
function bbDilate(out, w) {
  const w0 = w[0];
  const w1 = w[1];
  const w2 = w[2];

  const merged0 =
    w0 |
    ((w0 << 1) & UP_LEAK_MASK) |
    ((w0 >>> 1) & DOWN_LEAK_MASK) |
    (w0 << 16) |
    ((w0 >>> 16) | (w1 << 16));
  const merged1 =
    w1 |
    ((w1 << 1) & UP_LEAK_MASK) |
    ((w1 >>> 1) & DOWN_LEAK_MASK) |
    ((w1 << 16) | (w0 >>> 16)) |
    ((w1 >>> 16) | (w2 << 16));
  const merged2 =
    w2 |
    ((w2 << 1) & UP_LEAK_MASK) |
    ((w2 >>> 1) & DOWN_LEAK_MASK) |
    ((w2 << 16) | (w1 >>> 16)) |
    (w2 >>> 16);

  out[0] = merged0;
  out[1] = merged1;
  out[2] = merged2;
}

function popcount32(value) {
  let v = value >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

function bbPopcount(bits) {
  return popcount32(bits[0]) + popcount32(bits[1]) + popcount32(bits[2]);
}

function bbEquals(a, b) {
  return a[0] === b[0] >>> 0 && a[1] === b[1] >>> 0 && a[2] === b[2] >>> 0;
}

function bbIsEmpty(bits) {
  return bits[0] === 0 && bits[1] === 0 && bits[2] === 0;
}

// Sets every cell of `working` covered by a set bit of `bits` to EMPTY
// (word-major bit scan). Both erase sites in resolveFastBoard need exactly
// this operation, so it is inlined here rather than exposed as a generic
// callback-based iterator (avoids a closure allocation per chain step on
// the search hot path).
function clearMaskCells(working, bits) {
  for (let wi = 0; wi < BB_WORDS; wi += 1) {
    let word = bits[wi] >>> 0;
    while (word !== 0) {
      const lowestBit = word & -word;
      const bitIndex = 31 - Math.clz32(lowestBit);
      const lane = bitIndex >= LANE_BITS ? 1 : 0;
      const y = bitIndex - lane * LANE_BITS;
      const x = wi * 2 + lane;
      working[x * BOARD_HEIGHT + y] = FAST_COLORS.EMPTY;
      word ^= lowestBit;
    }
  }
}

// COLOR_MASKS[color] holds the bitboard for that FAST_COLORS code (indices
// 0..5, i.e. EMPTY through GARBAGE), rebuilt from scratch on every call to
// findMatchedGroups.
const COLOR_MASKS = [
  new Uint32Array(BB_WORDS),
  new Uint32Array(BB_WORDS),
  new Uint32Array(BB_WORDS),
  new Uint32Array(BB_WORDS),
  new Uint32Array(BB_WORDS),
  new Uint32Array(BB_WORDS),
];

function buildColorMasks(fastBoard) {
  for (let c = 0; c < COLOR_MASKS.length; c += 1) {
    COLOR_MASKS[c][0] = 0;
    COLOR_MASKS[c][1] = 0;
    COLOR_MASKS[c][2] = 0;
  }

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    const word = x >> 1;
    const laneShift = (x & 1) * LANE_BITS;
    for (let y = 0; y < BOARD_HEIGHT; y += 1) {
      const color = fastBoard[base + y];
      COLOR_MASKS[color][word] |= 1 << (laneShift + y);
    }
  }
}

const colorVisibleBuf = new Uint32Array(BB_WORDS);
const residualBuf = new Uint32Array(BB_WORDS);
const seedBuf = new Uint32Array(BB_WORDS);
const compBuf = new Uint32Array(BB_WORDS);
const nextBuf = new Uint32Array(BB_WORDS);
const eraseMaskBuf = new Uint32Array(BB_WORDS);

function findMatchedGroups(working) {
  buildColorMasks(working);
  eraseMaskBuf[0] = 0;
  eraseMaskBuf[1] = 0;
  eraseMaskBuf[2] = 0;
  const groups = [];

  for (let color = FAST_COLORS.RED; color <= FAST_COLORS.YELLOW; color += 1) {
    const colorMask = COLOR_MASKS[color];
    colorVisibleBuf[0] = colorMask[0] & VISIBLE_MASK;
    colorVisibleBuf[1] = colorMask[1] & VISIBLE_MASK;
    colorVisibleBuf[2] = colorMask[2] & VISIBLE_MASK;
    residualBuf[0] = colorVisibleBuf[0];
    residualBuf[1] = colorVisibleBuf[1];
    residualBuf[2] = colorVisibleBuf[2];

    while (!bbIsEmpty(residualBuf)) {
      let seedWord = 0;
      while (residualBuf[seedWord] === 0) {
        seedWord += 1;
      }
      seedBuf[0] = 0;
      seedBuf[1] = 0;
      seedBuf[2] = 0;
      const w = residualBuf[seedWord];
      seedBuf[seedWord] = w & -w;

      compBuf[0] = seedBuf[0];
      compBuf[1] = seedBuf[1];
      compBuf[2] = seedBuf[2];
      for (;;) {
        bbDilate(nextBuf, compBuf);
        nextBuf[0] &= colorVisibleBuf[0];
        nextBuf[1] &= colorVisibleBuf[1];
        nextBuf[2] &= colorVisibleBuf[2];
        if (bbEquals(nextBuf, compBuf)) {
          break;
        }
        compBuf[0] = nextBuf[0];
        compBuf[1] = nextBuf[1];
        compBuf[2] = nextBuf[2];
      }

      const size = bbPopcount(compBuf);
      if (size >= 4) {
        eraseMaskBuf[0] |= compBuf[0];
        eraseMaskBuf[1] |= compBuf[1];
        eraseMaskBuf[2] |= compBuf[2];
        groups.push({ color, size });
      }

      residualBuf[0] &= ~compBuf[0];
      residualBuf[1] &= ~compBuf[1];
      residualBuf[2] &= ~compBuf[2];
    }
  }

  return { groups, eraseMask: eraseMaskBuf };
}

const garbageAdjacentBuf = new Uint32Array(BB_WORDS);

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
    const { groups, eraseMask } = findMatchedGroups(working);
    if (groups.length === 0) {
      break;
    }

    chain += 1;
    const colorSet = new Set();
    let groupBonusSum = 0;
    for (const group of groups) {
      colorSet.add(group.color);
      groupBonusSum += groupBonusFor(group.size);
    }
    const erasedCount = bbPopcount(eraseMask);

    const multiplier = clamp(
      chainBonusFor(chain) + colorBonusFor(colorSet.size) + groupBonusSum,
      1,
      999,
    );
    totalScore += 10 * erasedCount * multiplier;

    clearMaskCells(working, eraseMask);

    // Garbage sitting adjacent to (or in the hidden rows above) an erased
    // cell is also cleared. Boundary is the full board (BOARD_MASK), not
    // just the visible rows, matching engine.js's clearAdjacentGarbage which
    // relies on isInsideBoard rather than VISIBLE_HEIGHT.
    bbDilate(garbageAdjacentBuf, eraseMask);
    garbageAdjacentBuf[0] &= BOARD_MASK & COLOR_MASKS[FAST_COLORS.GARBAGE][0];
    garbageAdjacentBuf[1] &= BOARD_MASK & COLOR_MASKS[FAST_COLORS.GARBAGE][1];
    garbageAdjacentBuf[2] &= BOARD_MASK & COLOR_MASKS[FAST_COLORS.GARBAGE][2];
    clearMaskCells(working, garbageAdjacentBuf);

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
