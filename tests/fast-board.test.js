import test from "node:test";
import assert from "node:assert/strict";

import {
  colorToCode,
  codeToColor,
  fastBoardHash,
  fastBoardKey,
  fastColumnHeights,
  fastEnumerateLegalActions,
  fastResolveTurn,
  fromLegacyBoard,
  pairToCodes,
  toLegacyBoard,
} from "../src/core/fast-board.js";
import {
  boardFromRows,
  boardToRows,
  createEmptyBoard,
  enumerateLegalActions,
  getColumnHeight,
} from "../src/core/board.js";
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLORS,
  PLAYABLE_COLORS,
  STORAGE_HEIGHT,
} from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";
import { createRng } from "../src/core/randomizer.js";

const ALL_COLORS = [
  COLORS.EMPTY,
  COLORS.RED,
  COLORS.GREEN,
  COLORS.BLUE,
  COLORS.YELLOW,
  COLORS.GARBAGE,
];

function randomColor(rng, palette) {
  return palette[rng.nextInt(palette.length)];
}

function randomPair(rng, palette) {
  return {
    axis: randomColor(rng, palette),
    child: randomColor(rng, palette),
  };
}

// Builds a gravity-stable board (no floating puyos): each column is filled
// bottom-up to a random height with no gaps, optionally mixing in garbage.
function generateStableBoard(rng, { paletteSize, maxHeight, garbagePercent }) {
  const board = createEmptyBoard();
  const palette = PLAYABLE_COLORS.slice(0, paletteSize);

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const height = rng.nextInt(maxHeight + 1);
    const clampedGarbagePercent = Math.max(0, Math.min(100, garbagePercent));
    for (let y = 0; y < height; y += 1) {
      if (clampedGarbagePercent > 0 && rng.nextInt(100) < clampedGarbagePercent) {
        board[y][x] = COLORS.GARBAGE;
      } else {
        board[y][x] = randomColor(rng, palette);
      }
    }
  }

  return board;
}

const BOARD_PROFILES = [
  { paletteSize: 2, maxHeight: 4, garbagePercent: 0 },
  { paletteSize: 2, maxHeight: 8, garbagePercent: 0 },
  { paletteSize: 2, maxHeight: STORAGE_HEIGHT - 1, garbagePercent: 5 },
  { paletteSize: 3, maxHeight: 6, garbagePercent: 0 },
  { paletteSize: 3, maxHeight: STORAGE_HEIGHT - 1, garbagePercent: 10 },
  { paletteSize: 4, maxHeight: 3, garbagePercent: 0 },
  { paletteSize: 4, maxHeight: 9, garbagePercent: 15 },
  { paletteSize: 4, maxHeight: STORAGE_HEIGHT - 1, garbagePercent: 20 },
];

function generateBoardSet(rng, count) {
  const boards = [];
  for (let i = 0; i < count; i += 1) {
    const profile = BOARD_PROFILES[i % BOARD_PROFILES.length];
    boards.push(generateStableBoard(rng, profile));
  }
  return boards;
}

test("fromLegacyBoard / toLegacyBoard round-trip on random boards", () => {
  const rng = createRng("fast-board-roundtrip");

  for (let i = 0; i < 100; i += 1) {
    const board = createEmptyBoard();
    for (let y = 0; y < BOARD_HEIGHT; y += 1) {
      for (let x = 0; x < BOARD_WIDTH; x += 1) {
        board[y][x] = randomColor(rng, ALL_COLORS);
      }
    }

    const fastBoard = fromLegacyBoard(board);
    const roundTripped = toLegacyBoard(fastBoard);
    assert.deepEqual(roundTripped, board);
  }
});

test("colorToCode / codeToColor are inverses for every known color", () => {
  for (const color of ALL_COLORS) {
    assert.equal(codeToColor(colorToCode(color)), color);
  }
});

test("fastColumnHeights matches getColumnHeight on random boards", () => {
  const rng = createRng("fast-board-heights");
  const boards = generateBoardSet(rng, 200);

  for (const board of boards) {
    const fastBoard = fromLegacyBoard(board);
    const fastHeights = fastColumnHeights(fastBoard);
    const legacyHeights = Array.from({ length: BOARD_WIDTH }, (_, x) =>
      getColumnHeight(board, x),
    );
    assert.deepEqual(fastHeights, legacyHeights);
  }
});

test("fastEnumerateLegalActions matches enumerateLegalActions", () => {
  const rng = createRng("fast-board-actions");
  const boards = generateBoardSet(rng, 200);

  for (const board of boards) {
    const fastBoard = fromLegacyBoard(board);
    // Half same-color pairs, half different-color pairs.
    const pair =
      rng.nextInt(2) === 0
        ? { axis: randomColor(rng, PLAYABLE_COLORS), child: undefined }
        : randomPair(rng, PLAYABLE_COLORS);
    if (pair.child === undefined) {
      pair.child = pair.axis;
    }

    const codes = pairToCodes(pair);
    const fastActions = fastEnumerateLegalActions(fastBoard, codes.axis, codes.child);
    const legacyActions = enumerateLegalActions(board, pair);

    assert.deepEqual(fastActions, legacyActions);
  }
});

test("fastResolveTurn matches resolveTurn across many boards/pairs/actions", () => {
  const rng = createRng("fast-board-resolve");
  const boards = generateBoardSet(rng, 60);

  let caseCount = 0;
  let chainedCaseCount = 0;

  for (const board of boards) {
    const fastBoard = fromLegacyBoard(board);

    for (let pairIndex = 0; pairIndex < 3; pairIndex += 1) {
      const pair = randomPair(rng, PLAYABLE_COLORS);
      const codes = pairToCodes(pair);
      const legacyActions = enumerateLegalActions(board, pair);

      for (const action of legacyActions) {
        const legacyResult = resolveTurn(board, pair, action);
        const fastResult = fastResolveTurn(fastBoard, codes.axis, codes.child, action);

        caseCount += 1;
        if (legacyResult.totalChains > 0) {
          chainedCaseCount += 1;
        }

        assert.equal(fastResult.topout, legacyResult.topout, `topout mismatch for ${JSON.stringify(action)}`);
        assert.equal(
          fastResult.totalChains,
          legacyResult.totalChains,
          `totalChains mismatch for ${JSON.stringify(action)}`,
        );
        assert.equal(
          fastResult.totalScore,
          legacyResult.totalScore,
          `totalScore mismatch for ${JSON.stringify(action)}`,
        );
        assert.equal(
          fastResult.allClear,
          legacyResult.allClear,
          `allClear mismatch for ${JSON.stringify(action)}`,
        );
        assert.deepEqual(
          toLegacyBoard(fastResult.board),
          legacyResult.finalBoard,
          `final board mismatch for ${JSON.stringify(action)}`,
        );

        // fastResolveTurn must not mutate the board it was given.
        assert.deepEqual(toLegacyBoard(fastBoard), board);
      }
    }
  }

  assert.ok(caseCount >= 2000, `expected at least 2000 cases, got ${caseCount}`);
  assert.ok(
    chainedCaseCount >= 200,
    `expected at least 200 chained cases, got ${chainedCaseCount}`,
  );
});

test("known chain scenarios match exactly (reused from core.test.js)", () => {
  const singleChainBoard = boardFromRows([
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "RRR...",
  ]);
  const singleChainFast = fromLegacyBoard(singleChainBoard);
  const singleChainCodes = pairToCodes({ axis: COLORS.RED, child: COLORS.GREEN });
  const singleChainResult = fastResolveTurn(
    singleChainFast,
    singleChainCodes.axis,
    singleChainCodes.child,
    { column: 3, orientation: "RIGHT" },
  );
  assert.equal(singleChainResult.totalChains, 1);
  assert.equal(singleChainResult.totalScore, 40);
  assert.deepEqual(boardToRows(toLegacyBoard(singleChainResult.board)).slice(-1), ["....G."]);

  const doubleChainBoard = boardFromRows([
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "GGGRRR",
  ]);
  const doubleChainFast = fromLegacyBoard(doubleChainBoard);
  const doubleChainCodes = pairToCodes({ axis: COLORS.RED, child: COLORS.GREEN });
  const doubleChainResult = fastResolveTurn(
    doubleChainFast,
    doubleChainCodes.axis,
    doubleChainCodes.child,
    { column: 3, orientation: "UP" },
  );
  assert.equal(doubleChainResult.totalChains, 2);
  assert.equal(doubleChainResult.totalScore, 360);

  const topoutBoard = boardFromRows([
    "......",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
  ]);
  const topoutFast = fromLegacyBoard(topoutBoard);
  const topoutCodes = pairToCodes({ axis: COLORS.BLUE, child: COLORS.YELLOW });
  const topoutResult = fastResolveTurn(topoutFast, topoutCodes.axis, topoutCodes.child, {
    column: 2,
    orientation: "UP",
  });
  assert.equal(topoutResult.topout, true);
  assert.equal(topoutResult.totalChains, 0);
  assert.equal(topoutResult.totalScore, 0);

  const garbageBoard = boardFromRows([
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "RRR.O.",
  ]);
  const garbageFast = fromLegacyBoard(garbageBoard);
  const garbageCodes = pairToCodes({ axis: COLORS.RED, child: COLORS.GREEN });
  const garbageResult = fastResolveTurn(garbageFast, garbageCodes.axis, garbageCodes.child, {
    column: 3,
    orientation: "RIGHT",
  });
  assert.equal(garbageResult.totalChains, 1);
  assert.deepEqual(boardToRows(toLegacyBoard(garbageResult.board)).slice(-1), ["....G."]);
});

test("fastBoardKey uniquely identifies board contents", () => {
  const rng = createRng("fast-board-key");
  const boards = generateBoardSet(rng, 30);
  const keys = boards.map((board) => fastBoardKey(fromLegacyBoard(board)));

  for (let i = 0; i < keys.length; i += 1) {
    for (let j = i + 1; j < keys.length; j += 1) {
      if (JSON.stringify(boards[i]) !== JSON.stringify(boards[j])) {
        assert.notEqual(keys[i], keys[j], `boards ${i} and ${j} differ but produced the same key`);
      }
    }
  }

  const sampleBoard = boards[0];
  const keyA = fastBoardKey(fromLegacyBoard(sampleBoard));
  const keyB = fastBoardKey(fromLegacyBoard(sampleBoard));
  assert.equal(keyA, keyB);
});

test("fastBoardHash is deterministic for the same board contents", () => {
  const rng = createRng("fast-board-hash-determinism");
  const boards = generateBoardSet(rng, 30);

  for (const board of boards) {
    const hashA = fastBoardHash(fromLegacyBoard(board));
    const hashB = fastBoardHash(fromLegacyBoard(board));
    assert.equal(hashA, hashB);
  }
});

test("fastBoardHash has zero collisions across 1000 random boards", () => {
  const rng = createRng("fast-board-hash-collisions");
  const boards = generateBoardSet(rng, 1000);
  const keyByHash = new Map();
  let collisions = 0;

  for (const board of boards) {
    const fastBoard = fromLegacyBoard(board);
    const hash = fastBoardHash(fastBoard);
    const key = fastBoardKey(fastBoard);

    const existingKey = keyByHash.get(hash);
    if (existingKey !== undefined && existingKey !== key) {
      collisions += 1;
    }
    keyByHash.set(hash, key);
  }

  assert.equal(
    collisions,
    0,
    `expected zero hash collisions across ${boards.length} boards, got ${collisions}`,
  );
});
