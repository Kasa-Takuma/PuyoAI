import test from "node:test";
import assert from "node:assert/strict";

import { extractBoardFeatures } from "../src/ai/features.js";
import { extractBoardFeaturesFast } from "../src/ai/features-fast.js";
import { boardFromRows, createEmptyBoard } from "../src/core/board.js";
import { BOARD_WIDTH, COLORS, PLAYABLE_COLORS, STORAGE_HEIGHT } from "../src/core/constants.js";
import { fromLegacyBoard } from "../src/core/fast-board.js";
import { createRng } from "../src/core/randomizer.js";

function randomColor(rng, palette) {
  return palette[rng.nextInt(palette.length)];
}

// Builds a gravity-stable board (no floating puyos): each column is filled
// bottom-up to a random height with no gaps, optionally mixing in garbage.
// Mirrors the generator used in tests/fast-board.test.js.
function generateStableBoard(rng, { paletteSize, maxHeight, garbagePercent }) {
  const board = createEmptyBoard();
  const palette = PLAYABLE_COLORS.slice(0, paletteSize);
  const clampedGarbagePercent = Math.max(0, Math.min(100, garbagePercent));

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const height = rng.nextInt(maxHeight + 1);
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
  { paletteSize: 2, maxHeight: 8, garbagePercent: 5 },
  { paletteSize: 2, maxHeight: STORAGE_HEIGHT - 1, garbagePercent: 10 },
  { paletteSize: 3, maxHeight: 6, garbagePercent: 0 },
  { paletteSize: 3, maxHeight: 9, garbagePercent: 15 },
  { paletteSize: 3, maxHeight: STORAGE_HEIGHT - 1, garbagePercent: 20 },
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

test("extractBoardFeaturesFast matches extractBoardFeatures exactly on random stable boards", () => {
  const rng = createRng("features-fast-parity");
  const boards = generateBoardSet(rng, 300);

  let highChainBoardCount = 0;

  for (const board of boards) {
    const fastBoard = fromLegacyBoard(board);

    const legacyFull = extractBoardFeatures(board, { includeVirtualChains: true });
    const fastFull = extractBoardFeaturesFast(fastBoard, { includeVirtualChains: true });
    assert.deepEqual(fastFull, legacyFull);

    const legacyBase = extractBoardFeatures(board, { includeVirtualChains: false });
    const fastBase = extractBoardFeaturesFast(fastBoard, { includeVirtualChains: false });
    assert.deepEqual(fastBase, legacyBase);

    if (legacyFull.bestVirtualChain >= 2) {
      highChainBoardCount += 1;
    }
  }

  assert.ok(
    highChainBoardCount >= 50,
    `expected at least 50 boards with bestVirtualChain >= 2, got ${highChainBoardCount}`,
  );
});

test("known GGGRRR board produces the expected virtual double chain", () => {
  const board = boardFromRows([
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
  const fastBoard = fromLegacyBoard(board);

  const features = extractBoardFeaturesFast(fastBoard);

  assert.equal(features.bestVirtualChain, 2);
  assert.ok(features.bestVirtualScore >= 360);
  assert.ok(features.virtualChainCount2Plus >= 1);
});

test("cache returns the same object for repeated calls and separates base/full entries", () => {
  const rng = createRng("features-fast-cache");
  const board = generateStableBoard(rng, {
    paletteSize: 4,
    maxHeight: 8,
    garbagePercent: 10,
  });

  const first = extractBoardFeaturesFast(fromLegacyBoard(board), {
    includeVirtualChains: true,
  });
  const second = extractBoardFeaturesFast(fromLegacyBoard(board), {
    includeVirtualChains: true,
  });
  assert.equal(first, second);

  const baseFeatures = extractBoardFeaturesFast(fromLegacyBoard(board), {
    includeVirtualChains: false,
  });
  assert.notEqual(first, baseFeatures);
});
