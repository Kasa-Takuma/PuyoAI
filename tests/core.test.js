import test from "node:test";
import assert from "node:assert/strict";

import {
  boardFromRows,
  boardToRows,
  computePlacementCells,
  createEmptyBoard,
  enumerateLegalActions,
} from "../src/core/board.js";
import { COLORS, ORIENTATIONS } from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";

test("same-color pair deduplicates to 11 legal actions", () => {
  const actions = enumerateLegalActions(createEmptyBoard(), {
    axis: COLORS.RED,
    child: COLORS.RED,
  });

  assert.equal(actions.length, 11);
});

test("different-color pair has 22 legal actions", () => {
  const actions = enumerateLegalActions(createEmptyBoard(), {
    axis: COLORS.RED,
    child: COLORS.GREEN,
  });

  assert.equal(actions.length, 22);
});

test("horizontal placement falls independently per column", () => {
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
    ".B....",
    ".B....",
    "R.....",
  ]);

  const cells = computePlacementCells(
    board,
    { axis: COLORS.YELLOW, child: COLORS.GREEN },
    { column: 0, orientation: ORIENTATIONS.RIGHT },
  );

  assert.deepEqual(
    cells.map(({ x, y, color }) => ({ x, y, color })),
    [
      { x: 0, y: 1, color: COLORS.YELLOW },
      { x: 1, y: 3, color: COLORS.GREEN },
    ],
  );
});

test("single chain scores 40 and leaves the non-cleared puyo", () => {
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
    "RRR...",
  ]);

  const result = resolveTurn(
    board,
    { axis: COLORS.RED, child: COLORS.GREEN },
    { column: 3, orientation: ORIENTATIONS.RIGHT },
  );

  assert.equal(result.totalChains, 1);
  assert.equal(result.totalScore, 40);
  assert.deepEqual(boardToRows(result.finalBoard).slice(-1), ["....G."]);
});

test("double chain scores 360", () => {
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

  const result = resolveTurn(
    board,
    { axis: COLORS.RED, child: COLORS.GREEN },
    { column: 3, orientation: ORIENTATIONS.UP },
  );

  assert.equal(result.totalChains, 2);
  assert.equal(result.totalScore, 360);
  assert.equal(result.stepScores[0], 40);
  assert.equal(result.stepScores[1], 320);
});

test("topout occurs when both placed puyos land outside visible rows", () => {
  const board = boardFromRows([
    "......",
    "......",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
    "R.....",
  ]);

  const result = resolveTurn(
    board,
    { axis: COLORS.BLUE, child: COLORS.YELLOW },
    { column: 0, orientation: ORIENTATIONS.UP },
  );

  assert.equal(result.topout, true);
  assert.equal(result.totalChains, 0);
  assert.equal(result.totalScore, 0);
});
