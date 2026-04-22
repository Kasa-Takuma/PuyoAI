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
import { resolveBoard, resolveTurn } from "../src/core/engine.js";

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

test("topout occurs when a placed puyo lands on the third-column twelfth row", () => {
  const board = boardFromRows([
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

  const result = resolveTurn(
    board,
    { axis: COLORS.BLUE, child: COLORS.YELLOW },
    { column: 2, orientation: ORIENTATIONS.UP },
  );

  assert.equal(result.topout, true);
  assert.equal(result.totalChains, 0);
  assert.equal(result.totalScore, 0);
});

test("thirteenth-row puyos do not participate in chain clearing", () => {
  const board = boardFromRows([
    "....RR",
    "....RR",
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
  ]);

  const result = resolveBoard(board);

  assert.equal(result.totalChains, 0);
  assert.equal(boardToRows(result.finalBoard)[1], "....RR");
  assert.equal(boardToRows(result.finalBoard)[2], "....RR");
});

test("fourteenth-row placements disappear instead of staying on the board", () => {
  const board = boardFromRows([
    "R.....",
    "G.....",
    "B.....",
    "Y.....",
    "R.....",
    "G.....",
    "B.....",
    "Y.....",
    "R.....",
    "G.....",
    "B.....",
    "Y.....",
    "R.....",
  ]);

  const result = resolveTurn(
    board,
    { axis: COLORS.BLUE, child: COLORS.YELLOW },
    { column: 0, orientation: ORIENTATIONS.UP },
  );

  assert.equal(result.topout, false);
  assert.equal(result.totalChains, 0);
  assert.deepEqual(boardToRows(result.finalBoard), boardToRows(board));
});

test("garbage puyos do not clear as a same-color group", () => {
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
    "OOOO..",
  ]);

  const result = resolveBoard(board);

  assert.equal(result.totalChains, 0);
  assert.deepEqual(boardToRows(result.finalBoard).slice(-1), ["OOOO.."]);
});

test("garbage adjacent to erased color puyos is cleared", () => {
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
    "RRR.O.",
  ]);

  const result = resolveTurn(
    board,
    { axis: COLORS.RED, child: COLORS.GREEN },
    { column: 3, orientation: ORIENTATIONS.RIGHT },
  );

  assert.equal(result.totalChains, 1);
  assert.deepEqual(boardToRows(result.finalBoard).slice(-1), ["....G."]);
  assert.equal(result.events[1].garbageCleared, 1);
});
