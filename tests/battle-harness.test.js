import test from "node:test";
import assert from "node:assert/strict";

import { createEmptyBoard } from "../src/core/board.js";
import { BOARD_HEIGHT, BOARD_WIDTH, COLORS } from "../src/core/constants.js";
import { createRng } from "../src/core/randomizer.js";
import { dropOjamaToBoard, runMatch } from "../tools/battle-harness.js";

function countGarbage(board) {
  let count = 0;
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      if (board[y][x] === COLORS.GARBAGE) {
        count += 1;
      }
    }
  }
  return count;
}

test("runMatch is deterministic given the same seed", () => {
  const options = { seed: 5000, maxRounds: 60 };

  const first = runMatch(options);
  const second = runMatch(options);

  assert.deepEqual(first, second);
});

test("runMatch completes within the round cap and reports a winner or draw", () => {
  const result = runMatch({ seed: 7, maxRounds: 60 });

  assert.ok(["A", "B", "draw"].includes(result.winner));
  assert.ok(result.rounds >= 1 && result.rounds <= 60);
  assert.ok(Number.isFinite(result.aMaxChain));
  assert.ok(Number.isFinite(result.bMaxChain));
});

test("dropOjamaToBoard deterministically adds exactly the requested garbage count", () => {
  const boardA = createEmptyBoard();
  const boardB = createEmptyBoard();

  dropOjamaToBoard(boardA, 8, createRng("garbage-seed"));
  dropOjamaToBoard(boardB, 8, createRng("garbage-seed"));

  assert.deepEqual(boardA, boardB);
  assert.equal(countGarbage(boardA), 8);
});
