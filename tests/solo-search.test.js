import test from "node:test";
import assert from "node:assert/strict";

import { searchSoloMove, soloChainUtility } from "../src/ai/solo-search.js";
import {
  createZeroSoloValueModel,
  encodeSoloValueInput,
  evaluateSoloValueInput,
  hydrateSoloValueModel,
  SOLO_VALUE_INPUT_DIM,
} from "../src/ai/solo-value.js";
import { createEmptyBoard } from "../src/core/board.js";
import { COLORS } from "../src/core/constants.js";

function pair(axis, child) {
  return { axis, child };
}

test("solo chain utility follows the fixed experiment reward", () => {
  assert.deepEqual(
    Array.from({ length: 16 }, (_, chains) => soloChainUtility(chains)),
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1.25, 1.5, 1.75, 2, 2],
  );
});

test("solo value input is 438 values in column-major cell order", () => {
  const board = createEmptyBoard();
  board[0][0] = COLORS.RED;
  board[1][0] = COLORS.GREEN;
  const input = encodeSoloValueInput({
    board,
    nextQueue: [pair(COLORS.BLUE, COLORS.YELLOW)],
  });

  assert.equal(input.length, SOLO_VALUE_INPUT_DIM);
  assert.equal(input[1], 1);
  assert.equal(input[5 + 2], 1);
  assert.equal(input[420 + 2], 1);
  assert.equal(input[420 + 4 + 3], 1);
  assert.equal(input[420 + 8], 1);
  assert.equal(input[420 + 9 + 8], 0);
  assert.equal(input.reduce((sum, value) => sum + value, 0), 84 + 3);
});

test("solo value ReLU MLP produces value and calibrated death probability", () => {
  const raw = {
    inputDim: SOLO_VALUE_INPUT_DIM,
    layers: [
      {
        inputDim: SOLO_VALUE_INPUT_DIM,
        outputDim: 64,
        activation: "relu",
        weights: new Array(SOLO_VALUE_INPUT_DIM * 64).fill(0),
        bias: new Array(64).fill(0),
      },
      {
        inputDim: 64,
        outputDim: 32,
        activation: "relu",
        weights: new Array(64 * 32).fill(0),
        bias: new Array(32).fill(0),
      },
      {
        inputDim: 32,
        outputDim: 2,
        activation: "linear",
        weights: new Array(32 * 2).fill(0),
        bias: [1.5, 0],
      },
    ],
  };
  const prediction = evaluateSoloValueInput(
    hydrateSoloValueModel(raw),
    new Float32Array(SOLO_VALUE_INPUT_DIM),
  );
  assert.equal(prediction.value, 1.5);
  assert.equal(prediction.deathProbability, 0.5);
});

test("solo search sees only the current pair and NEXT2", () => {
  const board = createEmptyBoard();
  const model = createZeroSoloValueModel();
  const currentPair = pair(COLORS.RED, COLORS.GREEN);
  const visible = [
    pair(COLORS.BLUE, COLORS.YELLOW),
    pair(COLORS.RED, COLORS.BLUE),
  ];
  const first = searchSoloMove({
    board,
    currentPair,
    nextQueue: [...visible, pair(COLORS.YELLOW, COLORS.YELLOW)],
    model,
  });
  const second = searchSoloMove({
    board,
    currentPair,
    nextQueue: [...visible, pair(COLORS.GREEN, COLORS.GREEN)],
    model,
  });

  assert.equal(first.bestActionKey, second.bestActionKey);
  assert.equal(first.bestScore, second.bestScore);
  assert.deepEqual(first.candidates, second.candidates);
});

test("solo search keeps a candidate for every living first action", () => {
  const analysis = searchSoloMove({
    board: createEmptyBoard(),
    currentPair: pair(COLORS.RED, COLORS.GREEN),
    nextQueue: [
      pair(COLORS.BLUE, COLORS.YELLOW),
      pair(COLORS.RED, COLORS.BLUE),
    ],
    settings: { beamWidth: 32 },
    model: createZeroSoloValueModel(),
  });

  assert.equal(analysis.kind, "solo-search");
  assert.equal(analysis.settings.depth, 3);
  assert.equal(analysis.candidates.length, 22);
  assert.ok(analysis.bestAction);
  assert.ok(analysis.expandedNodeCount > 22);
});
