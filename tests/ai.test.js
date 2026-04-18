import test from "node:test";
import assert from "node:assert/strict";

import { createAiSnapshot, createPolicyTrainingSample } from "../src/ai/dataset.js";
import { extractBoardFeatures } from "../src/ai/features.js";
import { searchBestMove } from "../src/ai/search.js";
import { boardFromRows } from "../src/core/board.js";
import { COLORS } from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";

test("search AI finds the obvious double-chain action", () => {
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
    "......",
    "GGGRRR",
  ]);
  const currentPair = {
    axis: COLORS.RED,
    child: COLORS.GREEN,
  };

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24 },
  });
  const result = resolveTurn(board, currentPair, analysis.bestAction);

  assert.equal(result.totalChains, 2);
  assert.equal(result.totalScore, 360);
});

test("search AI avoids immediate topout when a safe move exists", () => {
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
  const currentPair = {
    axis: COLORS.BLUE,
    child: COLORS.YELLOW,
  };

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24 },
  });
  const result = resolveTurn(board, currentPair, analysis.bestAction);

  assert.equal(result.topout, false);
});

test("search analysis can be serialized into a training sample", () => {
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
  const currentPair = {
    axis: COLORS.RED,
    child: COLORS.GREEN,
  };
  const nextQueue = [
    { axis: COLORS.BLUE, child: COLORS.YELLOW },
    { axis: COLORS.GREEN, child: COLORS.RED },
  ];

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue,
    settings: { depth: 2, beamWidth: 24 },
  });
  const snapshot = createAiSnapshot({
    presetId: "doubleChain",
    seed: "test-seed",
    turn: 4,
    totalScore: 120,
    board,
    currentPair,
    nextQueue,
  });
  const sample = createPolicyTrainingSample(snapshot, analysis);

  assert.equal(sample.search.objective, "chain_builder_v3");
  assert.equal(sample.bestActionKey, analysis.bestActionKey);
  assert.equal(sample.candidates.length, analysis.candidates.length);
  assert.equal(sample.state.turn, 4);
  assert.equal(sample.search.settings.depth, 2);
});

test("feature extraction sees the virtual double-chain trigger on the demo board", () => {
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

  const features = extractBoardFeatures(board);

  assert.equal(features.bestVirtualChain, 2);
  assert.ok(features.bestVirtualScore >= 360);
  assert.ok(features.virtualChainCount2Plus >= 1);
});
