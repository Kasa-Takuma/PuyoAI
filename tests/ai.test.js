import test from "node:test";
import assert from "node:assert/strict";

import {
  createAiSnapshot,
  createChainFocusTrainingSample,
  createPolicyTrainingSample,
  createSlimPolicyTrainingSample,
} from "../src/ai/dataset.js";
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

test("search AI preserves the selected search profile in its analysis settings", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v4" },
  });

  assert.equal(analysis.objective, "chain_builder_v4");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v4");
});

test("search AI accepts the v5 search profile", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v5" },
  });

  assert.equal(analysis.objective, "chain_builder_v5");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v5");
});

test("search AI accepts the v6 search profile", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v6" },
  });

  assert.equal(analysis.objective, "chain_builder_v6");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v6");
});

test("search AI accepts the v7 search profile", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v7" },
  });

  assert.equal(analysis.objective, "chain_builder_v7");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v7");
});

test("search AI accepts the v7a search profile", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v7a" },
  });

  assert.equal(analysis.objective, "chain_builder_v7a");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v7a");
});

test("search AI accepts the v8 search profile", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v8" },
  });

  assert.equal(analysis.objective, "chain_builder_v8");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v8");
});

test("search AI accepts the v9 search profile", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v9" },
  });

  assert.equal(analysis.objective, "chain_builder_v9");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v9");
});

test("search AI accepts the v9a search profile", () => {
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

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v9a" },
  });

  assert.equal(analysis.objective, "chain_builder_v9a");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v9a");
});

test("slim policy sample keeps only lightweight supervision fields", () => {
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
  const nextQueue = [{ axis: COLORS.BLUE, child: COLORS.YELLOW }];
  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue,
    settings: { depth: 1, beamWidth: 24 },
  });
  const snapshot = createAiSnapshot({
    presetId: "doubleChain",
    seed: "slim-seed",
    turn: 3,
    totalScore: 0,
    board,
    currentPair,
    nextQueue,
  });

  const sample = createSlimPolicyTrainingSample(snapshot, analysis);

  assert.equal(sample.kind, "search_policy_slim");
  assert.equal(sample.bestActionKey, analysis.bestActionKey);
  assert.equal(sample.state.boardRows.length > 0, true);
  assert.equal(Array.isArray(sample.topCandidates), true);
  assert.equal("candidates" in sample, false);
});

test("chain focus sample includes trigger metadata", () => {
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
  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24 },
  });
  const snapshot = createAiSnapshot({
    presetId: "doubleChain",
    seed: "focus-seed",
    turn: 7,
    totalScore: 320,
    board,
    currentPair,
    nextQueue: [],
  });

  const sample = createChainFocusTrainingSample(snapshot, analysis, {
    workerId: 2,
    gameSeed: "batch:worker-2:game-5",
    triggerTurn: 9,
    triggerChains: 10,
    triggerScore: 12400,
    thresholdChains: 10,
    offsetFromTrigger: -2,
  });

  assert.equal(sample.kind, "search_policy_chain_focus");
  assert.equal(sample.focus.triggerChains, 10);
  assert.equal(sample.focus.thresholdChains, 10);
  assert.equal(sample.focus.offsetFromTrigger, -2);
  assert.equal(sample.focus.gameSeed, "batch:worker-2:game-5");
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
