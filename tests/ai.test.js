import test from "node:test";
import assert from "node:assert/strict";

import {
  createAiSnapshot,
  createChainFocusTrainingSample,
  createPolicyTrainingSample,
  createSlimPolicyTrainingSample,
  createValueTrainingSample,
} from "../src/ai/dataset.js";
import { extractBoardFeatures } from "../src/ai/features.js";
import { searchBestMove } from "../src/ai/search.js";
import { encodeValueInput } from "../src/ai/value.js";
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

test("search AI accepts the v9b search profile", () => {
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
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v9b" },
  });

  assert.equal(analysis.objective, "chain_builder_v9b");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v9b");
});

test("search AI accepts the v10 search profile", () => {
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
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v10" },
  });

  assert.equal(analysis.objective, "chain_builder_v10");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v10");
});

test("search AI accepts the v11 search profile", () => {
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
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v11" },
  });

  assert.equal(analysis.objective, "chain_builder_v11");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v11");
});

test("search AI accepts the v12 search profile", () => {
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
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v12" },
  });

  assert.equal(analysis.objective, "chain_builder_v12");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v12");
});

test("search AI accepts the v13 search profile", () => {
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
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v13" },
  });

  assert.equal(analysis.objective, "chain_builder_v13");
  assert.equal(analysis.settings.searchProfile, "chain_builder_v13");
});

test("v12AC prefers a reachable one-chain all clear", () => {
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
    "RR....",
  ]);
  const currentPair = {
    axis: COLORS.RED,
    child: COLORS.RED,
  };

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v12_ac" },
  });
  const result = resolveTurn(board, currentPair, analysis.bestAction);

  assert.equal(analysis.objective, "chain_builder_v12_ac");
  assert.equal(result.totalChains, 1);
  assert.equal(result.allClear, true);
});

test("search AI accepts a temporary tuned profile config", () => {
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
    settings: {
      depth: 1,
      beamWidth: 24,
      searchProfile: "chain_builder_v9b",
      profileConfig: {
        id: "test_tuned_v9b",
        baseProfileId: "chain_builder_v9b",
        turnWeights: { elevenPlusBonus: 360_000 },
        boardWeights: { bestVirtualChain: 940 },
        bonusScales: { v9b: 1.05 },
      },
    },
  });

  assert.equal(analysis.objective, "test_tuned_v9b");
  assert.equal(analysis.settings.searchProfile, "test_tuned_v9b");
  assert.equal(analysis.settings.baseSearchProfile, "chain_builder_v9b");
  assert.equal(analysis.settings.profileConfig.baseProfileId, "chain_builder_v9b");
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

test("value sample includes immediate and future labels", () => {
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
    settings: { depth: 1, beamWidth: 24, searchProfile: "chain_builder_v11" },
  });
  const snapshot = createAiSnapshot({
    presetId: "doubleChain",
    seed: "value-seed",
    turn: 3,
    totalScore: 120,
    board,
    currentPair,
    nextQueue: [],
  });

  const sample = createValueTrainingSample({
    snapshot,
    analysis,
    workerId: 1,
    gameSeed: "batch:worker-1:game-1",
    features: { stackCells: 6, bestVirtualChain: 2 },
    immediate: { chains: 0, score: 0, topout: false, actionKey: "RIGHT:3" },
    future: {
      12: { complete: true, stepsObserved: 12, maxChains: 8, chains10Plus: 0 },
      24: { complete: true, stepsObserved: 24, maxChains: 11, chains10Plus: 1 },
      48: { complete: false, stepsObserved: 30, maxChains: 11, chains10Plus: 1 },
    },
  });

  assert.equal(sample.kind, "search_value");
  assert.equal(sample.context.searchProfile, "chain_builder_v11");
  assert.equal(sample.future[24].maxChains, 11);
  assert.equal(sample.search.bestActionKey, analysis.bestActionKey);
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

test("sampling settings are echoed and default off", () => {
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

  assert.equal(analysis.settings.sampleCount, 0);
  assert.equal(analysis.settings.dedupe, true);
  assert.equal(analysis.sampling, null);
});

test("sampled search is deterministic", () => {
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
  const settings = {
    depth: 2,
    beamWidth: 8,
    searchProfile: "chain_builder_v13",
    sampleCount: 3,
    sampleDepth: 3,
    sampleBeamWidth: 4,
    sampleTopK: 4,
  };

  const first = searchBestMove({ board, currentPair, nextQueue, settings });
  const second = searchBestMove({ board, currentPair, nextQueue, settings });

  assert.equal(first.bestActionKey, second.bestActionKey);
  assert.equal(first.bestScore, second.bestScore);
  assert.equal(first.candidates[0].sampleScore, second.candidates[0].sampleScore);
  assert.ok(first.sampling.evaluatedCandidates >= 1);
});

test("dedupe does not lose the best action and reduces node count", () => {
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
  const currentPair = {
    axis: COLORS.BLUE,
    child: COLORS.YELLOW,
  };
  const nextQueue = [
    { axis: COLORS.RED, child: COLORS.GREEN },
    { axis: COLORS.GREEN, child: COLORS.BLUE },
  ];

  const withDedupe = searchBestMove({
    board,
    currentPair,
    nextQueue,
    settings: { depth: 3, beamWidth: 24, dedupe: true },
  });
  const withoutDedupe = searchBestMove({
    board,
    currentPair,
    nextQueue,
    settings: { depth: 3, beamWidth: 24, dedupe: false },
  });

  assert.ok(withDedupe.expandedNodeCount <= withoutDedupe.expandedNodeCount);
  assert.ok(withDedupe.bestAction !== null);
  assert.ok(withoutDedupe.bestAction !== null);
});

test("rollout survives a near-topout board", () => {
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
  const currentPair = {
    axis: COLORS.BLUE,
    child: COLORS.YELLOW,
  };

  const analysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24, sampleCount: 2, sampleDepth: 2 },
  });
  const result = resolveTurn(board, currentPair, analysis.bestAction);

  assert.equal(result.topout, false);
});

test("sample refine settings are echoed", () => {
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

  const defaultAnalysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { depth: 1, beamWidth: 24 },
  });

  assert.equal(defaultAnalysis.settings.sampleRefineLeaf, false);
  assert.equal(defaultAnalysis.settings.sampleValueWeight, 0);

  const explicitAnalysis = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: {
      depth: 1,
      beamWidth: 24,
      sampleRefineLeaf: true,
      sampleValueWeight: 250,
    },
  });

  assert.equal(explicitAnalysis.settings.sampleRefineLeaf, true);
  assert.equal(explicitAnalysis.settings.sampleValueWeight, 250);
});

test("refined sampled search is deterministic and returns a legal action", () => {
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
  const settings = {
    depth: 2,
    beamWidth: 8,
    searchProfile: "chain_builder_v13",
    sampleCount: 2,
    sampleDepth: 2,
    sampleBeamWidth: 4,
    sampleTopK: 3,
    sampleRefineLeaf: true,
  };

  const first = searchBestMove({ board, currentPair, nextQueue, settings });
  const second = searchBestMove({ board, currentPair, nextQueue, settings });

  assert.equal(first.bestActionKey, second.bestActionKey);
  assert.equal(first.bestScore, second.bestScore);
  assert.doesNotThrow(() => resolveTurn(board, currentPair, first.bestAction));
});

test("sample value weight uses the provided value model", () => {
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

  const featureKeys = ["stackCells", "maxHeight"];
  const featureScales = { stackCells: 78, maxHeight: 13 };
  const probeModel = { maxNextPairs: 1, featureKeys, featureScales };
  const probeInput = encodeValueInput({
    board,
    currentPair,
    nextQueue: [],
    turn: 0,
    totalScore: 0,
    features: {},
    model: probeModel,
  });
  const inputDim = probeInput.length;
  const weights = new Array(inputDim).fill(0);
  weights[inputDim - 1] = 1;
  weights[inputDim - 2] = 1;
  const valueModel = {
    maxNextPairs: 1,
    targetNames: ["objective"],
    featureKeys,
    featureScales,
    layers: [
      {
        inputDim,
        outputDim: 1,
        activation: "linear",
        weights,
        bias: [5],
      },
    ],
  };

  const settings = {
    depth: 1,
    beamWidth: 24,
    searchProfile: "chain_builder_v13",
    sampleCount: 1,
    sampleDepth: 1,
    sampleBeamWidth: 4,
    sampleTopK: 2,
    sampleSeed: "value-refine-seed",
  };

  const withWeight = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { ...settings, sampleValueWeight: 1000 },
    valueModel,
  });
  const withoutWeight = searchBestMove({
    board,
    currentPair,
    nextQueue: [],
    settings,
    valueModel,
  });

  assert.equal(withWeight.sampling.sampleValueWeight, 1000);
  assert.equal(withoutWeight.sampling.sampleValueWeight, 0);
  assert.notEqual(
    withWeight.candidates[0].sampleScore,
    withoutWeight.candidates[0].sampleScore,
  );
});
