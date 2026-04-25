#!/usr/bin/env node
import { once } from "node:events";
import { readFile } from "node:fs/promises";

import { ACTION_INDEX, ACTION_KEYS } from "../src/ai/action-vocab.js";
import { boardToRows, encodeAction, enumerateLegalActions } from "../src/core/board.js";
import { PLAYABLE_COLORS } from "../src/core/constants.js";
import { createRng } from "../src/core/randomizer.js";
import { applyAction, createGameState } from "../src/app/state.js";

const BOARD_HEIGHT = 14;
const BOARD_WIDTH = 6;
const BOARD_SYMBOLS = [".", "R", "G", "B", "Y"];
const BOARD_SYMBOL_INDEX = Object.freeze(
  Object.fromEntries(BOARD_SYMBOLS.map((symbol, index) => [symbol, index])),
);
const PLAYABLE_COLOR_INDEX = Object.freeze(
  Object.fromEntries(PLAYABLE_COLORS.map((symbol, index) => [symbol, index])),
);

const DEFAULT_REWARD_CONFIG = Object.freeze({
  survivalReward: 0.002,
  scoreScale: 100_000,
  scoreCap: 4.0,
  chainSquareScale: 80,
  allClearBonus: 0.8,
  topoutPenalty: 2.0,
});

function parseInteger(value, fallback, min = 0, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function parseFloatArg(value, fallback, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function parseArgs(argv) {
  const args = {
    policyModel: "models/policy_mlp.web.json",
    turns: 50_000,
    games: 0,
    seed: "policy-rl",
    presetId: "sandbox",
    temperature: 1.12,
    epsilon: 0.04,
    topK: 0,
    discount: 0.995,
    maxTurnsPerGame: 0,
    depthFeature: 3,
    beamWidthFeature: 24,
    reportEvery: 5_000,
    reward: { ...DEFAULT_REWARD_CONFIG },
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--policy-model") {
      args.policyModel = next || args.policyModel;
      index += 1;
    } else if (arg === "--turns") {
      args.turns = parseInteger(next, args.turns, 1);
      index += 1;
    } else if (arg === "--games") {
      args.games = parseInteger(next, args.games, 0);
      index += 1;
    } else if (arg === "--seed") {
      args.seed = next || args.seed;
      index += 1;
    } else if (arg === "--preset") {
      args.presetId = next || args.presetId;
      index += 1;
    } else if (arg === "--temperature") {
      args.temperature = parseFloatArg(next, args.temperature, 0.05, 10);
      index += 1;
    } else if (arg === "--epsilon") {
      args.epsilon = parseFloatArg(next, args.epsilon, 0, 1);
      index += 1;
    } else if (arg === "--top-k") {
      args.topK = parseInteger(next, args.topK, 0, ACTION_KEYS.length);
      index += 1;
    } else if (arg === "--discount") {
      args.discount = parseFloatArg(next, args.discount, 0, 1);
      index += 1;
    } else if (arg === "--max-turns-per-game") {
      args.maxTurnsPerGame = parseInteger(next, args.maxTurnsPerGame, 0);
      index += 1;
    } else if (arg === "--depth-feature") {
      args.depthFeature = parseInteger(next, args.depthFeature, 1, 4);
      index += 1;
    } else if (arg === "--beam-width-feature") {
      args.beamWidthFeature = parseInteger(next, args.beamWidthFeature, 4, 96);
      index += 1;
    } else if (arg === "--survival-reward") {
      args.reward.survivalReward = parseFloatArg(next, args.reward.survivalReward);
      index += 1;
    } else if (arg === "--score-scale") {
      args.reward.scoreScale = parseFloatArg(next, args.reward.scoreScale, 1);
      index += 1;
    } else if (arg === "--topout-penalty") {
      args.reward.topoutPenalty = parseFloatArg(next, args.reward.topoutPenalty, 0);
      index += 1;
    } else if (arg === "--report-every") {
      args.reportEvery = parseInteger(next, args.reportEvery, 1);
      index += 1;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  return args;
}

function printHelp() {
  console.log(`Usage:
  node tools/generate-rl-rollouts.js --policy-model models/policy_mlp.web.json [options] > rollouts.jsonl

Options:
  --policy-model PATH      Exported web policy JSON used for stochastic rollouts.
  --turns N                Total environment turns to generate. Default: 50000
  --games N                Stop after N games; 0 means keep starting games until turns are reached.
  --seed TEXT              Deterministic seed base. Default: policy-rl
  --temperature N          Softmax sampling temperature. Default: 1.12
  --epsilon N              Uniform exploration probability. Default: 0.04
  --top-k N                Sample only from the top N legal logits; 0 means all legal actions.
  --discount N             Return discount used inside each episode. Default: 0.995
  --max-turns-per-game N   Optional per-game cap; 0 means no extra cap.
  --depth-feature N        Encoded depth feature for checkpoint compatibility. Default: 3
  --beam-width-feature N   Encoded beam-width feature for checkpoint compatibility. Default: 24
  --survival-reward N      Tiny reward for a non-terminal placement. Default: 0.002
  --score-scale N          Divisor for score reward. Default: 100000
  --topout-penalty N       Penalty subtracted on topout. Default: 2
  --report-every N         Progress interval in generated turns. Default: 5000`);
}

function gelu(value) {
  const cubic = value * value * value;
  return (
    0.5 *
    value *
    (1 + Math.tanh(0.7978845608028654 * (value + 0.044715 * cubic)))
  );
}

function hydrateModel(model) {
  return {
    ...model,
    layers: model.layers.map((layer) => ({
      ...layer,
      weights: Float32Array.from(layer.weights),
      bias: Float32Array.from(layer.bias),
    })),
  };
}

async function loadPolicyModel(path) {
  const payload = JSON.parse(await readFile(path, "utf8"));
  if (!Array.isArray(payload.layers) || payload.layers.length === 0) {
    throw new Error(`Policy model has no exported layers: ${path}`);
  }
  return hydrateModel(payload);
}

function encodeBoardRows(boardRows) {
  const encoded = new Float32Array(BOARD_HEIGHT * BOARD_WIDTH * BOARD_SYMBOLS.length);
  let offset = 0;

  for (const row of boardRows) {
    for (const symbol of row) {
      encoded[offset + BOARD_SYMBOL_INDEX[symbol]] = 1;
      offset += BOARD_SYMBOLS.length;
    }
  }

  return encoded;
}

function encodePair(pair) {
  const vector = new Float32Array(PLAYABLE_COLORS.length * 2);
  vector[PLAYABLE_COLOR_INDEX[pair.axis]] = 1;
  vector[PLAYABLE_COLORS.length + PLAYABLE_COLOR_INDEX[pair.child]] = 1;
  return vector;
}

function encodeNextQueue(nextQueue, maxNextPairs) {
  const vector = new Float32Array(maxNextPairs * PLAYABLE_COLORS.length * 2);
  nextQueue.slice(0, maxNextPairs).forEach((pair, index) => {
    const base = index * PLAYABLE_COLORS.length * 2;
    vector[base + PLAYABLE_COLOR_INDEX[pair.axis]] = 1;
    vector[base + PLAYABLE_COLORS.length + PLAYABLE_COLOR_INDEX[pair.child]] = 1;
  });
  return vector;
}

function encodeSettings(depthFeature, beamWidthFeature) {
  return new Float32Array([depthFeature / 4, beamWidthFeature / 96]);
}

function concatenateVectors(vectors) {
  const totalLength = vectors.reduce((sum, vector) => sum + vector.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;

  for (const vector of vectors) {
    merged.set(vector, offset);
    offset += vector.length;
  }

  return merged;
}

function encodePolicyInput(state, model, args) {
  const input = concatenateVectors([
    encodeBoardRows(boardToRows(state.board)),
    encodePair(state.currentPair),
    encodeNextQueue(state.nextQueue, model.maxNextPairs ?? 5),
    encodeSettings(args.depthFeature, args.beamWidthFeature),
  ]);
  const expectedDim = model.layers[0]?.inputDim;
  if (expectedDim && input.length !== expectedDim) {
    throw new Error(
      `Policy input dimension mismatch: encoded ${input.length}, model expects ${expectedDim}`,
    );
  }
  return input;
}

function runLayer(input, layer) {
  const output = new Float32Array(layer.outputDim);
  const weights = layer.weights;
  const bias = layer.bias;

  for (let row = 0; row < layer.outputDim; row += 1) {
    let sum = bias[row];
    const base = row * layer.inputDim;
    for (let column = 0; column < layer.inputDim; column += 1) {
      sum += weights[base + column] * input[column];
    }
    output[row] = layer.activation === "gelu" ? gelu(sum) : sum;
  }

  return output;
}

function forward(model, input) {
  let activations = input;
  for (const layer of model.layers) {
    activations = runLayer(activations, layer);
  }
  return Array.from(activations);
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const shifted = logits.map((value) => Math.exp(value - maxLogit));
  const total = shifted.reduce((sum, value) => sum + value, 0);
  return shifted.map((value) => value / total);
}

function randomFloat(rng) {
  return rng.nextUint32() / 0x100000000;
}

function sampleFromDistribution(entries, rng) {
  const roll = randomFloat(rng);
  let cumulative = 0;
  for (const entry of entries) {
    cumulative += entry.probability;
    if (roll <= cumulative) {
      return entry;
    }
  }
  return entries.at(-1);
}

function buildActionDistribution({ model, state, args }) {
  const input = encodePolicyInput(state, model, args);
  const logits = forward(model, input);
  const legalActions = enumerateLegalActions(state.board, state.currentPair);
  const ranked = legalActions
    .map((action) => {
      const actionKey = encodeAction(action);
      return {
        action,
        actionKey,
        logit: logits[ACTION_INDEX[actionKey]] ?? Number.NEGATIVE_INFINITY,
      };
    })
    .sort((left, right) => right.logit - left.logit);
  const pool = args.topK > 0 ? ranked.slice(0, Math.max(1, args.topK)) : ranked;
  const policyProbs = softmax(pool.map((entry) => entry.logit / args.temperature));
  const uniformProb = 1 / Math.max(pool.length, 1);
  const distribution = pool.map((entry, index) => ({
    ...entry,
    probability: (1 - args.epsilon) * policyProbs[index] + args.epsilon * uniformProb,
  }));
  const entropy = -distribution.reduce(
    (sum, entry) => sum + entry.probability * Math.log(Math.max(entry.probability, 1e-12)),
    0,
  );

  return {
    distribution,
    entropy,
    legalActionKeys: legalActions.map((action) => encodeAction(action)),
  };
}

function clonePair(pair) {
  return { axis: pair.axis, child: pair.child };
}

function cloneAction(action) {
  return { column: action.column, orientation: action.orientation };
}

function snapshotState(state, args) {
  return {
    boardRows: boardToRows(state.board),
    currentPair: clonePair(state.currentPair),
    nextQueue: state.nextQueue.slice(0, 5).map((pair) => clonePair(pair)),
    turn: state.turn,
    totalScore: state.totalScore,
    settings: {
      depth: args.depthFeature,
      beamWidth: args.beamWidthFeature,
    },
  };
}

function chainThresholdReward(chains) {
  let reward = 0;
  if (chains >= 7) {
    reward += 0.6;
  }
  if (chains >= 10) {
    reward += 1.4;
  }
  if (chains >= 11) {
    reward += 1.4;
  }
  if (chains >= 12) {
    reward += 1.8;
  }
  if (chains >= 13) {
    reward += 2.5;
  }
  return reward;
}

function rewardForResult(result, rewardConfig) {
  const chains = result.totalChains;
  const scoreReward = Math.min(result.totalScore / rewardConfig.scoreScale, rewardConfig.scoreCap);
  const chainReward = chains >= 2 ? (chains * chains) / rewardConfig.chainSquareScale : 0;
  const allClearReward = result.allClear ? rewardConfig.allClearBonus : 0;
  const survivalReward = result.topout ? 0 : rewardConfig.survivalReward;
  const topoutPenalty = result.topout ? rewardConfig.topoutPenalty : 0;
  return (
    survivalReward +
    scoreReward +
    chainReward +
    chainThresholdReward(chains) +
    allClearReward -
    topoutPenalty
  );
}

function finalizeReturns(steps, discount) {
  let runningReturn = 0;
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    runningReturn = steps[index].reward + discount * runningReturn;
    steps[index].return = runningReturn;
  }
}

function summarizeEpisode(steps, state) {
  return {
    turns: steps.length,
    totalReward: steps.reduce((sum, step) => sum + step.reward, 0),
    discountedReturn: steps[0]?.return ?? 0,
    totalScore: state.totalScore,
    maxChains: Math.max(0, ...steps.map((step) => step.immediate.chains)),
    chainEvents: steps.filter((step) => step.immediate.chains > 0).length,
    chains7Plus: steps.filter((step) => step.immediate.chains >= 7).length,
    chains10Plus: steps.filter((step) => step.immediate.chains >= 10).length,
    chains11Plus: steps.filter((step) => step.immediate.chains >= 11).length,
    chains12Plus: steps.filter((step) => step.immediate.chains >= 12).length,
    chains13Plus: steps.filter((step) => step.immediate.chains >= 13).length,
    topout: state.gameOver,
  };
}

async function writeJsonLine(payload) {
  const line = `${JSON.stringify(payload)}\n`;
  if (!process.stdout.write(line)) {
    await once(process.stdout, "drain");
  }
}

function reportProgress(payload) {
  process.stderr.write(`${JSON.stringify(payload)}\n`);
}

async function runEpisode({ model, args, gameIndex, actionRng, totalTurnsRemaining }) {
  const gameSeed = `${args.seed}:game-${gameIndex}`;
  const state = createGameState({
    presetId: args.presetId,
    seed: gameSeed,
    aiSettings: {
      depth: args.depthFeature,
      beamWidth: args.beamWidthFeature,
    },
  });
  const steps = [];
  const maxTurns =
    args.maxTurnsPerGame > 0
      ? Math.min(args.maxTurnsPerGame, totalTurnsRemaining)
      : totalTurnsRemaining;

  while (!state.gameOver && steps.length < maxTurns) {
    const stateSnapshot = snapshotState(state, args);
    const actionDistribution = buildActionDistribution({ model, state, args });
    if (actionDistribution.distribution.length === 0) {
      break;
    }

    const selected = sampleFromDistribution(actionDistribution.distribution, actionRng);
    const result = applyAction(state, selected.action, "policy-rl");
    if (!result) {
      break;
    }

    steps.push({
      state: stateSnapshot,
      action: cloneAction(selected.action),
      actionKey: selected.actionKey,
      legalActionKeys: actionDistribution.legalActionKeys,
      behaviorProbability: selected.probability,
      behaviorLogProb: Math.log(Math.max(selected.probability, 1e-12)),
      behaviorEntropy: actionDistribution.entropy,
      reward: rewardForResult(result, args.reward),
      immediate: {
        chains: result.totalChains,
        score: result.totalScore,
        topout: result.topout,
        allClear: result.allClear,
      },
    });
  }

  finalizeReturns(steps, args.discount);
  return {
    kind: "policy_rl_episode",
    version: 1,
    generator: "puyoai-policy-rl-rollout-v1",
    createdAt: new Date().toISOString(),
    seed: gameSeed,
    gameIndex,
    policy: {
      modelPath: args.policyModel,
      modelName: model.name ?? null,
      objective: model.objective ?? "learned_policy_mlp",
      temperature: args.temperature,
      epsilon: args.epsilon,
      topK: args.topK,
      discount: args.discount,
      depthFeature: args.depthFeature,
      beamWidthFeature: args.beamWidthFeature,
      reward: args.reward,
    },
    summary: summarizeEpisode(steps, state),
    steps,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const model = await loadPolicyModel(args.policyModel);
  const actionRng = createRng(`${args.seed}:actions`);
  const startedAt = performance.now();
  let totalTurns = 0;
  let gameIndex = 0;
  let nextReportTurn = args.reportEvery;
  let bestChain = 0;
  let topouts = 0;
  let totalReward = 0;

  reportProgress({
    stage: "start",
    policyModel: args.policyModel,
    turns: args.turns,
    games: args.games,
    seed: args.seed,
    temperature: args.temperature,
    epsilon: args.epsilon,
    topK: args.topK,
    discount: args.discount,
  });

  while (totalTurns < args.turns && (args.games === 0 || gameIndex < args.games)) {
    gameIndex += 1;
    const episode = await runEpisode({
      model,
      args,
      gameIndex,
      actionRng,
      totalTurnsRemaining: args.turns - totalTurns,
    });
    if (episode.steps.length === 0) {
      break;
    }

    totalTurns += episode.steps.length;
    bestChain = Math.max(bestChain, episode.summary.maxChains);
    topouts += episode.summary.topout ? 1 : 0;
    totalReward += episode.summary.totalReward;
    await writeJsonLine(episode);

    if (totalTurns >= nextReportTurn) {
      reportProgress({
        stage: "progress",
        totalTurns,
        game: gameIndex,
        bestChain,
        topouts,
        totalReward,
        elapsedMs: Math.round(performance.now() - startedAt),
      });
      nextReportTurn += args.reportEvery;
    }
  }

  reportProgress({
    stage: "complete",
    totalTurns,
    gamesStarted: gameIndex,
    bestChain,
    topouts,
    totalReward,
    elapsedMs: Math.round(performance.now() - startedAt),
  });
}

main().catch((error) => {
  reportProgress({
    stage: "error",
    message: error instanceof Error ? error.message : String(error),
  });
  process.exitCode = 1;
});
