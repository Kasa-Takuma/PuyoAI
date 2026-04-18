import { ACTION_INDEX, ACTION_KEYS } from "./action-vocab.js";
import { boardToRows, encodeAction, enumerateLegalActions } from "../core/board.js";
import { PLAYABLE_COLORS } from "../core/constants.js";

const BOARD_HEIGHT = 14;
const BOARD_WIDTH = 6;
const BOARD_SYMBOLS = [".", "R", "G", "B", "Y"];
const MODEL_URL = new URL("../../models/policy_mlp.web.json", import.meta.url);
const DEFAULT_SETTINGS = Object.freeze({
  depth: 3,
  beamWidth: 24,
});

const BOARD_SYMBOL_INDEX = Object.freeze(
  Object.fromEntries(BOARD_SYMBOLS.map((symbol, index) => [symbol, index])),
);
const PLAYABLE_COLOR_INDEX = Object.freeze(
  Object.fromEntries(PLAYABLE_COLORS.map((symbol, index) => [symbol, index])),
);

let learnedPolicyPromise = null;

function gelu(value) {
  const cubic = value * value * value;
  return (
    0.5 *
    value *
    (1 + Math.tanh(0.7978845608028654 * (value + 0.044715 * cubic)))
  );
}

function normalizeSettings(settings = {}) {
  return {
    depth: Math.max(1, Math.min(4, Number.parseInt(settings.depth, 10) || DEFAULT_SETTINGS.depth)),
    beamWidth: Math.max(
      4,
      Math.min(96, Number.parseInt(settings.beamWidth, 10) || DEFAULT_SETTINGS.beamWidth),
    ),
  };
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

function encodeSettings(settings) {
  return new Float32Array([settings.depth / 4, settings.beamWidth / 96]);
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

function encodePolicyInput({ board, currentPair, nextQueue = [], settings = {}, maxNextPairs = 5 }) {
  const normalizedSettings = normalizeSettings(settings);
  return concatenateVectors([
    encodeBoardRows(boardToRows(board)),
    encodePair(currentPair),
    encodeNextQueue(nextQueue, maxNextPairs),
    encodeSettings(normalizedSettings),
  ]);
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const shifted = logits.map((value) => Math.exp(value - maxLogit));
  const total = shifted.reduce((sum, value) => sum + value, 0);
  return shifted.map((value) => value / total);
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

export async function loadLearnedPolicyModel() {
  if (!learnedPolicyPromise) {
    learnedPolicyPromise = fetch(MODEL_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load learned policy model: ${response.status}`);
        }
        return response.json();
      })
      .then((model) => hydrateModel(model));
  }

  return learnedPolicyPromise;
}

export async function analyzeLearnedMove({
  board,
  currentPair,
  nextQueue = [],
  settings = {},
}) {
  const startedAt = performance.now();
  const model = await loadLearnedPolicyModel();
  const normalizedSettings = normalizeSettings(settings);
  const input = encodePolicyInput({
    board,
    currentPair,
    nextQueue,
    settings: normalizedSettings,
    maxNextPairs: model.maxNextPairs ?? 5,
  });
  const logits = forward(model, input);
  const probabilities = softmax(logits);
  const legalActions = enumerateLegalActions(board, currentPair);

  const candidates = legalActions
    .map((action) => {
      const actionKey = encodeAction(action);
      const actionIndex = ACTION_INDEX[actionKey];
      return {
        action,
        actionKey,
        probability: probabilities[actionIndex] ?? 0,
        logit: logits[actionIndex] ?? Number.NEGATIVE_INFINITY,
      };
    })
    .sort((left, right) => right.probability - left.probability);

  const bestAction = candidates[0]?.action ?? legalActions[0] ?? null;
  const bestActionKey = bestAction ? encodeAction(bestAction) : ACTION_KEYS[0];

  return {
    kind: "learned",
    objective: model.objective ?? "learned_policy_mlp",
    settings: normalizedSettings,
    bestAction,
    bestActionKey,
    bestScore: candidates[0]?.probability ?? 0,
    candidates,
    expandedNodeCount: 0,
    candidateCount: candidates.length,
    elapsedMs: performance.now() - startedAt,
    modelName: model.name ?? "policy_mlp",
    inputDim: input.length,
  };
}
