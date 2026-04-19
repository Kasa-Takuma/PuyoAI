import { boardToRows } from "../core/board.js";
import { PLAYABLE_COLORS } from "../core/constants.js";

const BOARD_HEIGHT = 14;
const BOARD_WIDTH = 6;
const BOARD_SYMBOLS = [".", "R", "G", "B", "Y"];
const VALUE_MODEL_URL = new URL("../../models/value_mlp.web.json", import.meta.url);
const DEFAULT_MAX_NEXT_PAIRS = 5;
const DEFAULT_VALUE_WEIGHT = 80_000;
const VALUE_TARGET_INDEX = Object.freeze({
  objective: 0,
  max_chain: 1,
  chains10_plus: 2,
  chains11_plus: 3,
  chains12_plus: 4,
  topout: 5,
});
const FALLBACK_FEATURE_KEYS = Object.freeze([
  "stackCells",
  "maxHeight",
  "hiddenCells",
  "dangerCells",
  "surfaceRoughness",
  "steepWalls",
  "valleyPenalty",
  "adjacency",
  "group2Count",
  "group3Count",
  "surfaceExtendableGroup2Count",
  "surfaceReadyGroup3Count",
  "isolatedSingles",
  "colorBalance",
  "columnsUsed",
  "bestVirtualChain",
  "bestVirtualScore",
  "virtualChainCount2Plus",
  "virtualChainCount3Plus",
  "topVirtualChainSum",
  "topVirtualScoreSum",
]);
const FALLBACK_FEATURE_SCALES = Object.freeze({
  stackCells: 78,
  maxHeight: 13,
  hiddenCells: 6,
  dangerCells: 18,
  surfaceRoughness: 48,
  steepWalls: 24,
  valleyPenalty: 24,
  adjacency: 78,
  group2Count: 18,
  group3Count: 18,
  surfaceExtendableGroup2Count: 18,
  surfaceReadyGroup3Count: 18,
  isolatedSingles: 40,
  colorBalance: 1,
  columnsUsed: 6,
  bestVirtualChain: 14,
  bestVirtualScore: 300_000,
  virtualChainCount2Plus: 48,
  virtualChainCount3Plus: 24,
  topVirtualChainSum: 36,
  topVirtualScoreSum: 600_000,
});

const BOARD_SYMBOL_INDEX = Object.freeze(
  Object.fromEntries(BOARD_SYMBOLS.map((symbol, index) => [symbol, index])),
);
const PLAYABLE_COLOR_INDEX = Object.freeze(
  Object.fromEntries(PLAYABLE_COLORS.map((symbol, index) => [symbol, index])),
);

let valueModelPromise = null;

function gelu(value) {
  const cubic = value * value * value;
  return (
    0.5 *
    value *
    (1 + Math.tanh(0.7978845608028654 * (value + 0.044715 * cubic)))
  );
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

function scaledFeature(features, key, featureScales) {
  const rawValue = Number(features?.[key] ?? 0);
  const scale = Number(featureScales?.[key] ?? FALLBACK_FEATURE_SCALES[key] ?? 1);
  const value = rawValue / Math.max(scale, 1e-6);
  return Math.max(-4, Math.min(4, value));
}

function encodeFeatures(features, featureKeys, featureScales) {
  return Float32Array.from(
    featureKeys.map((key) => scaledFeature(features, key, featureScales)),
  );
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
    maxNextPairs: model.maxNextPairs ?? DEFAULT_MAX_NEXT_PAIRS,
    targetNames: model.targetNames ?? Object.keys(VALUE_TARGET_INDEX),
    featureKeys: model.featureKeys ?? [...FALLBACK_FEATURE_KEYS],
    featureScales: model.featureScales ?? FALLBACK_FEATURE_SCALES,
    layers: model.layers.map((layer) => ({
      ...layer,
      weights: Float32Array.from(layer.weights),
      bias: Float32Array.from(layer.bias),
    })),
  };
}

export function normalizeValueAssistSettings(settings = {}) {
  const parsedWeight = Number.parseFloat(settings.valueWeight);
  return {
    useValueModel:
      settings.useValueModel === true ||
      settings.useValueModel === "true" ||
      settings.useValueModel === "on",
    valueWeight: Math.max(
      0,
      Math.min(
        1_000_000,
        Number.isFinite(parsedWeight) ? parsedWeight : DEFAULT_VALUE_WEIGHT,
      ),
    ),
  };
}

export async function loadSearchValueModel() {
  if (!valueModelPromise) {
    valueModelPromise = fetch(VALUE_MODEL_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load value model: ${response.status}`);
        }
        return response.json();
      })
      .then((model) => hydrateModel(model));
  }

  return valueModelPromise;
}

export function encodeValueInput({
  board,
  currentPair,
  nextQueue = [],
  turn = 0,
  totalScore = 0,
  features = {},
  model,
}) {
  const featureKeys = model?.featureKeys ?? FALLBACK_FEATURE_KEYS;
  const featureScales = model?.featureScales ?? FALLBACK_FEATURE_SCALES;
  const maxNextPairs = model?.maxNextPairs ?? DEFAULT_MAX_NEXT_PAIRS;

  return concatenateVectors([
    encodeBoardRows(boardToRows(board)),
    encodePair(currentPair),
    encodeNextQueue(nextQueue, maxNextPairs),
    Float32Array.from([
      Math.min(Number(turn || 0) / 1000, 4),
      Math.min(Number(totalScore || 0) / 2_000_000, 4),
    ]),
    encodeFeatures(features, featureKeys, featureScales),
  ]);
}

export function evaluateValueModel({
  model,
  board,
  currentPair,
  nextQueue = [],
  turn = 0,
  totalScore = 0,
  features = {},
}) {
  if (!model) {
    return null;
  }

  const input = encodeValueInput({
    board,
    currentPair,
    nextQueue,
    turn,
    totalScore,
    features,
    model,
  });
  const outputs = forward(model, input);
  const targetIndex =
    model.targetNames?.indexOf("objective") ?? VALUE_TARGET_INDEX.objective;
  const objectiveIndex = targetIndex >= 0 ? targetIndex : VALUE_TARGET_INDEX.objective;

  return {
    inputDim: input.length,
    outputs,
    objective: outputs[objectiveIndex] ?? 0,
    maxChain: (outputs[VALUE_TARGET_INDEX.max_chain] ?? 0) * 14,
    chains10Plus: (outputs[VALUE_TARGET_INDEX.chains10_plus] ?? 0) * 3,
    chains11Plus: (outputs[VALUE_TARGET_INDEX.chains11_plus] ?? 0) * 2,
    chains12Plus: outputs[VALUE_TARGET_INDEX.chains12_plus] ?? 0,
    topoutRisk: outputs[VALUE_TARGET_INDEX.topout] ?? 0,
  };
}
