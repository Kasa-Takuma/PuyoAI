import { BOARD_HEIGHT, BOARD_WIDTH, PLAYABLE_COLORS } from "../core/constants.js";
import { fromLegacyBoard, pairToCodes } from "../core/fast-board.js";

export const SOLO_VALUE_INPUT_DIM = BOARD_WIDTH * BOARD_HEIGHT * 5 + 2 * 9;
export const SOLO_VALUE_MODEL_URL = new URL(
  "../../models/solo_value.web.json",
  import.meta.url,
);

let modelPromise = null;

function sigmoid(value) {
  if (value >= 0) {
    const exponential = Math.exp(-value);
    return 1 / (1 + exponential);
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function validateLayer(layer, expectedInputDim) {
  if (!Number.isInteger(layer?.inputDim) || !Number.isInteger(layer?.outputDim)) {
    throw new Error("Solo value model layer dimensions must be integers.");
  }
  if (layer.inputDim !== expectedInputDim) {
    throw new Error(
      `Solo value model layer input mismatch: expected ${expectedInputDim}, got ${layer.inputDim}.`,
    );
  }
  if (layer.weights?.length !== layer.inputDim * layer.outputDim) {
    throw new Error("Solo value model layer has the wrong number of weights.");
  }
  if (layer.bias?.length !== layer.outputDim) {
    throw new Error("Solo value model layer has the wrong number of biases.");
  }
}

export function hydrateSoloValueModel(rawModel) {
  if (!rawModel || !Array.isArray(rawModel.layers) || rawModel.layers.length !== 3) {
    throw new Error("Solo value model must contain exactly three layers.");
  }
  if ((rawModel.inputDim ?? SOLO_VALUE_INPUT_DIM) !== SOLO_VALUE_INPUT_DIM) {
    throw new Error(
      `Solo value model input must be ${SOLO_VALUE_INPUT_DIM} values.`,
    );
  }

  let expectedInputDim = SOLO_VALUE_INPUT_DIM;
  const layers = rawModel.layers.map((layer, index) => {
    validateLayer(layer, expectedInputDim);
    if (index < 2 && layer.activation !== "relu") {
      throw new Error("Solo value model hidden layers must use ReLU.");
    }
    if (index === 2 && layer.activation !== "linear") {
      throw new Error("Solo value model output layer must be linear.");
    }
    expectedInputDim = layer.outputDim;
    return {
      ...layer,
      weights: Float32Array.from(layer.weights),
      bias: Float32Array.from(layer.bias),
    };
  });
  if (expectedInputDim !== 2) {
    throw new Error("Solo value model must produce value and death-logit outputs.");
  }

  return {
    ...rawModel,
    format: rawModel.format ?? "puyoai-solo-value-v1",
    inputDim: SOLO_VALUE_INPUT_DIM,
    maxNextPairs: 2,
    layers,
  };
}

export function encodeSoloValueInputFast(fastBoard, nextQueue = []) {
  const input = new Float32Array(SOLO_VALUE_INPUT_DIM);
  let offset = 0;

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < BOARD_HEIGHT; y += 1) {
      const state = fastBoard[base + y];
      if (state < 0 || state > 4) {
        throw new Error("Solo value model only supports empty cells and four colors.");
      }
      input[offset + state] = 1;
      offset += 5;
    }
  }

  for (let pairIndex = 0; pairIndex < 2; pairIndex += 1) {
    const pair = nextQueue[pairIndex];
    if (pair) {
      const codes = pairToCodes(pair);
      input[offset + codes.axis - 1] = 1;
      input[offset + 4 + codes.child - 1] = 1;
      input[offset + 8] = 1;
    }
    offset += 9;
  }

  return input;
}

export function encodeSoloValueInput({ board, nextQueue = [] }) {
  return encodeSoloValueInputFast(fromLegacyBoard(board), nextQueue);
}

function runLayer(input, layer) {
  const output = new Float32Array(layer.outputDim);
  for (let row = 0; row < layer.outputDim; row += 1) {
    let sum = layer.bias[row];
    const base = row * layer.inputDim;
    for (let column = 0; column < layer.inputDim; column += 1) {
      sum += layer.weights[base + column] * input[column];
    }
    output[row] = layer.activation === "relu" ? Math.max(0, sum) : sum;
  }
  return output;
}

export function evaluateSoloValueInput(model, input) {
  if (!model) {
    throw new Error("A solo value model is required.");
  }
  let activations = input;
  for (const layer of model.layers) {
    activations = runLayer(activations, layer);
  }
  return {
    value: activations[0],
    deathLogit: activations[1],
    deathProbability: sigmoid(activations[1]),
  };
}

export function evaluateSoloValueFast({ model, fastBoard, nextQueue = [] }) {
  return evaluateSoloValueInput(
    model,
    encodeSoloValueInputFast(fastBoard, nextQueue.slice(0, 2)),
  );
}

export function createZeroSoloValueModel(name = "solo_value_zero") {
  const dimensions = [SOLO_VALUE_INPUT_DIM, 64, 32, 2];
  return hydrateSoloValueModel({
    format: "puyoai-solo-value-v1",
    name,
    inputDim: SOLO_VALUE_INPUT_DIM,
    maxNextPairs: 2,
    targetHorizon: 128,
    layers: dimensions.slice(1).map((outputDim, index) => ({
      inputDim: dimensions[index],
      outputDim,
      activation: index < 2 ? "relu" : "linear",
      weights: new Array(dimensions[index] * outputDim).fill(0),
      bias: new Array(outputDim).fill(0),
    })),
  });
}

export async function loadSoloValueModel(url = SOLO_VALUE_MODEL_URL) {
  if (url !== SOLO_VALUE_MODEL_URL) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load solo value model: ${response.status}`);
    }
    return hydrateSoloValueModel(await response.json());
  }

  if (!modelPromise) {
    modelPromise = fetch(SOLO_VALUE_MODEL_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load solo value model: ${response.status}`);
        }
        return response.json();
      })
      .then(hydrateSoloValueModel);
  }
  return modelPromise;
}

export const SOLO_PLAYABLE_COLOR_COUNT = PLAYABLE_COLORS.length;
