import { BOARD_WIDTH, ORIENTATIONS } from "../core/constants.js";

export const ACTION_KEYS = [];

for (let column = 0; column < BOARD_WIDTH; column += 1) {
  ACTION_KEYS.push(`${ORIENTATIONS.UP}:${column}`);
  ACTION_KEYS.push(`${ORIENTATIONS.DOWN}:${column}`);
}

for (let column = 0; column < BOARD_WIDTH - 1; column += 1) {
  ACTION_KEYS.push(`${ORIENTATIONS.RIGHT}:${column}`);
}

for (let column = 1; column < BOARD_WIDTH; column += 1) {
  ACTION_KEYS.push(`${ORIENTATIONS.LEFT}:${column}`);
}

export const ACTION_INDEX = Object.freeze(
  Object.fromEntries(ACTION_KEYS.map((actionKey, index) => [actionKey, index])),
);
