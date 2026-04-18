export const BOARD_WIDTH = 6;
export const BOARD_HEIGHT = 14;
export const VISIBLE_HEIGHT = 12;
export const HIDDEN_ROWS = BOARD_HEIGHT - VISIBLE_HEIGHT;
export const NEXT_PREVIEW_COUNT = 3;

export const COLORS = Object.freeze({
  EMPTY: ".",
  RED: "R",
  GREEN: "G",
  BLUE: "B",
  YELLOW: "Y",
});

export const PLAYABLE_COLORS = [
  COLORS.RED,
  COLORS.GREEN,
  COLORS.BLUE,
  COLORS.YELLOW,
];

export const ORIENTATIONS = Object.freeze({
  UP: "UP",
  RIGHT: "RIGHT",
  DOWN: "DOWN",
  LEFT: "LEFT",
});

export const ORIENTATION_ORDER = [
  ORIENTATIONS.UP,
  ORIENTATIONS.RIGHT,
  ORIENTATIONS.DOWN,
  ORIENTATIONS.LEFT,
];

export const COLOR_LABELS = Object.freeze({
  [COLORS.RED]: "赤",
  [COLORS.GREEN]: "緑",
  [COLORS.BLUE]: "青",
  [COLORS.YELLOW]: "黄",
});

export const EVENT_TYPES = Object.freeze({
  PLACE: "place",
  CLEAR: "clear",
  GRAVITY: "gravity",
  SETTLE: "settle",
});
