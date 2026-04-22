import { BOARD_HEIGHT, BOARD_WIDTH, COLORS, ORIENTATIONS } from "../core/constants.js";

const PPSIM_TO_PUYOAI_COLOR = Object.freeze({
  0: COLORS.EMPTY,
  1: COLORS.RED,
  2: COLORS.BLUE,
  3: COLORS.GREEN,
  4: COLORS.YELLOW,
  5: COLORS.GARBAGE,
});

const PLAYABLE_PPSIM_COLORS = new Set([1, 2, 3, 4]);

function convertBoardColor(color) {
  return PPSIM_TO_PUYOAI_COLOR[color] ?? COLORS.EMPTY;
}

function convertPlayableColor(color) {
  if (!PLAYABLE_PPSIM_COLORS.has(color)) {
    return null;
  }
  return convertBoardColor(color);
}

export function convertBoard(ppsimBoard) {
  return Array.from({ length: BOARD_HEIGHT }, (_, y) =>
    Array.from({ length: BOARD_WIDTH }, (_, x) =>
      convertBoardColor(ppsimBoard?.[y]?.[x] ?? 0),
    ),
  );
}

export function convertCurrentPair(currentPuyo) {
  if (!currentPuyo) {
    return null;
  }

  const axis = convertPlayableColor(currentPuyo.mainColor);
  const child = convertPlayableColor(currentPuyo.subColor);
  if (!axis || !child) {
    return null;
  }

  return { axis, child };
}

export function convertNextQueue(upcomingPairs = []) {
  return upcomingPairs
    .map((pair) => {
      const sub = Array.isArray(pair) ? pair[0] : null;
      const main = Array.isArray(pair) ? pair[1] : null;
      const axis = convertPlayableColor(main);
      const child = convertPlayableColor(sub);
      return axis && child ? { axis, child } : null;
    })
    .filter(Boolean);
}

export function convertActionToPpsimPlacement(action) {
  if (!action) {
    return null;
  }

  const rotationByOrientation = {
    [ORIENTATIONS.UP]: 0,
    [ORIENTATIONS.LEFT]: 1,
    [ORIENTATIONS.DOWN]: 2,
    [ORIENTATIONS.RIGHT]: 3,
  };
  const rotation = rotationByOrientation[action.orientation];

  if (rotation === undefined) {
    return null;
  }

  return {
    mainX: action.column,
    rotation,
  };
}

export function summarizeAction(action) {
  if (!action) {
    return "no action";
  }
  return `${action.orientation}:${action.column + 1}`;
}
