import { boardFromRows, createEmptyBoard } from "./board.js";
import { COLORS, ORIENTATIONS } from "./constants.js";

function pair(axis, child) {
  return { axis, child };
}

export const PRESETS = {
  sandbox: {
    id: "sandbox",
    name: "Sandbox",
    description: "空盤面からシード付きで積み始める標準モードです。",
    board: createEmptyBoard(),
    currentPair: pair(COLORS.RED, COLORS.GREEN),
    nextQueue: [
      pair(COLORS.BLUE, COLORS.YELLOW),
      pair(COLORS.GREEN, COLORS.RED),
      pair(COLORS.YELLOW, COLORS.BLUE),
    ],
  },
  singleChain: {
    id: "singleChain",
    name: "1 Chain Demo",
    description: "右向きに置くと 1 連鎖 40 点になります。",
    board: boardFromRows([
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
      "RRR...",
    ]),
    currentPair: pair(COLORS.RED, COLORS.GREEN),
    nextQueue: [
      pair(COLORS.BLUE, COLORS.YELLOW),
      pair(COLORS.GREEN, COLORS.RED),
    ],
    suggestedAction: {
      column: 3,
      orientation: ORIENTATIONS.RIGHT,
    },
  },
  doubleChain: {
    id: "doubleChain",
    name: "2 Chain Demo",
    description: "3 列目に縦置きすると 2 連鎖 360 点になります。",
    board: boardFromRows([
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
    ]),
    currentPair: pair(COLORS.RED, COLORS.GREEN),
    nextQueue: [
      pair(COLORS.BLUE, COLORS.YELLOW),
      pair(COLORS.GREEN, COLORS.RED),
    ],
    suggestedAction: {
      column: 3,
      orientation: ORIENTATIONS.UP,
    },
  },
  topout: {
    id: "topout",
    name: "Topout Demo",
    description: "見えない 2 行に両方入ると即敗北します。",
    board: boardFromRows([
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
    ]),
    currentPair: pair(COLORS.BLUE, COLORS.YELLOW),
    nextQueue: [
      pair(COLORS.GREEN, COLORS.RED),
      pair(COLORS.YELLOW, COLORS.BLUE),
    ],
    suggestedAction: {
      column: 0,
      orientation: ORIENTATIONS.UP,
    },
  },
};
