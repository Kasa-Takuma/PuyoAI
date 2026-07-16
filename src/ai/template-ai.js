// Lightweight beam-search AI modeled on gata272/puyoAI2 (puyoAI.js /
// puyo-ai-worker.js), reusing the fast bitboard core (src/core/fast-board.js)
// for the search hot path instead of puyoAI2's 2D-array + string-key
// approach. The opening book lives in template-opening-book.js.
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COLORS,
  ORIENTATIONS,
  PLAYABLE_COLORS,
  STORAGE_HEIGHT,
  TOP_OUT_COLUMN,
  TOP_OUT_ROW,
  VISIBLE_HEIGHT,
} from "../core/constants.js";
import { encodeAction, enumerateLegalActions, isBoardEmpty } from "../core/board.js";
import {
  FAST_COLORS,
  colorToCode,
  fastBoardHash,
  fastColumnHeights,
  fastEnumerateLegalActions,
  fastResolveTurn,
  fromLegacyBoard,
  pairToCodes,
} from "../core/fast-board.js";
import { buildOpeningPlan } from "./template-opening-book.js";

const FAST_EMPTY = FAST_COLORS.EMPTY;
const FAST_GARBAGE = FAST_COLORS.GARBAGE;
const NEIGHBOR_OFFSETS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

const DEFAULT_BEAM_WIDTH = 14;

function clampBeamWidth(value) {
  return Math.max(4, Math.min(48, Number.parseInt(value, 10) || DEFAULT_BEAM_WIDTH));
}

// Incoming-garbage (予告おじゃま) mechanics. These mirror ppsim2/puyoSim.js's
// live rules exactly (see consumeOjamaForAttack / dropOjamaToBoard /
// ALL_CLEAR_SCORE_BONUS there) so the search can look ahead through the same
// offset-then-drop sequence the battle page applies after every move.
const NUISANCE_TARGET_POINTS = 70;
const MAX_OJAMA_DROP_PER_TURN = 30;
const ALL_CLEAR_ATTACK_BONUS = 2100;
// 70 * 8, kept proportional to chainOutcomeValue's totalScore * 8 term: the
// offsetting benefit itself already shows up through the cleaner simulated
// board, so this only rewards damage that actually reaches the opponent.
const OJAMA_SENT_BONUS = 560;
// puyoAI2's worker penalized pendingOjama * 60 as a danger term; here it only
// fires for pending that survives the simulated drop (pending > 30).
const OJAMA_DANGER_PENALTY = -60;
// 凝視 (opponent board reading): a strong penalty, not a hard veto like
// topout/lethal's -1e15, since the opponent's ready fire is only a threat -
// it may never actually come.
const OPPONENT_LETHAL_PENALTY = -500000;

// 攻撃タイミング判断 (attack timing): attacking a near-topout / already-buried
// opponent is worth more, since they're less able to shrug off the garbage.
const OFFENSE_TOPOUT_HEIGHT_THRESHOLD = 9; // TOP_OUT_ROW is 11; 9 is "close"
const OFFENSE_TOPOUT_BONUS = 1.0;
const OFFENSE_MAX_HEIGHT_THRESHOLD = 10;
const OFFENSE_MAX_HEIGHT_BONUS = 0.5;
const OFFENSE_PENDING_BONUS = 0.5; // already sitting on unresolved garbage
const OFFENSE_MULTIPLIER_CAP = 2.5;
// とどめ (kill detection): a large bonus for a fire whose outgoing garbage,
// stacked on the opponent's own pending, would top them out outright.
const OPPONENT_KILL_BONUS = 800000;

const TEMPLATE_LIBRARY = [
  { mask: [1, 1, 1, 1, 0, 0], profile: [0, 1, 2, 3, 0, 0], weight: 1.0 },
  { mask: [0, 0, 1, 1, 1, 1], profile: [0, 0, 3, 2, 1, 0], weight: 1.0 },
  { mask: [1, 1, 1, 1, 1, 0], profile: [0, 1, 2, 2, 1, 0], weight: 1.3 },
  { mask: [0, 1, 1, 1, 1, 1], profile: [0, 1, 2, 2, 1, 0], weight: 1.3 },
  { mask: [1, 1, 1, 1, 1, 1], profile: [2, 1, 0, 0, 1, 2], weight: 1.1 },
  { mask: [0, 1, 1, 1, 1, 0], profile: [0, 1, 2, 3, 2, 1], weight: 1.05 },
  { mask: [1, 1, 1, 1, 1, 1], profile: [1, 2, 1, 1, 2, 1], weight: 0.95 },
];

function computeHeights(fastBoard) {
  const heights = new Array(BOARD_WIDTH);
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    let height = 0;
    for (let y = VISIBLE_HEIGHT - 1; y >= 0; y -= 1) {
      if (fastBoard[base + y] !== FAST_EMPTY) {
        height = y + 1;
        break;
      }
    }
    heights[x] = height;
  }
  return heights;
}

function countHoles(fastBoard, heights) {
  let holes = 0;
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let y = 0; y < heights[x]; y += 1) {
      if (fastBoard[base + y] === FAST_EMPTY) {
        holes += 1;
      }
    }
  }
  return holes;
}

function templateScore(heights) {
  let best1 = 0;
  let best2 = 0;

  for (const template of TEMPLATE_LIBRARY) {
    const maskedCols = [];
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      if (template.mask[x]) {
        maskedCols.push(x);
      }
    }
    if (maskedCols.length === 0) {
      continue;
    }

    let base = Infinity;
    for (const x of maskedCols) {
      base = Math.min(base, heights[x] - template.profile[x]);
    }

    let s = 0;
    let occupied = 0;
    for (const x of maskedCols) {
      const target = base + template.profile[x];
      const diff = Math.abs(heights[x] - target);
      s += Math.max(0, 8 - diff * 3);
      if (heights[x] > 0) {
        occupied += 1;
      }
    }
    s += occupied * 2;
    s *= template.weight;

    if (s > best1) {
      best2 = best1;
      best1 = s;
    } else if (s > best2) {
      best2 = s;
    }
  }

  return best1 + best2 * 0.5;
}

function findLooseGroups(fastBoard) {
  const visited = new Uint8Array(BOARD_WIDTH * VISIBLE_HEIGHT);
  const groups = [];

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    for (let y = 0; y < VISIBLE_HEIGHT; y += 1) {
      const visitedIndex = x * VISIBLE_HEIGHT + y;
      if (visited[visitedIndex]) {
        continue;
      }
      const color = fastBoard[x * BOARD_HEIGHT + y];
      if (color === FAST_EMPTY || color === FAST_GARBAGE) {
        visited[visitedIndex] = 1;
        continue;
      }

      const stack = [[x, y]];
      visited[visitedIndex] = 1;
      const cells = [];

      while (stack.length > 0) {
        const [cx, cy] = stack.pop();
        cells.push([cx, cy]);

        for (const [dx, dy] of NEIGHBOR_OFFSETS) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= VISIBLE_HEIGHT) {
            continue;
          }
          const neighborVisitedIndex = nx * VISIBLE_HEIGHT + ny;
          if (visited[neighborVisitedIndex] || fastBoard[nx * BOARD_HEIGHT + ny] !== color) {
            continue;
          }
          visited[neighborVisitedIndex] = 1;
          stack.push([nx, ny]);
        }
      }

      groups.push({ color, cells });
    }
  }

  return groups;
}

function openNeighborCount(fastBoard, cells) {
  const seen = new Set();
  let count = 0;

  for (const [x, y] of cells) {
    for (const [dx, dy] of NEIGHBOR_OFFSETS) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= VISIBLE_HEIGHT) {
        continue;
      }
      if (fastBoard[nx * BOARD_HEIGHT + ny] !== FAST_EMPTY) {
        continue;
      }
      const key = nx * VISIBLE_HEIGHT + ny;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      count += 1;
    }
  }

  return count;
}

function seedScore(fastBoard) {
  const groups = findLooseGroups(fastBoard);
  let s = 0;

  for (const group of groups) {
    const size = group.cells.length;
    if (size === 1) {
      s += 1;
    } else if (size === 2) {
      s += 12 + openNeighborCount(fastBoard, group.cells) * 2;
    } else if (size === 3) {
      s += 35 + openNeighborCount(fastBoard, group.cells) * 4;
    }
  }

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    for (let y = 0; y < VISIBLE_HEIGHT; y += 1) {
      const c = fastBoard[x * BOARD_HEIGHT + y];
      if (c === FAST_EMPTY || c === FAST_GARBAGE) {
        continue;
      }

      if (
        x + 2 < BOARD_WIDTH &&
        fastBoard[(x + 1) * BOARD_HEIGHT + y] === c &&
        fastBoard[(x + 2) * BOARD_HEIGHT + y] === c
      ) {
        const leftEmpty = x - 1 >= 0 && fastBoard[(x - 1) * BOARD_HEIGHT + y] === FAST_EMPTY;
        const rightEmpty = x + 3 < BOARD_WIDTH && fastBoard[(x + 3) * BOARD_HEIGHT + y] === FAST_EMPTY;
        if (leftEmpty || rightEmpty) {
          s += 16;
        }
      }

      if (
        y + 2 < VISIBLE_HEIGHT &&
        fastBoard[x * BOARD_HEIGHT + y + 1] === c &&
        fastBoard[x * BOARD_HEIGHT + y + 2] === c
      ) {
        const belowEmpty = y - 1 >= 0 && fastBoard[x * BOARD_HEIGHT + y - 1] === FAST_EMPTY;
        const aboveEmpty = y + 3 < VISIBLE_HEIGHT && fastBoard[x * BOARD_HEIGHT + y + 3] === FAST_EMPTY;
        if (belowEmpty || aboveEmpty) {
          s += 16;
        }
      }

      if (x + 1 < BOARD_WIDTH && y + 1 < VISIBLE_HEIGHT) {
        const right = fastBoard[(x + 1) * BOARD_HEIGHT + y];
        const up = fastBoard[x * BOARD_HEIGHT + y + 1];
        if (right === c && up === c) {
          s += 20;
        }
      }
    }
  }

  return s;
}

function groupBonuses(fastBoard) {
  const groups = findLooseGroups(fastBoard);
  let s = 0;

  for (const group of groups) {
    const size = group.cells.length;
    if (size === 2) {
      s += 10;
    } else if (size === 3) {
      s += 30 + openNeighborCount(fastBoard, group.cells) * 3;
    } else if (size >= 5) {
      s += Math.min(80, size * 8);
    }
  }

  return s;
}

function dangerPenalty(fastBoard, heights) {
  let penalty = 0;

  if (fastBoard[TOP_OUT_COLUMN * BOARD_HEIGHT + TOP_OUT_ROW] !== FAST_EMPTY) {
    penalty += 1000000;
  }
  if (heights[TOP_OUT_COLUMN] >= TOP_OUT_ROW + 1) {
    penalty += 250000;
  }
  if (heights[TOP_OUT_COLUMN] >= TOP_OUT_ROW - 1) {
    penalty += 80000;
  }
  for (let y = Math.max(0, TOP_OUT_ROW - 2); y <= TOP_OUT_ROW; y += 1) {
    if (fastBoard[TOP_OUT_COLUMN * BOARD_HEIGHT + y] !== FAST_EMPTY) {
      penalty += 25000;
    }
  }

  return penalty;
}

function colorBalance(fastBoard) {
  const counts = new Map();
  for (const color of PLAYABLE_COLORS) {
    counts.set(colorToCode(color), 0);
  }
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    for (let y = 0; y < VISIBLE_HEIGHT; y += 1) {
      const c = fastBoard[x * BOARD_HEIGHT + y];
      if (counts.has(c)) {
        counts.set(c, counts.get(c) + 1);
      }
    }
  }
  const sorted = [...counts.values()].sort((a, b) => b - a);
  return 0.6 * (sorted[0] + sorted[1]) - 0.8 * (sorted[2] + sorted[3]);
}

function evaluateBoard(fastBoard) {
  const heights = computeHeights(fastBoard);
  const holes = countHoles(fastBoard, heights);
  const maxHeight = Math.max(...heights);
  let bumpiness = 0;
  for (let x = 1; x < BOARD_WIDTH; x += 1) {
    bumpiness += Math.abs(heights[x] - heights[x - 1]);
  }

  let s = 0;
  s += templateScore(heights) * 18;
  s += seedScore(fastBoard) * 10;
  s += groupBonuses(fastBoard);
  s -= holes * 38;
  s -= bumpiness * 10;
  s -= maxHeight * 30;
  s -= dangerPenalty(fastBoard, heights);

  if (maxHeight >= 9) {
    s -= 120;
  }
  if (maxHeight >= 10) {
    s -= 260;
  }

  s += colorBalance(fastBoard);

  return s;
}

function chainOutcomeValue(result) {
  return (
    Math.pow(result.totalChains, 2.15) * 32000 +
    result.totalScore * 8 +
    (result.allClear ? 250000 : 0)
  );
}

function attackFromResult(result) {
  return Math.floor((result.totalScore + (result.allClear ? ALL_CLEAR_ATTACK_BONUS : 0)) / NUISANCE_TARGET_POINTS);
}

// Simulates ppsim2's post-move offset-then-drop sequence on a fast board:
// `attack` first cancels out `pendingOjama` (consumeOjamaForAttack), then up
// to MAX_OJAMA_DROP_PER_TURN of whatever remains lands on the board
// (dropOjamaToBoard) before the next piece. The real game shuffles columns
// randomly per round of 6; this is a deterministic expectation approximation
// of that spread — floor(drop / 6) onto every column, plus the remainder
// one-each onto the columns currently lowest (ties broken leftmost), with any
// per-column overflow redistributed to the next lowest non-full column.
export function simulateOjamaSettle(fastBoard, pendingOjama, attack) {
  const pending = Math.max(0, pendingOjama | 0);
  const atk = Math.max(0, attack | 0);
  const canceled = Math.min(pending, atk);
  const outgoing = atk - canceled;
  const remaining = pending - canceled;
  const drop = Math.min(remaining, MAX_OJAMA_DROP_PER_TURN);
  const pendingAfter = remaining - drop;

  if (drop === 0) {
    return { board: fastBoard, pendingAfter, outgoing, lethal: false };
  }

  const heights = fastColumnHeights(fastBoard);
  let emptyCells = 0;
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    emptyCells += BOARD_HEIGHT - heights[x];
  }

  const working = new Uint8Array(fastBoard);

  if (drop > emptyCells) {
    return { board: working, pendingAfter, outgoing, lethal: true };
  }

  const evenShare = Math.floor(drop / BOARD_WIDTH);
  const remainder = drop % BOARD_WIDTH;
  const counts = new Array(BOARD_WIDTH).fill(evenShare);
  const lowestFirst = [...Array(BOARD_WIDTH).keys()].sort((a, b) => heights[a] - heights[b] || a - b);
  for (let i = 0; i < remainder; i += 1) {
    counts[lowestFirst[i]] += 1;
  }

  let overflow = 0;
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const room = BOARD_HEIGHT - heights[x];
    if (counts[x] > room) {
      overflow += counts[x] - room;
      counts[x] = room;
    }
  }
  while (overflow > 0) {
    let bestX = -1;
    let bestFilledHeight = Infinity;
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const filledHeight = heights[x] + counts[x];
      if (filledHeight < BOARD_HEIGHT && filledHeight < bestFilledHeight) {
        bestFilledHeight = filledHeight;
        bestX = x;
      }
    }
    if (bestX === -1) {
      break;
    }
    counts[bestX] += 1;
    overflow -= 1;
  }

  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    const base = x * BOARD_HEIGHT;
    for (let i = 0; i < counts[x]; i += 1) {
      working[base + heights[x] + i] = FAST_GARBAGE;
    }
  }

  const lethal = working[TOP_OUT_COLUMN * BOARD_HEIGHT + TOP_OUT_ROW] !== FAST_EMPTY;

  return { board: working, pendingAfter, outgoing, lethal };
}

// 凝視: don't sit in a position where the opponent's ready fire kills us.
// Only probes (an extra simulateOjamaSettle call) when opponentThreat > 0, so
// the solo (no-opponent-read) path pays zero extra cost and is untouched.
function opponentThreatPenalty(settle, opponentThreat) {
  if (opponentThreat <= 0) {
    return 0;
  }
  const survivalProbe = simulateOjamaSettle(settle.board, settle.pendingAfter + opponentThreat, 0);
  return survivalProbe.lethal ? OPPONENT_LETHAL_PENALTY : 0;
}

// 攻撃タイミング判断: scales how much a fire's outgoing garbage is worth by how
// vulnerable the opponent already looks (computed once per analyzeTemplateMove
// call from their board reading, not per node). Exactly 1 when there's no
// opponent read, so the solo path is unaffected.
function computeOffenseMultiplier(opponentFastBoard, opponentPendingOjama) {
  const oppHeights = fastColumnHeights(opponentFastBoard);
  const oppTopoutColumnHeight = oppHeights[TOP_OUT_COLUMN];
  const oppMaxHeight = Math.max(...oppHeights);

  let multiplier = 1;
  if (oppTopoutColumnHeight >= OFFENSE_TOPOUT_HEIGHT_THRESHOLD) {
    multiplier += OFFENSE_TOPOUT_BONUS;
  }
  if (oppMaxHeight >= OFFENSE_MAX_HEIGHT_THRESHOLD) {
    multiplier += OFFENSE_MAX_HEIGHT_BONUS;
  }
  if (opponentPendingOjama > 0) {
    multiplier += OFFENSE_PENDING_BONUS;
  }
  return Math.min(OFFENSE_MULTIPLIER_CAP, multiplier);
}

// とどめ (kill detection): if this fire's outgoing garbage, stacked on top of
// whatever the opponent already has pending, would top them out, add a large
// bonus. This ignores the opponent's own move entirely (no game-tree search
// over their side) — a heuristic approximation, not a guarantee they can't
// offset it by firing their own chain first (phase-2 scope).
function opponentKillBonus(settle, opponentFastBoard, opponentPendingOjama) {
  if (!opponentFastBoard || settle.outgoing <= 0) {
    return 0;
  }
  const killProbe = simulateOjamaSettle(opponentFastBoard, opponentPendingOjama + settle.outgoing, 0);
  return killProbe.lethal ? OPPONENT_KILL_BONUS : 0;
}

// 副砲 (sub-chain) weight: values a second, independent fire found during the
// leaf probe below — i.e. counter-fire readiness while the main chain still
// stands, distinct from (and lighter than) the main chain's own 0.9 weight.
const SUB_FIRE_WEIGHT = 0.35;

// Faithful to puyoAI2's pseudoLeafScore: probes every column with a
// monochrome PAIR (not a single puyo) of each playable color, dropped
// straight onto the fast board (no legacy conversion / cloning). mainProbe is
// the best resulting chain value, a proxy for the board's chain potential;
// subProbe is the best probe far enough away (>=2 columns) and a different
// color from mainProbe, a crude independence check so the same chain isn't
// counted twice through a second trigger cell. bestAttack is the largest
// attackFromResult() among all probes, used for 凝視 (opponent threat reads)
// rather than leafValue's own chain-shape scoring.
function virtualFireProbes(fastBoard) {
  const heights = fastColumnHeights(fastBoard);

  const probes = [];
  let mainProbe = { value: 0, column: -1, code: -1 };
  let bestAttack = 0;
  for (const color of PLAYABLE_COLORS) {
    const code = colorToCode(color);
    for (let column = 0; column < BOARD_WIDTH; column += 1) {
      if (heights[column] >= VISIBLE_HEIGHT) {
        continue;
      }
      const result = fastResolveTurn(fastBoard, code, code, { column, orientation: ORIENTATIONS.UP });
      if (result.topout) {
        continue;
      }
      const probe = { value: chainOutcomeValue(result), column, code };
      probes.push(probe);
      if (probe.value > mainProbe.value) {
        mainProbe = probe;
      }
      const attack = attackFromResult(result);
      if (attack > bestAttack) {
        bestAttack = attack;
      }
    }
  }

  let subProbe = null;
  for (const probe of probes) {
    if (probe === mainProbe || probe.code === mainProbe.code || Math.abs(probe.column - mainProbe.column) < 2) {
      continue;
    }
    if (!subProbe || probe.value > subProbe.value) {
      subProbe = probe;
    }
  }

  return { mainProbe, subProbe, bestAttack };
}

function leafValue(fastBoard) {
  const base = evaluateBoard(fastBoard);
  const { mainProbe, subProbe } = virtualFireProbes(fastBoard);
  return base + 0.9 * mainProbe.value + SUB_FIRE_WEIGHT * (subProbe?.value ?? 0);
}

function cloneRootAction(action) {
  return { column: action.column, orientation: action.orientation };
}

// Only the root candidate's best-known score can improve as deeper levels
// resolve; ties keep the first (higher-priority root action order) value.
function updateRootBest(candidates, rootIndex, value) {
  if (value > candidates[rootIndex].searchScore) {
    candidates[rootIndex].searchScore = value;
  }
}

// Caps the number of expensive leafValue() calls (each ~24 virtual chain
// probes) to the highest-potential survivors, mirroring puyoAI2's
// LEAF_PSEUDO_BRANCH_LIMIT = 8. The rest fall back to the cheap
// shaped + evaluateBoard score.
const LEAF_FULL_EVAL_LIMIT = 8;

function scoreLeafFrontier(candidates, leafFrontier, beamWidth) {
  leafFrontier.sort((a, b) => (b.shaped + b.evalValue) - (a.shaped + a.evalValue));
  const fullEvalCount = Math.min(LEAF_FULL_EVAL_LIMIT, beamWidth);

  leafFrontier.forEach((entry, index) => {
    // leafValue() runs its own fresh evaluateBoard() internally, so the
    // danger and opponent-threat penalties (already folded into the cheap
    // entry.evalValue) have to be re-added alongside it for the full-eval
    // branch.
    const value =
      index < fullEvalCount
        ? entry.shaped + leafValue(entry.board) + entry.pendingAfter * OJAMA_DANGER_PENALTY + entry.threatPenalty
        : entry.shaped + entry.evalValue;
    updateRootBest(candidates, entry.rootIndex, value);
  });
}

// Level-synchronized global beam search: instead of running an independent
// beamWidth-wide search per depth-1 survivor (the old per-node recursion,
// which expanded roughly beamWidth^2 leaves), every level keeps a single
// global top-beamWidth frontier across all parents. `shaped` on a frontier
// entry accumulates evaluateBoard(board) * 0.25 for every non-final level
// the entry has passed through *before* the current one (the level a node
// terminates or leafs out on never adds its own shaped term), matching the
// score composition of the original recursive search exactly.
//
// Battle awareness: every non-topout move first settles `pendingOjama`
// against that move's own attack (simulateOjamaSettle) before the resulting
// board is evaluated or expanded further — mirroring ppsim2's offset-then-
// drop sequence. `pending` on a frontier entry is the incoming garbage still
// queued for that node (root entries start from the payload's pendingOjama;
// deeper levels carry their parent's settle.pendingAfter forward). No *new*
// incoming attack is modeled during the lookahead, only the initial pending
// draining through, since future opponent attacks aren't knowable.
//
// 凝視: `opponentThreat` (the opponent's best immediate attack, read once per
// analyzeTemplateMove call, not per node) applies opponentThreatPenalty to
// every node's evalValue when > 0, discouraging positions that would top out
// if that threat landed on top of our own pending garbage.
//
// 攻撃タイミング判断: `offenseMultiplier` scales a fire's outgoing-garbage bonus
// (1 with no opponent read); `opponentFastBoard`/`opponentPendingOjama` (also
// read once per call) feed the per-fire kill-detection bonus.
function runBeamSearch({
  board,
  currentPair,
  nextQueue,
  settings,
  pendingOjama = 0,
  opponentThreat = 0,
  offenseMultiplier = 1,
  opponentFastBoard = null,
  opponentPendingOjama = 0,
}) {
  const pieces = [currentPair, ...nextQueue].slice(0, 3);
  const beamWidth = clampBeamWidth(settings.templateBeamWidth);
  const fastRoot = fromLegacyBoard(board);
  const { axis, child } = pairToCodes(currentPair);
  const rootActions = fastEnumerateLegalActions(fastRoot, axis, child);
  const stats = { expandedNodeCount: 0 };
  const candidates = [];
  const isRootFinal = pieces.length === 1;
  const rootPending = Math.max(0, pendingOjama | 0);

  let frontier = [];
  const leafFrontier = [];

  for (const rawAction of rootActions) {
    const action = cloneRootAction(rawAction);
    const result = fastResolveTurn(fastRoot, axis, child, action);
    stats.expandedNodeCount += 1;

    const rootIndex = candidates.length;
    const candidate = {
      action,
      actionKey: encodeAction(action),
      searchScore: -Infinity,
      immediateChains: result.totalChains,
      immediateScore: result.totalScore,
      immediateTopout: result.topout,
      immediateAllClear: result.allClear,
      immediateAttack: 0,
      immediateOutgoing: 0,
    };
    candidates.push(candidate);

    if (result.topout) {
      updateRootBest(candidates, rootIndex, -1e15);
      continue;
    }

    const attack = attackFromResult(result);
    const settle = simulateOjamaSettle(result.board, rootPending, attack);
    candidate.immediateAttack = attack;
    candidate.immediateOutgoing = settle.outgoing;

    if (settle.lethal) {
      updateRootBest(candidates, rootIndex, -1e15);
      continue;
    }

    // Chain fires no longer end the line (puyoAI2's semantics): a fired
    // chain's realized value is banked into `shaped` and the node continues,
    // so "fire a small counter chain now, keep the main chain for later" is
    // visible to the search. `banked` is 0 for a non-firing move.
    const settledEval = evaluateBoard(settle.board);
    const firing = result.totalChains > 0;
    const banked = firing
      ? chainOutcomeValue(result) +
        settle.outgoing * OJAMA_SENT_BONUS * offenseMultiplier +
        opponentKillBonus(settle, opponentFastBoard, opponentPendingOjama)
      : 0;
    const threatPenalty = opponentThreatPenalty(settle, opponentThreat);
    const evalValue = settledEval + settle.pendingAfter * OJAMA_DANGER_PENALTY + threatPenalty;

    if (firing) {
      // Pruning floor: identical to the old terminal-fire formula, so a fire
      // whose continuation gets pruned out of the beam still scores as before.
      updateRootBest(candidates, rootIndex, banked + settledEval * 0.1);
    } else {
      // Fallback score (level-0 quickValue) in case this root's whole subtree
      // is later pruned out of the global beam.
      updateRootBest(candidates, rootIndex, evalValue);
    }

    const shaped = banked;
    if (isRootFinal) {
      leafFrontier.push({ board: settle.board, rootIndex, shaped, evalValue, pendingAfter: settle.pendingAfter, threatPenalty });
    } else {
      frontier.push({ board: settle.board, rootIndex, shaped: shaped + evalValue * 0.25, evalValue, pending: settle.pendingAfter });
    }
  }

  for (let depth = 1; depth < pieces.length && frontier.length > 0; depth += 1) {
    const isFinalLevel = depth === pieces.length - 1;
    const { axis: levelAxis, child: levelChild } = pairToCodes(pieces[depth]);

    const expanded = [];
    for (const parent of frontier) {
      const actions = fastEnumerateLegalActions(parent.board, levelAxis, levelChild);
      for (const action of actions) {
        const result = fastResolveTurn(parent.board, levelAxis, levelChild, action);
        stats.expandedNodeCount += 1;

        if (result.topout) {
          updateRootBest(candidates, parent.rootIndex, -1e15);
          continue;
        }

        const attack = attackFromResult(result);
        const settle = simulateOjamaSettle(result.board, parent.pending, attack);

        if (settle.lethal) {
          updateRootBest(candidates, parent.rootIndex, -1e15);
          continue;
        }

        // Same non-terminal-firing treatment as the root level (see above):
        // a fire banks its realized value into `shaped` and continues.
        const settledEval = evaluateBoard(settle.board);
        const firing = result.totalChains > 0;
        const banked = firing
          ? chainOutcomeValue(result) +
            settle.outgoing * OJAMA_SENT_BONUS * offenseMultiplier +
            opponentKillBonus(settle, opponentFastBoard, opponentPendingOjama)
          : 0;
        if (firing) {
          updateRootBest(
            candidates,
            parent.rootIndex,
            parent.shaped + banked + settledEval * 0.1,
          );
        }

        const shaped = parent.shaped + banked;
        const threatPenalty = opponentThreatPenalty(settle, opponentThreat);
        const evalValue = settledEval + settle.pendingAfter * OJAMA_DANGER_PENALTY + threatPenalty;
        const quickValue = evalValue + chainOutcomeValue(result) * 0.01;
        expanded.push({
          board: settle.board,
          rootIndex: parent.rootIndex,
          shaped,
          evalValue,
          pending: settle.pendingAfter,
          threatPenalty,
          sortValue: shaped + quickValue,
          hash: fastBoardHash(settle.board),
        });
      }
    }

    const bestByHash = new Map();
    for (const entry of expanded) {
      const existing = bestByHash.get(entry.hash);
      if (!existing || entry.shaped > existing.shaped) {
        bestByHash.set(entry.hash, entry);
      }
    }
    const deduped = [...bestByHash.values()];
    deduped.sort((a, b) => b.sortValue - a.sortValue);
    const survivors = deduped.slice(0, beamWidth);

    if (isFinalLevel) {
      for (const entry of survivors) {
        leafFrontier.push({
          board: entry.board,
          rootIndex: entry.rootIndex,
          shaped: entry.shaped,
          evalValue: entry.evalValue,
          pendingAfter: entry.pending,
          threatPenalty: entry.threatPenalty,
        });
      }
      frontier = [];
    } else {
      frontier = survivors.map((entry) => ({
        board: entry.board,
        rootIndex: entry.rootIndex,
        shaped: entry.shaped + entry.evalValue * 0.25,
        evalValue: entry.evalValue,
        pending: entry.pending,
      }));
    }
  }

  scoreLeafFrontier(candidates, leafFrontier, beamWidth);

  candidates.sort((a, b) => b.searchScore - a.searchScore);

  return {
    bestAction: candidates[0]?.action ?? null,
    bestScore: candidates[0]?.searchScore ?? -Infinity,
    candidates,
    expandedNodeCount: stats.expandedNodeCount,
  };
}

let openingState = { plan: null, movesUsed: 0, active: false };

export function resetTemplateOpeningState() {
  openingState = { plan: null, movesUsed: 0, active: false };
}

function countOccupiedCells(board) {
  let count = 0;
  for (let y = 0; y < STORAGE_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      if (board[y][x] !== COLORS.EMPTY) {
        count += 1;
      }
    }
  }
  return count;
}

function actionsEqual(a, b) {
  return a.column === b.column && a.orientation === b.orientation;
}

export function analyzeTemplateMove({
  board,
  currentPair,
  nextQueue = [],
  settings = {},
  pendingOjama = 0,
  opponent = null,
}) {
  const startedAt = performance.now();

  if (!currentPair) {
    return {
      kind: "template",
      opening: false,
      patternKey: null,
      bestAction: null,
      bestActionKey: null,
      bestScore: -Infinity,
      candidates: [],
      candidateCount: 0,
      expandedNodeCount: 0,
      opponent: null,
      elapsedMs: performance.now() - startedAt,
    };
  }

  // 凝視 / 攻撃タイミング判断: read the opponent's board once per call (not per
  // search node) for their best immediate attack and how vulnerable they
  // already look. Malformed/absent opponent data degrades to "no read"
  // rather than throwing - battle robustness over strictness.
  let opponentThreat = 0;
  let offenseMultiplier = 1;
  let opponentFastBoard = null;
  let opponentPendingOjama = 0;
  let opponentResult = null;
  if (opponent && opponent.board) {
    try {
      opponentFastBoard = fromLegacyBoard(opponent.board);
      opponentThreat = virtualFireProbes(opponentFastBoard).bestAttack;
      opponentPendingOjama = opponent.pendingOjama | 0;
      offenseMultiplier = computeOffenseMultiplier(opponentFastBoard, opponentPendingOjama);
      opponentResult = { threat: opponentThreat, offenseMultiplier };
    } catch {
      opponentThreat = 0;
      offenseMultiplier = 1;
      opponentFastBoard = null;
      opponentPendingOjama = 0;
      opponentResult = null;
    }
  }

  // Fixed opening plans assume a quiet, garbage-free start; under incoming
  // pressure the beam search (which is battle-aware) should decide instead.
  const openingBookEligible = pendingOjama <= 0;

  if (openingBookEligible && isBoardEmpty(board) && nextQueue.length >= 2) {
    const plan = buildOpeningPlan([currentPair, nextQueue[0], nextQueue[1]]);
    openingState = { plan, movesUsed: 0, active: plan !== null };
  }

  let openingAction = null;
  if (openingBookEligible && openingState.active && openingState.movesUsed < 3) {
    const plannedAction = openingState.plan.actions[openingState.movesUsed];
    const occupied = countOccupiedCells(board);
    const legalActions = enumerateLegalActions(board, currentPair);
    const isLegal = legalActions.some((action) => actionsEqual(action, plannedAction));
    if (occupied === openingState.movesUsed * 2 && isLegal) {
      openingAction = plannedAction;
      openingState.movesUsed += 1;
    } else {
      openingState.active = false;
    }
  }
  if (openingState.movesUsed >= 3) {
    openingState.active = false;
  }

  const beamResult = runBeamSearch({
    board,
    currentPair,
    nextQueue,
    settings,
    pendingOjama,
    opponentThreat,
    offenseMultiplier,
    opponentFastBoard,
    opponentPendingOjama,
  });

  const opening = openingAction !== null;
  let bestAction;
  let bestScore;
  let candidates;

  if (opening) {
    bestAction = openingAction;
    const openingKey = encodeAction(openingAction);
    const matching = beamResult.candidates.find((candidate) => candidate.actionKey === openingKey);
    bestScore = matching?.searchScore ?? beamResult.bestScore;

    candidates = [
      matching ?? {
        action: openingAction,
        actionKey: openingKey,
        searchScore: bestScore,
        immediateChains: 0,
        immediateScore: 0,
        immediateTopout: false,
        immediateAllClear: false,
        immediateAttack: 0,
        immediateOutgoing: 0,
      },
    ];
    const seen = new Set([openingKey]);
    for (const candidate of beamResult.candidates) {
      if (seen.has(candidate.actionKey)) {
        continue;
      }
      seen.add(candidate.actionKey);
      candidates.push(candidate);
    }
  } else {
    bestAction = beamResult.bestAction;
    bestScore = beamResult.bestScore;
    candidates = beamResult.candidates;
  }

  return {
    kind: "template",
    opening,
    patternKey: openingState.plan?.patternKey ?? null,
    bestAction,
    bestActionKey: bestAction ? encodeAction(bestAction) : null,
    bestScore,
    candidates,
    candidateCount: candidates.length,
    expandedNodeCount: beamResult.expandedNodeCount,
    opponent: opponentResult,
    elapsedMs: performance.now() - startedAt,
  };
}
