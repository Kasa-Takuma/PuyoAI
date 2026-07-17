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
import { extractBoardFeaturesFast } from "./features-fast.js";
import { scoreBoardFeatures } from "./features.js";
import { createRng, nextPair } from "../core/randomizer.js";

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

// 段階的重み調整 (phase-adaptive weights): in a SAFE position (no incoming
// pressure, opponent not threatening, our own stack still low) the search
// shifts to a chain-growth profile instead of the legacy battle-ready one.
// The phase is computed once per analyzeTemplateMove call from the ROOT
// board and used for the whole search; boards reached deeper in the line
// may in reality be taller by then — an accepted approximation.
// mainFireBase/maxHeightBattle/maxHeightSafe/safeFireBonus (see
// DEFAULT_TEMPLATE_WEIGHTS below) hold the actual battle/safe values; the
// maxHeight>=9/>=10 near-topout penalties in evaluateBoard stay unchanged as
// the real guardrails regardless of phase. Valuing the standing main chain
// slightly ABOVE its realized value (safeFireBonus) makes the search patient
// (it grows the chain instead of cashing it out); the height decay (bonus
// shrinks to 0 by rootMaxHeight 9) guarantees it still fires before the
// board gets dangerous.
const SAFE_MAIN_FIRE_HEIGHT_CAP = 9;
const SAFE_MAX_ROOT_HEIGHT = 8;
const SAFE_OPPONENT_THREAT_CEILING = 6;
// 改善4 (adaptive beam width): safe phases can afford a wider search since
// there's no incoming danger to react to quickly; battle phase keeps the
// user's (narrow, fast) beam width setting exactly as before this existed.
const SAFE_MIN_BEAM_WIDTH = 24;
// 改善2: blends the repo's evolution-tuned v13 board-feature evaluation into
// the leaf, on top of the template heuristics above, so growth mode can see
// multi-step chain structure (bestVirtualChain, topVirtualChainSum, etc.)
// that the simple template scoring can't represent. Battle-phase behavior is
// untouched: profile.featureBlend is always 0 there (see LEGACY_GROWTH_
// PROFILE and buildGrowthProfile). Measured (solo sweep 0/8/24/48 ->
// avgMaxChain 3.63/3.5/3.3/3.5; battle 14-12, noise) to give no benefit on
// its own with the 3-ply main search as the bottleneck - defaulted to 0
// (off) pending a retest once sampling (改善3, below) is in place. The
// setting/machinery stays so that retest is just a config flip.
const DEFAULT_FEATURE_BLEND = 0;
const MAX_FEATURE_BLEND = 64;
const FEATURE_BLEND_PROFILE_ID = "chain_builder_v13";
// 改善3 (sampled lookahead): safe-phase-only, deterministic sampled
// continuations past the leaf's 3-ply horizon, run for only the very top of
// the full-eval band (see scoreLeafFrontier / runLeafSamples below).
const DEFAULT_TEMPLATE_SAMPLE_COUNT = 2; // 0 = off
const MAX_TEMPLATE_SAMPLE_COUNT = 4;
const DEFAULT_TEMPLATE_SAMPLE_DEPTH = 2;
const MAX_TEMPLATE_SAMPLE_DEPTH = 3;
const DEFAULT_TEMPLATE_SAMPLE_BEAM = 4;
const MAX_TEMPLATE_SAMPLE_BEAM = 8;
const MIN_TEMPLATE_SAMPLE_BEAM = 2;
const DEFAULT_TEMPLATE_SAMPLE_TOPK = 2;
const MAX_TEMPLATE_SAMPLE_TOPK = 4;
// Fixed salt for the sampling rng seed, alongside fastBoardHash(board) and
// the sample index, so the same position always analyzes identically.
const SAMPLE_RNG_SALT = "template-leaf-sample-v1";
// 改善5 (mid-search refine): the cheap per-level sortValue (shaped +
// evalValue + chainOutcomeValue*0.01) is chain-potential-blind past the
// current node's own immediate fire, so a board 2 plies from a big chain
// ranks no better than a tidy chain-dead one - skeletons get cut from the
// beam before a leaf ever sees them. Safe-phase-only (always 0 in battle -
// see LEGACY_GROWTH_PROFILE); widening the cheap-sorted slice by this many
// extra candidates before re-ranking it by virtual-fire potential lets a
// skeleton survive the cut. 0 = off.
const DEFAULT_TEMPLATE_MID_REFINE = 12;
const MAX_TEMPLATE_MID_REFINE = 24;

// 改善6 (tunable evaluation weights): every scalar below was previously a
// hardcoded multiplier/bonus inlined at its usage site (evaluateBoard,
// colorBalance, leafValue, scoreLeafFrontier, buildGrowthProfile's
// mainFireWeight formula). Collecting them here lets an outside tuner
// (tools/evolve-template-weights.js) search this space via
// settings.evalWeights without touching search/battle-only constants
// (OJAMA_SENT_BONUS, kill/threat penalties, offense multipliers, etc. stay
// fixed - those model fixed game rules, not tunable heuristics). Unlike
// featureBlend/sampleCount/midRefine, these apply in BOTH phases:
// evaluateBoard/leafValue run unconditionally regardless of safe/battle, so a
// weight override affects both (maxHeightBattle/maxHeightSafe/safeFireBonus
// are the only phase-specific entries here, exactly mirroring the old
// LEGACY_*/SAFE_* split).
export const DEFAULT_TEMPLATE_WEIGHTS = Object.freeze({
  templateScore: 18,
  seedScore: 10,
  holePenalty: -38,
  bumpiness: -10,
  maxHeightBattle: -30,
  maxHeightSafe: -14,
  topPressure1: -120,
  topPressure2: -260,
  colorTop: 0.6,
  colorBottom: -0.8,
  mainFireBase: 0.9,
  safeFireBonus: 0.2,
  subFire: 0.35,
  sampleGain: 0.5,
});

// Merges settings.evalWeights over the defaults, ignoring non-numeric/non-
// finite entries so malformed overrides degrade to "no override" rather than
// producing NaN scores. Returns the DEFAULT_TEMPLATE_WEIGHTS singleton itself
// (same reference) when there's nothing valid to override, so
// buildGrowthProfile's battle-phase fast path (see below) can stay bit-
// identical to the pre-改善6 frozen LEGACY_GROWTH_PROFILE singleton.
function mergeEvalWeights(overrides) {
  if (!overrides || typeof overrides !== "object") {
    return DEFAULT_TEMPLATE_WEIGHTS;
  }
  const merged = { ...DEFAULT_TEMPLATE_WEIGHTS };
  let changed = false;
  for (const key of Object.keys(DEFAULT_TEMPLATE_WEIGHTS)) {
    const value = overrides[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      merged[key] = value;
      changed = true;
    }
  }
  return changed ? Object.freeze(merged) : DEFAULT_TEMPLATE_WEIGHTS;
}

const LEGACY_GROWTH_PROFILE = Object.freeze({
  mainFireWeight: DEFAULT_TEMPLATE_WEIGHTS.mainFireBase,
  maxHeightPenalty: DEFAULT_TEMPLATE_WEIGHTS.maxHeightBattle,
  featureBlend: 0,
  sampleCount: 0,
  sampleDepth: DEFAULT_TEMPLATE_SAMPLE_DEPTH,
  sampleBeam: DEFAULT_TEMPLATE_SAMPLE_BEAM,
  sampleTopK: DEFAULT_TEMPLATE_SAMPLE_TOPK,
  midRefine: 0,
  weights: DEFAULT_TEMPLATE_WEIGHTS,
});

function clampFeatureBlend(value) {
  const parsed = Number.parseFloat(value);
  const resolved = Number.isFinite(parsed) ? parsed : DEFAULT_FEATURE_BLEND;
  return Math.max(0, Math.min(MAX_FEATURE_BLEND, resolved));
}

function clampIntSetting(value, min, max, defaultValue) {
  const parsed = Number.parseInt(value, 10);
  const resolved = Number.isFinite(parsed) ? parsed : defaultValue;
  return Math.max(min, Math.min(max, resolved));
}

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

function colorBalance(fastBoard, weights) {
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
  return weights.colorTop * (sorted[0] + sorted[1]) + weights.colorBottom * (sorted[2] + sorted[3]);
}

function evaluateBoard(fastBoard, profile = LEGACY_GROWTH_PROFILE) {
  const weights = profile.weights;
  const heights = computeHeights(fastBoard);
  const holes = countHoles(fastBoard, heights);
  const maxHeight = Math.max(...heights);
  let bumpiness = 0;
  for (let x = 1; x < BOARD_WIDTH; x += 1) {
    bumpiness += Math.abs(heights[x] - heights[x - 1]);
  }

  let s = 0;
  s += templateScore(heights) * weights.templateScore;
  s += seedScore(fastBoard) * weights.seedScore;
  s += groupBonuses(fastBoard);
  s += holes * weights.holePenalty;
  s += bumpiness * weights.bumpiness;
  s += maxHeight * profile.maxHeightPenalty;
  s -= dangerPenalty(fastBoard, heights);

  if (maxHeight >= 9) {
    s += weights.topPressure1;
  }
  if (maxHeight >= 10) {
    s += weights.topPressure2;
  }

  s += colorBalance(fastBoard, weights);

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
// stands, distinct from (and lighter than) the main chain's own mainFireBase
// weight. See DEFAULT_TEMPLATE_WEIGHTS.subFire.

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

// 段階的重み調整: builds the active growth profile for this analyzeTemplateMove
// call. Outside a safe position, this is exactly the legacy profile (bit-
// identical behavior, featureBlend/sample settings included - sampleCount 0
// there disables 改善3 entirely, and the weights fast path below keeps
// referential identity with LEGACY_GROWTH_PROFILE when evalWeights has no
// valid overrides). In a safe position, mainFireWeight ramps from
// mainFireBase+safeFireBonus on an empty board down to mainFireBase (legacy)
// as rootMaxHeight approaches SAFE_MAIN_FIRE_HEIGHT_CAP, maxHeightPenalty is
// relaxed to maxHeightSafe, and featureBlend/sampleSettings carry the
// (already-clamped) settings.
function buildGrowthProfile(safe, rootMaxHeight, featureBlend, sampleSettings, midRefine, weights) {
  if (!safe) {
    if (weights === DEFAULT_TEMPLATE_WEIGHTS) {
      return LEGACY_GROWTH_PROFILE;
    }
    return {
      mainFireWeight: weights.mainFireBase,
      maxHeightPenalty: weights.maxHeightBattle,
      featureBlend: 0,
      sampleCount: 0,
      sampleDepth: DEFAULT_TEMPLATE_SAMPLE_DEPTH,
      sampleBeam: DEFAULT_TEMPLATE_SAMPLE_BEAM,
      sampleTopK: DEFAULT_TEMPLATE_SAMPLE_TOPK,
      midRefine: 0,
      weights,
    };
  }
  const mainFireWeight =
    weights.mainFireBase +
    (weights.safeFireBonus * Math.max(0, SAFE_MAIN_FIRE_HEIGHT_CAP - rootMaxHeight)) / SAFE_MAIN_FIRE_HEIGHT_CAP;
  return { mainFireWeight, maxHeightPenalty: weights.maxHeightSafe, featureBlend, ...sampleSettings, midRefine, weights };
}

function leafValue(fastBoard, profile = LEGACY_GROWTH_PROFILE) {
  const base = evaluateBoard(fastBoard, profile);
  const { mainProbe, subProbe } = virtualFireProbes(fastBoard);
  return base + profile.mainFireWeight * mainProbe.value + profile.weights.subFire * (subProbe?.value ?? 0);
}

// 改善3: a small level-synchronized beam over `pairs` (drawn deterministically
// - see runLeafSamples), starting from `startBoard`, using the SAME node
// semantics as the main search: fastResolveTurn per action; topout prunes
// the branch; a fire banks chainOutcomeValue into the line's `shaped` and
// continues (non-terminal, mirroring the main search's 攻撃タイミング判断-era
// continuation). No simulateOjamaSettle call is needed here: sampling only
// ever runs in the safe phase, where pendingOjama is 0 at the root and (per
// the main search's own invariant - no *new* incoming attack is modeled
// during lookahead) stays exactly 0 through the whole line, so there is
// never anything to offset or drop. Returns null if every branch topped out
// before the sample completed (a genuinely dead line, not a 0-value one).
function runSampleContinuation(startBoard, pairs, profile) {
  let frontier = [{ board: startBoard, shaped: 0 }];

  for (let depth = 0; depth < pairs.length && frontier.length > 0; depth += 1) {
    const { axis, child } = pairToCodes(pairs[depth]);
    const expanded = [];

    for (const parent of frontier) {
      const actions = fastEnumerateLegalActions(parent.board, axis, child);
      for (const action of actions) {
        const result = fastResolveTurn(parent.board, axis, child, action);
        if (result.topout) {
          continue;
        }

        const firing = result.totalChains > 0;
        const shaped = firing ? parent.shaped + chainOutcomeValue(result) : parent.shaped;
        const evalValue = evaluateBoard(result.board, profile);
        const quickValue = evalValue + chainOutcomeValue(result) * 0.01;
        expanded.push({
          board: result.board,
          shaped,
          sortValue: shaped + quickValue,
          hash: fastBoardHash(result.board),
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
    frontier = deduped.slice(0, profile.sampleBeam);
  }

  if (frontier.length === 0) {
    return null;
  }

  let best = -Infinity;
  for (const entry of frontier) {
    const value = entry.shaped + leafValue(entry.board, profile);
    if (value > best) {
      best = value;
    }
  }
  return best;
}

// 改善3: averages runSampleContinuation over profile.sampleCount independent
// samples (each its own deterministic rng, so re-analyzing the same position
// always gives the same result), drawing profile.sampleDepth random pairs
// per sample. Samples that topped out entirely are excluded from the
// average rather than counted as 0 (a dead line isn't "worth nothing", it's
// just not informative); returns null only if every sample died.
function runLeafSamples(entryBoard, profile) {
  const boardHash = fastBoardHash(entryBoard);
  const outcomes = [];

  for (let sampleIndex = 0; sampleIndex < profile.sampleCount; sampleIndex += 1) {
    const rng = createRng(`${SAMPLE_RNG_SALT}:${boardHash}:${sampleIndex}`);
    const pairs = Array.from({ length: profile.sampleDepth }, () => nextPair(rng));
    const outcome = runSampleContinuation(entryBoard, pairs, profile);
    if (outcome !== null) {
      outcomes.push(outcome);
    }
  }

  if (outcomes.length === 0) {
    return null;
  }
  return outcomes.reduce((sum, outcome) => sum + outcome, 0) / outcomes.length;
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

function scoreLeafFrontier(candidates, leafFrontier, beamWidth, profile) {
  leafFrontier.sort((a, b) => (b.shaped + b.evalValue) - (a.shaped + a.evalValue));
  const fullEvalCount = Math.min(LEAF_FULL_EVAL_LIMIT, beamWidth);

  leafFrontier.forEach((entry, index) => {
    const inFullEvalBand = index < fullEvalCount;

    // leafValue() runs its own fresh evaluateBoard() internally, so the
    // danger and opponent-threat penalties (already folded into the cheap
    // entry.evalValue) have to be re-added alongside it for the full-eval
    // branch. `leafPart` is kept around (rather than recomputed) since 改善3
    // below needs the exact same value as the "static estimate" it replaces
    // a weighted fraction of.
    const leafPart = inFullEvalBand ? leafValue(entry.board, profile) : null;
    let value = inFullEvalBand
      ? entry.shaped + leafPart + entry.pendingAfter * OJAMA_DANGER_PENALTY + entry.threatPenalty
      : entry.shaped + entry.evalValue;

    // 改善2: v13 feature blend, only for the same top-8 band that already
    // pays for the full leafValue() probe, and only when profile.
    // featureBlend > 0 (always 0 outside the safe/growth phase), so battle
    // phase and featureBlend: 0 never pay this extra cost.
    if (inFullEvalBand && profile.featureBlend > 0) {
      const features = extractBoardFeaturesFast(entry.board, { includeVirtualChains: true });
      value += profile.featureBlend * scoreBoardFeatures(features, FEATURE_BLEND_PROFILE_ID);
    }

    // 改善3: sampled lookahead, only for the very top of the full-eval band
    // (index < profile.sampleTopK) and only when profile.sampleCount > 0
    // (always 0 outside the safe phase). The sampled continuation REPLACES a
    // weights.sampleGain fraction of the static leafValue estimate
    // (`leafPart`) with the deeper, sampled one - not a bonus stacked on top.
    if (inFullEvalBand && index < profile.sampleTopK && profile.sampleCount > 0) {
      const sampleOutcome = runLeafSamples(entry.board, profile);
      if (sampleOutcome !== null) {
        value += profile.weights.sampleGain * (sampleOutcome - leafPart);
      }
    }

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
//
// 段階的重み調整: `settings.phaseAdaptive` (default true) gates a chain-growth
// profile used only in a SAFE position (see buildGrowthProfile); when false,
// or whenever the position isn't safe, `profile` is exactly LEGACY_GROWTH_
// PROFILE and every score in this function is bit-identical to before phase
// adaptivity existed.
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
  const fastRoot = fromLegacyBoard(board);
  const { axis, child } = pairToCodes(currentPair);
  const rootActions = fastEnumerateLegalActions(fastRoot, axis, child);
  const stats = { expandedNodeCount: 0 };
  const candidates = [];
  const isRootFinal = pieces.length === 1;
  const rootPending = Math.max(0, pendingOjama | 0);

  // 段階的重み調整: computed once from the root board, used for the whole
  // search (see the comment above the phase-adaptive constants for the
  // "deeper boards may be taller by then" approximation this accepts).
  const phaseAdaptive = settings.phaseAdaptive !== false;
  const rootMaxHeight = Math.max(...fastColumnHeights(fastRoot));
  const safe =
    phaseAdaptive &&
    rootPending === 0 &&
    opponentThreat < SAFE_OPPONENT_THREAT_CEILING &&
    rootMaxHeight <= SAFE_MAX_ROOT_HEIGHT;
  const featureBlend = clampFeatureBlend(settings.featureBlend);
  const sampleSettings = {
    sampleCount: clampIntSetting(settings.templateSampleCount, 0, MAX_TEMPLATE_SAMPLE_COUNT, DEFAULT_TEMPLATE_SAMPLE_COUNT),
    sampleDepth: clampIntSetting(settings.templateSampleDepth, 1, MAX_TEMPLATE_SAMPLE_DEPTH, DEFAULT_TEMPLATE_SAMPLE_DEPTH),
    sampleBeam: clampIntSetting(settings.templateSampleBeam, MIN_TEMPLATE_SAMPLE_BEAM, MAX_TEMPLATE_SAMPLE_BEAM, DEFAULT_TEMPLATE_SAMPLE_BEAM),
    sampleTopK: clampIntSetting(settings.templateSampleTopK, 1, MAX_TEMPLATE_SAMPLE_TOPK, DEFAULT_TEMPLATE_SAMPLE_TOPK),
  };
  const midRefine = clampIntSetting(settings.templateMidRefine, 0, MAX_TEMPLATE_MID_REFINE, DEFAULT_TEMPLATE_MID_REFINE);
  const evalWeights = mergeEvalWeights(settings.evalWeights);
  const profile = buildGrowthProfile(safe, rootMaxHeight, featureBlend, sampleSettings, midRefine, evalWeights);
  const phase = safe ? "safe" : "battle";

  // 改善4 (adaptive beam width): computed after `safe` is known, so it must
  // stay below the phase-adaptive block above.
  const requestedBeamWidth = clampBeamWidth(settings.templateBeamWidth);
  const beamWidth = safe ? Math.max(requestedBeamWidth, SAFE_MIN_BEAM_WIDTH) : requestedBeamWidth;

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
    const settledEval = evaluateBoard(settle.board, profile);
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
        const settledEval = evaluateBoard(settle.board, profile);
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

    // 改善5 (mid-search refine): re-rank the cheap-sorted slice by adding
    // virtual-fire chain potential before cutting to beamWidth - makes "2
    // plies from a big chain" outrank "tidy but chain-dead" during the
    // search; leaf-only evaluation could not do this (measured). Ranking key
    // only: refinedScore never gets written back onto the entry, so it can't
    // leak into shaped/evalValue/sortValue or double-count against
    // leafValue's own (freshly-computed) virtual fire probe later.
    let survivors;
    if (profile.midRefine > 0) {
      const refineSlice = deduped.slice(0, beamWidth + profile.midRefine);
      const refined = refineSlice.map((entry) => ({
        entry,
        refinedScore: entry.sortValue + profile.mainFireWeight * virtualFireProbes(entry.board).mainProbe.value,
      }));
      refined.sort((a, b) => b.refinedScore - a.refinedScore);
      survivors = refined.slice(0, beamWidth).map((r) => r.entry);
    } else {
      survivors = deduped.slice(0, beamWidth);
    }

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

  scoreLeafFrontier(candidates, leafFrontier, beamWidth, profile);

  candidates.sort((a, b) => b.searchScore - a.searchScore);

  return {
    bestAction: candidates[0]?.action ?? null,
    bestScore: candidates[0]?.searchScore ?? -Infinity,
    candidates,
    expandedNodeCount: stats.expandedNodeCount,
    phase,
  };
}

// Opening-book state is per battle-harness instance (settings.instanceId),
// not a single module-level singleton, so two players sharing this module
// (e.g. a self-play harness) don't corrupt each other's opening sequence.
const openingStates = new Map();

function getOpeningState(instanceId) {
  let state = openingStates.get(instanceId);
  if (!state) {
    state = { plan: null, movesUsed: 0, active: false };
    openingStates.set(instanceId, state);
  }
  return state;
}

export function resetTemplateOpeningState(instanceId) {
  if (instanceId === undefined) {
    openingStates.clear();
  } else {
    openingStates.delete(instanceId);
  }
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
      phase: null,
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
  const instanceId = settings.instanceId ?? "default";
  let openingState = getOpeningState(instanceId);

  if (openingBookEligible && isBoardEmpty(board) && nextQueue.length >= 2) {
    const plan = buildOpeningPlan([currentPair, nextQueue[0], nextQueue[1]]);
    openingState = { plan, movesUsed: 0, active: plan !== null };
    openingStates.set(instanceId, openingState);
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
    phase: beamResult.phase,
    elapsedMs: performance.now() - startedAt,
  };
}
