import test, { beforeEach } from "node:test";
import assert from "node:assert/strict";

import {
  analyzeTemplateMove,
  DEFAULT_TEMPLATE_WEIGHTS,
  resetTemplateOpeningState,
  simulateOjamaSettle,
} from "../src/ai/template-ai.js";
import { buildOpeningPlan } from "../src/ai/template-opening-book.js";
import { applyPlacement, boardFromRows, createEmptyBoard, enumerateLegalActions, encodeAction } from "../src/core/board.js";
import { COLORS } from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";
import { fastColumnHeights, fromLegacyBoard } from "../src/core/fast-board.js";

beforeEach(() => {
  resetTemplateOpeningState();
});

test("buildOpeningPlan classifies a non-monochrome AA-AB-AC pattern with legal actions", () => {
  const p1 = { axis: COLORS.RED, child: COLORS.RED };
  const p2 = { axis: COLORS.RED, child: COLORS.GREEN };
  const p3 = { axis: COLORS.RED, child: COLORS.BLUE };

  const plan = buildOpeningPlan([p1, p2, p3]);

  assert.ok(plan);
  assert.equal(plan.patternKey, "AA-AB-AC");
  assert.equal(plan.actions.length, 3);

  const board = createEmptyBoard();
  const legal1 = enumerateLegalActions(board, p1);
  assert.ok(legal1.some((a) => a.column === plan.actions[0].column && a.orientation === plan.actions[0].orientation));
});

test("buildOpeningPlan classifies a monochrome AA-AA-AA pattern with legal actions", () => {
  const p1 = { axis: COLORS.YELLOW, child: COLORS.YELLOW };
  const p2 = { axis: COLORS.YELLOW, child: COLORS.YELLOW };
  const p3 = { axis: COLORS.YELLOW, child: COLORS.YELLOW };

  const plan = buildOpeningPlan([p1, p2, p3]);

  assert.ok(plan);
  assert.equal(plan.patternKey, "AA-AA-AA");

  const board = createEmptyBoard();
  const legal1 = enumerateLegalActions(board, p1);
  assert.ok(legal1.some((a) => a.column === plan.actions[0].column && a.orientation === plan.actions[0].orientation));
});

test("buildOpeningPlan returns null for a pattern absent from the table", () => {
  const p1 = { axis: COLORS.RED, child: COLORS.GREEN };
  const p2 = { axis: COLORS.BLUE, child: COLORS.YELLOW };
  const p3 = { axis: COLORS.GREEN, child: COLORS.RED };

  const plan = buildOpeningPlan([p1, p2, p3]);

  assert.equal(plan, null);
});

test("analyzeTemplateMove uses the opening book for the first three moves in order", () => {
  const p1 = { axis: COLORS.RED, child: COLORS.RED };
  const p2 = { axis: COLORS.RED, child: COLORS.GREEN };
  const p3 = { axis: COLORS.RED, child: COLORS.BLUE };
  const board0 = createEmptyBoard();

  const analysis1 = analyzeTemplateMove({ board: board0, currentPair: p1, nextQueue: [p2, p3] });
  assert.equal(analysis1.opening, true);
  assert.equal(analysis1.patternKey, "AA-AB-AC");
  assert.deepEqual(analysis1.bestAction, { column: 0, orientation: "RIGHT" });
  assert.equal(analysis1.bestActionKey, encodeAction(analysis1.bestAction));

  const board1 = applyPlacement(board0, p1, analysis1.bestAction).board;
  const analysis2 = analyzeTemplateMove({ board: board1, currentPair: p2, nextQueue: [p3] });
  assert.equal(analysis2.opening, true);
  assert.deepEqual(analysis2.bestAction, { column: 2, orientation: "DOWN" });

  const board2 = applyPlacement(board1, p2, analysis2.bestAction).board;
  const analysis3 = analyzeTemplateMove({ board: board2, currentPair: p3, nextQueue: [] });
  assert.equal(analysis3.opening, true);
  assert.deepEqual(analysis3.bestAction, { column: 1, orientation: "DOWN" });

  const board3 = applyPlacement(board2, p3, analysis3.bestAction).board;
  const analysis4 = analyzeTemplateMove({
    board: board3,
    currentPair: { axis: COLORS.GREEN, child: COLORS.YELLOW },
    nextQueue: [],
  });
  assert.equal(analysis4.opening, false);
});

test("analyzeTemplateMove falls back to beam search when the opening plan desyncs", () => {
  const p1 = { axis: COLORS.RED, child: COLORS.RED };
  const p2 = { axis: COLORS.RED, child: COLORS.GREEN };
  const p3 = { axis: COLORS.RED, child: COLORS.BLUE };
  const board0 = createEmptyBoard();

  const analysis1 = analyzeTemplateMove({ board: board0, currentPair: p1, nextQueue: [p2, p3] });
  assert.equal(analysis1.opening, true);

  const garbageBoard = boardFromRows([
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
    "O.....",
  ]);
  const analysis2 = analyzeTemplateMove({ board: garbageBoard, currentPair: p2, nextQueue: [] });
  assert.equal(analysis2.opening, false);
  assert.notEqual(analysis2.bestAction, null);
});

test("beam search finds an obvious two-chain action", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const analysis = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const result = resolveTurn(board, currentPair, analysis.bestAction);

  assert.ok(result.totalChains >= 2);
});

test("beam search avoids an immediate topout when a safe move exists", () => {
  const board = boardFromRows([
    "......",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
  ]);
  const currentPair = { axis: COLORS.BLUE, child: COLORS.YELLOW };

  const analysis = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const result = resolveTurn(board, currentPair, analysis.bestAction);

  assert.equal(result.topout, false);
});

test("analysis contract: keys, ordering, and determinism", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };
  const nextQueue = [{ axis: COLORS.BLUE, child: COLORS.YELLOW }];

  const first = analyzeTemplateMove({ board, currentPair, nextQueue });
  const second = analyzeTemplateMove({ board, currentPair, nextQueue });

  assert.equal(first.opening, false);
  assert.equal(first.bestActionKey, encodeAction(first.bestAction));
  assert.ok(first.candidates.length > 0);
  for (let i = 1; i < first.candidates.length; i += 1) {
    assert.ok(first.candidates[i - 1].searchScore >= first.candidates[i].searchScore);
  }
  assert.deepEqual(first.candidates[0].action, first.bestAction);
  assert.equal(first.bestActionKey, second.bestActionKey);
});

test("analyzeTemplateMove returns no candidates when currentPair is missing", () => {
  const board = createEmptyBoard();
  const analysis = analyzeTemplateMove({ board, currentPair: null, nextQueue: [] });

  assert.equal(analysis.bestAction, null);
  assert.equal(analysis.candidates.length, 0);
});

test("simulateOjamaSettle offsets pending against attack before dropping the remainder", () => {
  const empty = fromLegacyBoard(createEmptyBoard());

  const partiallyCanceled = simulateOjamaSettle(empty, 10, 4);
  assert.equal(partiallyCanceled.outgoing, 0);
  assert.equal(partiallyCanceled.pendingAfter, 0);
  assert.deepEqual([...fastColumnHeights(partiallyCanceled.board)], [1, 1, 1, 1, 1, 1]);

  const fullyCanceled = simulateOjamaSettle(empty, 3, 10);
  assert.equal(fullyCanceled.outgoing, 7);
  assert.equal(fullyCanceled.pendingAfter, 0);
  assert.equal(fullyCanceled.board, empty);
  assert.deepEqual([...fastColumnHeights(fullyCanceled.board)], [0, 0, 0, 0, 0, 0]);
});

test("simulateOjamaSettle spreads the drop deterministically (even share + lowest-column remainder)", () => {
  const empty = fromLegacyBoard(createEmptyBoard());

  const first = simulateOjamaSettle(empty, 8, 0);
  const second = simulateOjamaSettle(empty, 8, 0);

  assert.equal(first.pendingAfter, 0);
  assert.equal(first.outgoing, 0);
  assert.equal(first.lethal, false);
  // floor(8/6) = 1 to every column, remainder 2 to the two lowest (tied at 0,
  // leftmost wins): columns 0 and 1 end up at height 2, the rest at height 1.
  assert.deepEqual([...fastColumnHeights(first.board)], [2, 2, 1, 1, 1, 1]);
  assert.deepEqual([...fastColumnHeights(second.board)], [2, 2, 1, 1, 1, 1]);
});

test("simulateOjamaSettle reports lethal when the drop reaches the topout cell", () => {
  const rows = ["......", ...Array(11).fill("..R...")];
  const board = fromLegacyBoard(boardFromRows(rows));

  const settle = simulateOjamaSettle(board, 6, 0);

  assert.equal(settle.lethal, true);
});

test("battle-aware search fires an available chain under heavy incoming pressure", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const underPressure = analyzeTemplateMove({ board, currentPair, nextQueue: [], pendingOjama: 24 });
  const pressureResult = resolveTurn(board, currentPair, underPressure.bestAction);
  assert.ok(pressureResult.totalChains > 0);

  const relaxed = analyzeTemplateMove({ board, currentPair, nextQueue: [], pendingOjama: 0 });
  assert.notEqual(relaxed.bestAction, null);
});

test("opening book is skipped entirely when pendingOjama is positive", () => {
  const p1 = { axis: COLORS.RED, child: COLORS.RED };
  const p2 = { axis: COLORS.RED, child: COLORS.GREEN };
  const p3 = { axis: COLORS.RED, child: COLORS.BLUE };
  const board0 = createEmptyBoard();

  const analysis = analyzeTemplateMove({ board: board0, currentPair: p1, nextQueue: [p2, p3], pendingOjama: 5 });

  assert.equal(analysis.opening, false);
  assert.notEqual(analysis.bestAction, null);
});

test("pendingOjama: 0 is behaviorally identical to omitting it", () => {
  const board = boardFromRows([
    "......",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
  ]);
  const currentPair = { axis: COLORS.BLUE, child: COLORS.YELLOW };

  const withoutField = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const withZero = analyzeTemplateMove({ board, currentPair, nextQueue: [], pendingOjama: 0 });

  assert.equal(withoutField.bestActionKey, withZero.bestActionKey);
  assert.equal(withoutField.bestScore, withZero.bestScore);
});

test("leafValue rewards an independent sub-chain (副砲) alongside the main chain", () => {
  // Both boards share an identical main group (R, columns 0-1) and an
  // identical-shaped secondary group (columns 4-5) two columns away, so
  // evaluateBoard's shape-based heuristics (holes/bumpiness/templateScore/
  // seedScore/groupBonuses) score them almost the same. The only meaningful
  // difference is the secondary group's color: same as main (R) in
  // `sameColor`, so it doesn't count as an independent second fire; a
  // different, far-away color (G) in `differentColor`, so it does.
  const sameColor = boardFromRows(["RR..RR"]);
  const differentColor = boardFromRows(["RR..GG"]);
  const currentPair = { axis: COLORS.BLUE, child: COLORS.BLUE };
  // This action stacks the current (unrelated) piece directly on top of the
  // main group at column 0, so leafValue's probe of the resulting board is
  // the same shape in both fixtures - isolating the secondary group's color
  // as the only real difference between the two candidate scores below.
  const probeKey = encodeAction({ column: 0, orientation: "UP" });

  const withoutSub = analyzeTemplateMove({ board: sameColor, currentPair, nextQueue: [] });
  const withSub = analyzeTemplateMove({ board: differentColor, currentPair, nextQueue: [] });

  const scoreWithoutSub = withoutSub.candidates.find((c) => c.actionKey === probeKey).searchScore;
  const scoreWithSub = withSub.candidates.find((c) => c.actionKey === probeKey).searchScore;

  // SUB_FIRE_WEIGHT (0.35) applied to a ~32000-point single-chain probe is a
  // multi-thousand-point swing, dwarfing any incidental base-heuristic noise
  // (a few points) between the two fixtures - a robust signal the sub-chain
  // term actually fired.
  assert.ok(
    scoreWithSub - scoreWithoutSub > 5000,
    `expected a large sub-chain bonus, got ${scoreWithSub - scoreWithoutSub}`,
  );
});

test("counter-fire preservation: fires a small independent chain under pressure without disturbing the main line", () => {
  // MAIN (columns 0-2, color B): a mature loose group. The extra Y on top of
  // column 0 keeps a lone puyo on the board after MAIN's own virtual fire, so
  // that probe doesn't accidentally trigger an all-clear bonus that would
  // swamp the comparison. currentPair is G, so MAIN (color B) cannot be
  // fired this turn - only SMALL can.
  // SMALL (columns 3-5, color G): ready to complete and fire this turn.
  const board = boardFromRows(["Y.....", "BBBGGG"]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };

  // pendingOjama chosen so the small fire's own attack (score 100 -> floor(100/70) = 1)
  // fully offsets it.
  const analysis = analyzeTemplateMove({ board, currentPair, nextQueue: [], pendingOjama: 1 });

  const result = resolveTurn(board, currentPair, analysis.bestAction);
  assert.ok(result.totalChains >= 1 && result.totalChains <= 2);

  // The main B group (columns 0-2, bottom row) must still be standing.
  assert.deepEqual(result.finalBoard[0].slice(0, 3), [COLORS.BLUE, COLORS.BLUE, COLORS.BLUE]);
});

test("凝視: opponent.threat reflects their best immediate virtual-fire attack", () => {
  // GGGRRR: a monochrome R,R (or G,G) drop at the seam completes a 5-cell
  // group (score 100 -> attack floor(100/70) = 1). This is the same fixture
  // used elsewhere in this file for its known, validated chain shape.
  const opponentBoard = boardFromRows([
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
  ]);
  const ownBoard = createEmptyBoard();
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const analysis = analyzeTemplateMove({
    board: ownBoard,
    currentPair,
    nextQueue: [],
    opponent: { board: opponentBoard },
  });

  // GGGRRR is a low, empty-ish board: no vulnerability flags, so the offense
  // multiplier ("攻撃タイミング判断") is exactly 1.
  assert.deepEqual(analysis.opponent, { threat: 1, offenseMultiplier: 1 });
});

test("opponent: null is behaviorally identical to omitting the field", () => {
  const board = boardFromRows([
    "......",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
    "..R...",
    "..G...",
    "..B...",
    "..Y...",
  ]);
  const currentPair = { axis: COLORS.BLUE, child: COLORS.YELLOW };

  const withoutField = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const withNull = analyzeTemplateMove({ board, currentPair, nextQueue: [], opponent: null });

  assert.equal(withoutField.bestActionKey, withNull.bestActionKey);
  assert.equal(withoutField.bestScore, withNull.bestScore);
  assert.equal(withoutField.opponent, null);
  assert.equal(withNull.opponent, null);
});

test("凝視: avoids a placement that the opponent's ready fire would top us out of", () => {
  // Column 2 (the topout column) sits 2 rows below the topout row; every
  // other column is packed high with inert garbage filler. Stacking our own
  // piece on column 2 leaves it exactly one garbage cell away from the
  // topout cell, and (being the lowest column) it is exactly where a small
  // opponent-threat drop lands.
  const board = boardFromRows([
    "OO.OOO",
    "OO.OOO",
    "OO.OOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
  ]);
  const currentPair = { axis: COLORS.YELLOW, child: COLORS.YELLOW };
  const bigThreatOpponent = boardFromRows([
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
  ]);

  const underThreat = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    opponent: { board: bigThreatOpponent },
  });
  assert.notEqual(underThreat.bestAction.column, 2);

  const noOpponent = analyzeTemplateMove({ board, currentPair, nextQueue: [], opponent: null });
  assert.notEqual(noOpponent.bestAction, null);
});

test("malformed opponent data is treated as null instead of throwing", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const analysis = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    opponent: { board: "garbage" },
  });

  assert.equal(analysis.opponent, null);
  assert.notEqual(analysis.bestAction, null);
});

test("攻撃タイミング判断: a vulnerable opponent scores the same fire higher than a healthy one", () => {
  // Our board: the classic GGGRRR two-chain, ready to fire this turn.
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };
  const fireKey = encodeAction({ column: 3, orientation: "UP" });

  const healthyOpponent = createEmptyBoard();
  // Vulnerable: their topout column (2) alone is packed to height 9, meeting
  // the OFFENSE_TOPOUT_HEIGHT_THRESHOLD flag (maxHeight stays at 9, below the
  // separate max-height flag, and pendingOjama is 0 - only one flag fires).
  const vulnerableOpponent = boardFromRows([
    "......",
    "......",
    "......",
    "..O...",
    "..O...",
    "..O...",
    "..O...",
    "..O...",
    "..O...",
    "..O...",
    "..O...",
    "..O...",
  ]);

  const withHealthy = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    opponent: { board: healthyOpponent, pendingOjama: 0 },
  });
  const withVulnerable = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    opponent: { board: vulnerableOpponent, pendingOjama: 0 },
  });

  assert.equal(withHealthy.opponent.offenseMultiplier, 1);
  assert.ok(withVulnerable.opponent.offenseMultiplier > 1);

  const healthyFireScore = withHealthy.candidates.find((c) => c.actionKey === fireKey).searchScore;
  const vulnerableFireScore = withVulnerable.candidates.find((c) => c.actionKey === fireKey).searchScore;
  assert.ok(vulnerableFireScore > healthyFireScore);
});

test("とどめ: a lethal-to-the-opponent fire scores well above the same fire against a roomy opponent", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  // Fragile opponent: their topout column (2) is one garbage cell away from
  // filling the topout row, and it is (tied-)lowest across the whole board,
  // so any nonzero outgoing garbage from our fire finishes them off.
  const fragileOpponent = boardFromRows([
    "......",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
    "OOOOOO",
  ]);
  const roomyOpponent = createEmptyBoard();

  const withFragile = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    opponent: { board: fragileOpponent, pendingOjama: 0 },
  });
  const withRoomy = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    opponent: { board: roomyOpponent, pendingOjama: 0 },
  });

  const result = resolveTurn(board, currentPair, withFragile.bestAction);
  assert.ok(result.totalChains >= 1);

  // OPPONENT_KILL_BONUS is 800000; both opponents also get an offense
  // multiplier from being probed, so require at least half the raw bonus as
  // a robust, non-overfit margin.
  assert.ok(withFragile.bestScore - withRoomy.bestScore > 400000);
});

test("攻撃タイミング判断 invariance: opponent null still matches omitting the field on a firing fixture", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const withoutField = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const withNull = analyzeTemplateMove({ board, currentPair, nextQueue: [], opponent: null });

  assert.equal(withoutField.bestActionKey, withNull.bestActionKey);
  assert.equal(withoutField.bestScore, withNull.bestScore);
});

test("段階的重み調整: battle-mode invariance — pendingOjama keeps the search battle-ready regardless of phaseAdaptive", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const adaptive = analyzeTemplateMove({ board, currentPair, nextQueue: [], pendingOjama: 24 });
  const legacy = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    pendingOjama: 24,
    settings: { phaseAdaptive: false },
  });

  // pendingOjama > 0 makes the position unsafe regardless of phaseAdaptive,
  // so both runs must land on the exact same battle-mode line.
  assert.equal(adaptive.phase, "battle");
  assert.equal(legacy.phase, "battle");
  assert.equal(adaptive.bestActionKey, legacy.bestActionKey);
  assert.equal(adaptive.bestScore, legacy.bestScore);
});

test("段階的重み調整: adaptive activity — a low, safe solo board scores differently with the growth profile", () => {
  // Small ready fire (G, columns 0-2) plus an independent 2-cell G group
  // (columns 4-5) that survives the fire; rootMaxHeight is 1, well within
  // the safe threshold.
  const board = boardFromRows(["GGG.RR"]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };

  const adaptive = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const legacy = analyzeTemplateMove({ board, currentPair, nextQueue: [], settings: { phaseAdaptive: false } });

  assert.equal(adaptive.phase, "safe");
  assert.equal(legacy.phase, "battle");
  assert.notEqual(adaptive.bestScore, legacy.bestScore);
});

test("段階的重み調整: growth behavior — patient in growth mode, cashes out in legacy mode", () => {
  // A small ready fire (2 G's stacked at column 0) sits far from an isolated
  // garbage cell (column 3) that keeps the board from fully clearing when
  // fired. The current G,G piece can either complete that fire now, or be
  // placed at column 2 - too far to touch the existing G's directly, but
  // close enough that a *virtual* probe bridging columns 0-2 (plus the
  // garbage-adjacency clear) reveals a big unrealized potential. Growth
  // mode's slight overvaluation of standing potential tips the balance
  // toward NOT firing; legacy mode cashes out immediately.
  const board = boardFromRows(["G.....", "G..O.."]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };

  // 改善3 (sampled lookahead) evolution: with sampling on (the new default),
  // the search now samples PAST firing this line too and discovers enough
  // downstream value there to prefer firing after all - a legitimately
  // better-informed decision, not a bug. templateSampleCount: 0 isolates the
  // original 段階的重み調整 behavior this test is about (mainFireWeight/
  // maxHeightPenalty patience) from that newer mechanism.
  const adaptive = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { templateSampleCount: 0 },
  });
  assert.equal(adaptive.phase, "safe");
  const adaptiveResult = resolveTurn(board, currentPair, adaptive.bestAction);
  assert.equal(adaptiveResult.totalChains, 0);

  // Legacy mode's choice is reported for documentation only - no assertion
  // on which action it picks, per the task's own guidance (it fires here,
  // cashing out the same structure immediately).
  resetTemplateOpeningState();
  const legacy = analyzeTemplateMove({ board, currentPair, nextQueue: [], settings: { phaseAdaptive: false } });
  assert.notEqual(legacy.bestAction, null);
});

test("段階的重み調整: opening state is isolated per settings.instanceId", () => {
  const p1 = { axis: COLORS.RED, child: COLORS.RED };
  const p2 = { axis: COLORS.RED, child: COLORS.GREEN };
  const p3 = { axis: COLORS.RED, child: COLORS.BLUE };

  let boardA = createEmptyBoard();
  let boardB = createEmptyBoard();

  // Interleave two "players" sharing this module, alternating instanceId,
  // each running the same 3-pair opening from an empty board.
  const a1 = analyzeTemplateMove({ board: boardA, currentPair: p1, nextQueue: [p2, p3], settings: { instanceId: "A" } });
  const b1 = analyzeTemplateMove({ board: boardB, currentPair: p1, nextQueue: [p2, p3], settings: { instanceId: "B" } });
  assert.equal(a1.opening, true);
  assert.equal(b1.opening, true);
  boardA = applyPlacement(boardA, p1, a1.bestAction).board;
  boardB = applyPlacement(boardB, p1, b1.bestAction).board;

  const a2 = analyzeTemplateMove({ board: boardA, currentPair: p2, nextQueue: [p3], settings: { instanceId: "A" } });
  const b2 = analyzeTemplateMove({ board: boardB, currentPair: p2, nextQueue: [p3], settings: { instanceId: "B" } });
  assert.equal(a2.opening, true);
  assert.equal(b2.opening, true);
  boardA = applyPlacement(boardA, p2, a2.bestAction).board;
  boardB = applyPlacement(boardB, p2, b2.bestAction).board;

  const a3 = analyzeTemplateMove({ board: boardA, currentPair: p3, nextQueue: [], settings: { instanceId: "A" } });
  const b3 = analyzeTemplateMove({ board: boardB, currentPair: p3, nextQueue: [], settings: { instanceId: "B" } });
  // Neither instance desynced despite the interleaving - both completed the
  // full 3-move plan, matching the single-instance opening-book fixture.
  assert.equal(a3.opening, true);
  assert.equal(b3.opening, true);
  assert.deepEqual(a3.bestAction, b3.bestAction);

  resetTemplateOpeningState("A");
  const afterResetA = analyzeTemplateMove({ board: createEmptyBoard(), currentPair: p1, nextQueue: [p2, p3], settings: { instanceId: "A" } });
  assert.equal(afterResetA.opening, true);
});

test("改善2 (v13 feature blend): featureBlend 0 reproduces the pre-blend/pre-sampling baseline exactly", () => {
  // Same fixture as the "growth behavior" test above. These exact numbers
  // (bestAction, bestScore) were recorded from this codebase before the v13
  // feature blend AND 改善3 (sampled lookahead) existed, so this is a direct
  // regression guard: featureBlend must be a strict no-op when 0.
  // templateSampleCount: 0 isolates that from 改善3's own (separately
  // tested) effect - since sampling now defaults on, leaving it enabled here
  // would also change this fixture's outcome for an unrelated reason. 改善4
  // (adaptive beam width) doesn't need neutralizing: this fixture has no
  // nextQueue, so the search never leaves the root level and beamWidth only
  // ever caps scoreLeafFrontier's already-8-capped full-eval band.
  const board = boardFromRows(["G.....", "G..O.."]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };

  const analysis = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { featureBlend: 0, templateSampleCount: 0 },
  });

  assert.equal(analysis.phase, "safe");
  assert.equal(analysis.bestActionKey, "UP:2");
  assert.equal(analysis.bestScore, 300799.76666666666);
});

test("改善2 (v13 feature blend): explicit blend (8) changes the safe-phase score, and the default now matches featureBlend: 0", () => {
  // DEFAULT_FEATURE_BLEND was flipped from 8 to 0 after measurement showed
  // no solo/battle benefit on its own (the 3-ply main-search horizon was the
  // real bottleneck, not the leaf's board-feature scoring) - the setting and
  // machinery stay for a future retest alongside sampling.
  const board = boardFromRows(["G.....", "G..O.."]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };
  // With 改善3 (sampled lookahead) on by default, the OVERALL bestAction/
  // bestScore on this fixture is now set by a firing candidate whose score
  // comes from the pruning floor (unaffected by featureBlend - the blend
  // only applies in scoreLeafFrontier's later pass, which the floor already
  // exceeds here). UP:2's own candidate score is the one this file already
  // established responds to featureBlend, so check that one directly rather
  // than the possibly-floor-dominated overall bestScore.
  const probeKey = encodeAction({ column: 2, orientation: "UP" });

  const withExplicitBlend = analyzeTemplateMove({ board, currentPair, nextQueue: [], settings: { featureBlend: 8 } });
  const withZeroBlend = analyzeTemplateMove({ board, currentPair, nextQueue: [], settings: { featureBlend: 0 } });
  const withDefault = analyzeTemplateMove({ board, currentPair, nextQueue: [] });

  const explicitScore = withExplicitBlend.candidates.find((c) => c.actionKey === probeKey).searchScore;
  const zeroScore = withZeroBlend.candidates.find((c) => c.actionKey === probeKey).searchScore;
  const defaultScore = withDefault.candidates.find((c) => c.actionKey === probeKey).searchScore;

  assert.equal(withExplicitBlend.phase, "safe");
  assert.notEqual(explicitScore, zeroScore);
  assert.equal(withDefault.bestActionKey, withZeroBlend.bestActionKey);
  assert.equal(withDefault.bestScore, withZeroBlend.bestScore);
  assert.equal(defaultScore, zeroScore);
});

test("改善2 (v13 feature blend): battle phase is unaffected by featureBlend", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const withZeroBlend = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    pendingOjama: 24,
    settings: { featureBlend: 0 },
  });
  const withBigBlend = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    pendingOjama: 24,
    settings: { featureBlend: 32 },
  });

  // pendingOjama > 0 keeps this in battle phase regardless of featureBlend,
  // where profile.featureBlend is always 0 - the setting must have zero
  // effect here.
  assert.equal(withZeroBlend.phase, "battle");
  assert.equal(withBigBlend.phase, "battle");
  assert.equal(withZeroBlend.bestActionKey, withBigBlend.bestActionKey);
  assert.equal(withZeroBlend.bestScore, withBigBlend.bestScore);
});

// A "structural preference" test (two safe boards with an equal immediate
// template score but a clearly better/worse v13-style chain skeleton) was
// attempted but not included: the natural way to vary v13's chain-cascade
// signal (e.g. recoloring one of two adjacent 3-groups) turned out to leave
// bestVirtualChain unchanged, because the classic 2-step cascade mechanic is
// color-agnostic - any two adjacent 3-groups of *any* two colors form the
// same cascade geometry, so swapping the second group's color didn't remove
// the structure it was meant to remove. Attempts to instead vary structural
// properties like group liberties ran into the opposite problem: this file's
// own template heuristics (seedScore/groupBonuses) already reward similar
// group-size/liberty properties, so most changes shift both scores together
// rather than isolating the v13-specific contribution. Skipping per the
// task's guidance rather than shipping a flaky or accidentally-tautological
// test; the three tests above already cover on/off, magnitude, and phase
// isolation directly.

test("改善3 (sampled lookahead): deterministic - the same safe fixture analyzed twice matches exactly", () => {
  const board = boardFromRows(["G.....", "G..O.."]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };

  resetTemplateOpeningState();
  const first = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  resetTemplateOpeningState();
  const second = analyzeTemplateMove({ board, currentPair, nextQueue: [] });

  assert.equal(first.phase, "safe");
  assert.equal(first.bestActionKey, second.bestActionKey);
  assert.equal(first.bestScore, second.bestScore);
});

test("改善3 (sampled lookahead): battle phase is unaffected by templateSampleCount", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };

  const withSampling = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    pendingOjama: 24,
    settings: { templateSampleCount: 4 },
  });
  const withoutSampling = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    pendingOjama: 24,
    settings: { templateSampleCount: 0 },
  });

  // pendingOjama > 0 keeps this in battle phase regardless of
  // templateSampleCount, where profile.sampleCount is always 0 - the
  // setting must have zero effect here.
  assert.equal(withSampling.phase, "battle");
  assert.equal(withoutSampling.phase, "battle");
  assert.equal(withSampling.bestActionKey, withoutSampling.bestActionKey);
  assert.equal(withSampling.bestScore, withoutSampling.bestScore);
});

test("改善3 (sampled lookahead): sampling on scores higher than sampling off and flips the decision", () => {
  // Same fixture as the "growth behavior" test above. With sampling off,
  // firing the small ready chain (RIGHT:1) is worth less than patiently
  // building (UP:2, the static leaf's pick). With sampling on, the search
  // looks past the fire and finds enough downstream value there that firing
  // wins instead - a decision flip, and a strictly higher bestScore too.
  const board = boardFromRows(["G.....", "G..O.."]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };

  const samplingOn = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const samplingOff = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { templateSampleCount: 0 },
  });

  assert.equal(samplingOn.phase, "safe");
  assert.ok(samplingOn.bestScore > samplingOff.bestScore);
  assert.notEqual(samplingOn.bestActionKey, samplingOff.bestActionKey);
});

test("改善5 (mid-search refine): templateMidRefine 0 reproduces the pre-改善5 baseline exactly", () => {
  // This exact fixture/score pair (bestActionKey, bestScore) was recorded
  // from this codebase before 改善5 (mid-search refine) existed - a direct
  // regression guard, like the featureBlend-0 pin above. nextQueue has 2
  // entries (3 pieces total) so the search's depth loop actually runs a
  // non-final beam cut, the only place 改善5 can change anything; the safe
  // phase (rootMaxHeight 2, no pressure) is where profile.midRefine is ever
  // nonzero.
  const board = boardFromRows([
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "......",
    "R.....",
    "RG....",
    "RGGB..",
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };
  const nextQueue = [
    { axis: COLORS.BLUE, child: COLORS.YELLOW },
    { axis: COLORS.GREEN, child: COLORS.RED },
  ];

  const analysis = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue,
    settings: { templateMidRefine: 0 },
  });

  assert.equal(analysis.phase, "safe");
  assert.equal(analysis.bestActionKey, "UP:0");
  assert.equal(analysis.bestScore, 223380.2063010609);
});

test("改善5 (mid-search refine): battle phase is unaffected by templateMidRefine", () => {
  const board = boardFromRows([
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
  ]);
  const currentPair = { axis: COLORS.RED, child: COLORS.GREEN };
  const nextQueue = [
    { axis: COLORS.BLUE, child: COLORS.YELLOW },
    { axis: COLORS.GREEN, child: COLORS.RED },
  ];

  const withZeroRefine = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue,
    pendingOjama: 24,
    settings: { templateMidRefine: 0 },
  });
  const withMaxRefine = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue,
    pendingOjama: 24,
    settings: { templateMidRefine: 24 },
  });

  // pendingOjama > 0 keeps this in battle phase regardless of
  // templateMidRefine, where profile.midRefine is always 0 (see LEGACY_
  // GROWTH_PROFILE) - the setting must have zero effect here.
  assert.equal(withZeroRefine.phase, "battle");
  assert.equal(withMaxRefine.phase, "battle");
  assert.equal(withZeroRefine.bestActionKey, withMaxRefine.bestActionKey);
  assert.equal(withZeroRefine.bestScore, withMaxRefine.bestScore);
  assert.deepEqual(withZeroRefine.candidates, withMaxRefine.candidates);
});

test("改善5 (mid-search refine): deterministic - the same safe fixture analyzed twice matches exactly", () => {
  const board = boardFromRows([
    "......",
    "......",
    "..B...",
    "..B...",
    "..YGG.",
    "BBBYR.",
  ]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };
  const nextQueue = [
    { axis: COLORS.GREEN, child: COLORS.RED },
    { axis: COLORS.BLUE, child: COLORS.BLUE },
  ];

  resetTemplateOpeningState();
  const first = analyzeTemplateMove({ board, currentPair, nextQueue });
  resetTemplateOpeningState();
  const second = analyzeTemplateMove({ board, currentPair, nextQueue });

  assert.equal(first.phase, "safe");
  assert.equal(first.bestActionKey, second.bestActionKey);
  assert.equal(first.bestScore, second.bestScore);
});

test("改善5 (mid-search refine): skeleton survival - refining the mid-search beam by virtual-fire potential flips the decision", () => {
  // Found via a scripted random search over safe fixtures (many random
  // boards/queues were tried; this is the first one that produced a clean
  // decision flip) rather than hand-built, since deliberately engineering a
  // board where the cheap per-level sort (shaped + evalValue +
  // chainOutcomeValue*0.01) discards a real skeleton candidate at the
  // depth-1 cut, while also being the eventual overall winner once kept
  // alive, turned out to depend on details of evaluateBoard/virtualFireProbes
  // that are impractical to reason about by hand. Verified directly against
  // this file's own depth-loop internals (temporary instrumentation, since
  // removed) before writing this test: with templateMidRefine off, the
  // eventual winner (UP:2, a candidate ranked ~5th by cheap sort at the
  // depth-1 cut, sortValue ~3219) never even reaches the depth-1 survivors,
  // which are dominated by a candidate under rootIndex 3 (sortValue
  // ~34275); refining the top (beamWidth + midRefine) slice by virtual-fire
  // potential promotes it back in, and it goes on to beat every other line.
  const board = boardFromRows(["..B...", "..B...", "..YGG.", "BBBYR."]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.GREEN };
  const nextQueue = [
    { axis: COLORS.GREEN, child: COLORS.RED },
    { axis: COLORS.BLUE, child: COLORS.BLUE },
  ];

  const refineOff = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue,
    settings: { templateMidRefine: 0 },
  });
  const refineOn = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue,
    settings: { templateMidRefine: 12 },
  });

  assert.equal(refineOff.phase, "safe");
  assert.equal(refineOn.phase, "safe");
  assert.notEqual(refineOff.bestActionKey, refineOn.bestActionKey);
  assert.ok(refineOn.bestScore > refineOff.bestScore);
});

test("改善6 (tunable evaluation weights): DEFAULT_TEMPLATE_WEIGHTS matches the previously-hardcoded scalars", () => {
  // This is the actual regression guard for "no evalWeights = bit-identical
  // behavior": if any of these ever drifted from the values this file's own
  // pinned-score tests were recorded against (e.g. the featureBlend-0 and
  // templateMidRefine-0 baselines above), those tests would fail first - this
  // just documents the mapping directly.
  assert.deepEqual(DEFAULT_TEMPLATE_WEIGHTS, {
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
});

test("改善6 (tunable evaluation weights): settings.evalWeights: {} is behaviorally identical to omitting it", () => {
  const board = boardFromRows(["R.....", "......"]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.BLUE };

  const withoutField = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const withEmpty = analyzeTemplateMove({ board, currentPair, nextQueue: [], settings: { evalWeights: {} } });

  assert.equal(withoutField.bestActionKey, withEmpty.bestActionKey);
  assert.equal(withoutField.bestScore, withEmpty.bestScore);
});

test("改善6 (tunable evaluation weights): an override changes bestScore on a fixture with a real hole", () => {
  // Column 0 holds an R suspended above an empty cell - countHoles(...) sees
  // exactly one hole here, so holePenalty (default -38) actually engages in
  // evaluateBoard's `s += holes * weights.holePenalty` term.
  const board = boardFromRows(["R.....", "......"]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.BLUE };

  const withDefault = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const withOverride = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    settings: { evalWeights: { holePenalty: -80 } },
  });

  assert.notEqual(withOverride.bestScore, withDefault.bestScore);
});

test("改善6 (tunable evaluation weights): non-finite/garbage evalWeights entries are ignored", () => {
  const board = boardFromRows(["R.....", "......"]);
  const currentPair = { axis: COLORS.GREEN, child: COLORS.BLUE };

  const withDefault = analyzeTemplateMove({ board, currentPair, nextQueue: [] });
  const withGarbage = analyzeTemplateMove({
    board,
    currentPair,
    nextQueue: [],
    settings: {
      evalWeights: {
        holePenalty: "not-a-number",
        subFire: Infinity,
        maxHeightSafe: NaN,
        unknownWeightKey: 12345,
      },
    },
  });

  assert.equal(withGarbage.bestActionKey, withDefault.bestActionKey);
  assert.equal(withGarbage.bestScore, withDefault.bestScore);
});
