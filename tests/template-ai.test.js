import test, { beforeEach } from "node:test";
import assert from "node:assert/strict";

import { analyzeTemplateMove, resetTemplateOpeningState, simulateOjamaSettle } from "../src/ai/template-ai.js";
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
