#!/usr/bin/env node
// Headless ppsim2-like battle harness: pits two template-ai configurations
// ("A" and "B") against each other under deterministic, seeded vs-puyo rules
// for A/B strength measurement. Mirrors ppsim2/puyoSim.js's turn/offset/drop
// sequence (see analyzeTemplateMove's own battle-aware comments in
// src/ai/template-ai.js for the same rules applied inside the search).
import { cloneBoard, createEmptyBoard } from "../src/core/board.js";
import { BOARD_HEIGHT, BOARD_WIDTH, COLORS, TOP_OUT_COLUMN, TOP_OUT_ROW } from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";
import { createRng, fillQueue } from "../src/core/randomizer.js";
import { analyzeTemplateMove, resetTemplateOpeningState } from "../src/ai/template-ai.js";

const DEFAULT_GAMES = 40;
const DEFAULT_SEED = 1000;
const DEFAULT_MAX_ROUNDS = 300;
const NUISANCE_TARGET_POINTS = 70;
const ALL_CLEAR_ATTACK_BONUS = 2100;
const MAX_OJAMA_DROP_PER_TURN = 30;

function parseIntArg(text, fallback) {
  const parsed = Number.parseInt(text, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseJsonArg(text, flagName) {
  if (text === undefined) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    console.error(`Invalid JSON for ${flagName}: ${error.message}`);
    process.exit(1);
  }
}

function parseArgs(argv) {
  const args = {
    games: DEFAULT_GAMES,
    seed: DEFAULT_SEED,
    aSettings: {},
    bSettings: {},
    maxRounds: DEFAULT_MAX_ROUNDS,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--games") {
      args.games = Math.max(1, parseIntArg(next, args.games));
      index += 1;
    } else if (arg === "--seed") {
      args.seed = parseIntArg(next, args.seed);
      index += 1;
    } else if (arg === "--a-settings") {
      args.aSettings = parseJsonArg(next, "--a-settings");
      index += 1;
    } else if (arg === "--b-settings") {
      args.bSettings = parseJsonArg(next, "--b-settings");
      index += 1;
    } else if (arg === "--max-rounds") {
      args.maxRounds = Math.max(1, parseIntArg(next, args.maxRounds));
      index += 1;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  return args;
}

function printHelp() {
  console.log(`Usage:
  node tools/battle-harness.js [options]

Options:
  --games N          Number of games to play. Default: ${DEFAULT_GAMES}
  --seed N           Base seed. Game g uses seed <seed + g>. Default: ${DEFAULT_SEED}
  --a-settings JSON  template-ai settings object for side A. Default: {}
  --b-settings JSON  template-ai settings object for side B. Default: {}
  --max-rounds N     Round cap before a game is scored a draw. Default: ${DEFAULT_MAX_ROUNDS}`);
}

// Fisher-Yates shuffle of the 6 columns, driven by the harness's own seeded
// rng so the garbage spread is reproducible independent of either player's
// piece rng.
function shuffleColumns(rng) {
  const order = [0, 1, 2, 3, 4, 5];
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = rng.nextInt(i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

function columnTop(board, x) {
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    if (board[y][x] === COLORS.EMPTY) {
      return y;
    }
  }
  return BOARD_HEIGHT;
}

function countEmptyCells(board) {
  let empty = 0;
  for (let x = 0; x < BOARD_WIDTH; x += 1) {
    empty += BOARD_HEIGHT - columnTop(board, x);
  }
  return empty;
}

// Mirrors puyoSim's dropOjamaToBoard: repeatedly shuffles the 6 columns and
// drops one garbage cell per column (in shuffled order) until `dropCount` is
// exhausted; the final partial round only uses the first r columns of a
// fresh shuffle.
export function dropOjamaToBoard(board, dropCount, rng) {
  let remaining = dropCount;
  while (remaining > 0) {
    const order = shuffleColumns(rng);
    const roundSize = Math.min(BOARD_WIDTH, remaining);
    for (let i = 0; i < roundSize; i += 1) {
      const column = order[i];
      const y = columnTop(board, column);
      if (y < BOARD_HEIGHT) {
        board[y][column] = COLORS.GARBAGE;
      }
    }
    remaining -= roundSize;
  }
  return board;
}

function createPlayer(label, instanceId, settings, pieceSeed) {
  return {
    label,
    instanceId,
    baseSettings: settings,
    board: createEmptyBoard(),
    pendingOjama: 0,
    nuisanceBuffer: 0,
    queue: [],
    rng: createRng(pieceSeed),
    maxChain: 0,
    totalSent: 0,
  };
}

function drawPair(player) {
  fillQueue(player.rng, player.queue, 3);
  const currentPair = player.queue.shift();
  const nextQueue = player.queue.slice(0, 2);
  return { currentPair, nextQueue };
}

function makeMove(mover, opponent, garbageRng) {
  const { currentPair, nextQueue } = drawPair(mover);
  const settings = structuredClone(mover.baseSettings);
  settings.instanceId = mover.instanceId;

  const analysis = analyzeTemplateMove({
    board: mover.board,
    currentPair,
    nextQueue,
    settings,
    pendingOjama: mover.pendingOjama,
    opponent: { board: cloneBoard(opponent.board), pendingOjama: opponent.pendingOjama },
  });

  if (!analysis.bestAction) {
    return { lost: true, reason: "no-action" };
  }

  const result = resolveTurn(mover.board, currentPair, analysis.bestAction);
  if (result.topout) {
    return { lost: true, reason: "topout" };
  }

  mover.board = result.finalBoard;
  mover.maxChain = Math.max(mover.maxChain, result.totalChains);

  const attackScore = result.totalScore + (result.allClear ? ALL_CLEAR_ATTACK_BONUS : 0);
  mover.nuisanceBuffer += attackScore / NUISANCE_TARGET_POINTS;
  const attack = Math.floor(mover.nuisanceBuffer);
  mover.nuisanceBuffer -= attack;

  const canceled = Math.min(mover.pendingOjama, attack);
  mover.pendingOjama -= canceled;
  const outgoing = attack - canceled;
  opponent.pendingOjama += outgoing;
  mover.totalSent += outgoing;

  const drop = Math.min(mover.pendingOjama, MAX_OJAMA_DROP_PER_TURN);
  mover.pendingOjama -= drop;

  if (drop > 0) {
    if (drop > countEmptyCells(mover.board)) {
      return { lost: true, reason: "garbage-overflow" };
    }
    dropOjamaToBoard(mover.board, drop, garbageRng);
    if (mover.board[TOP_OUT_ROW][TOP_OUT_COLUMN] !== COLORS.EMPTY) {
      return { lost: true, reason: "garbage-topout" };
    }
  }

  return { lost: false };
}

// Runs a single game between side A and side B. `seed` seeds both players'
// piece rngs identically (same tsumo sequence, standard in vs puyo) and
// derives the harness's own garbage-spread rng.
export function runMatch({
  seed,
  aSettings = {},
  bSettings = {},
  maxRounds = DEFAULT_MAX_ROUNDS,
  firstIsA = true,
  aInstanceId = "A",
  bInstanceId = "B",
}) {
  const garbageRng = createRng(`${seed}:garbage`);
  const playerA = createPlayer("A", aInstanceId, aSettings, seed);
  const playerB = createPlayer("B", bInstanceId, bSettings, seed);

  resetTemplateOpeningState();

  let winner = "draw";
  let reason = "max-rounds";
  let roundsPlayed = maxRounds;

  roundLoop: for (let round = 0; round < maxRounds; round += 1) {
    const order = firstIsA ? [playerA, playerB] : [playerB, playerA];
    for (const mover of order) {
      const opponent = mover === playerA ? playerB : playerA;
      const outcome = makeMove(mover, opponent, garbageRng);
      if (outcome.lost) {
        winner = opponent.label;
        reason = outcome.reason;
        roundsPlayed = round + 1;
        break roundLoop;
      }
    }
  }

  return {
    seed,
    firstIsA,
    winner,
    reason,
    rounds: roundsPlayed,
    aMaxChain: playerA.maxChain,
    bMaxChain: playerB.maxChain,
    aSent: playerA.totalSent,
    bSent: playerB.totalSent,
  };
}

// Runs `games` matches, alternating which side moves first per game, and
// aggregates per-side win/loss/draw and chain/ojama stats.
export function runGames({
  games = DEFAULT_GAMES,
  seed = DEFAULT_SEED,
  aSettings = {},
  bSettings = {},
  maxRounds = DEFAULT_MAX_ROUNDS,
}) {
  const perGame = [];
  const aggregate = {
    a: { wins: 0, losses: 0, draws: 0, maxChains: [], sent: [] },
    b: { wins: 0, losses: 0, draws: 0, maxChains: [], sent: [] },
    rounds: [],
  };

  for (let g = 0; g < games; g += 1) {
    const gameSeed = seed + g;
    const firstIsA = g % 2 === 0;
    const result = runMatch({ seed: gameSeed, aSettings, bSettings, maxRounds, firstIsA });
    perGame.push({ index: g, ...result });

    aggregate.a.maxChains.push(result.aMaxChain);
    aggregate.b.maxChains.push(result.bMaxChain);
    aggregate.a.sent.push(result.aSent);
    aggregate.b.sent.push(result.bSent);
    aggregate.rounds.push(result.rounds);

    if (result.winner === "A") {
      aggregate.a.wins += 1;
      aggregate.b.losses += 1;
    } else if (result.winner === "B") {
      aggregate.b.wins += 1;
      aggregate.a.losses += 1;
    } else {
      aggregate.a.draws += 1;
      aggregate.b.draws += 1;
    }
  }

  return { games, seed, maxRounds, aSettings, bSettings, perGame, aggregate };
}

function average(values) {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function max(values) {
  return values.length === 0 ? 0 : Math.max(...values);
}

function round(value, digits = 2) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : value;
}

function summarizeSide(label, side, games) {
  return {
    label,
    wins: side.wins,
    losses: side.losses,
    draws: side.draws,
    winRate: games > 0 ? side.wins / games : 0,
    avgMaxChain: average(side.maxChains),
    maxMaxChain: max(side.maxChains),
    avgSent: average(side.sent),
  };
}

function printTable(report) {
  const summaryA = summarizeSide("A", report.aggregate.a, report.games);
  const summaryB = summarizeSide("B", report.aggregate.b, report.games);
  const avgRounds = average(report.aggregate.rounds);
  const maxRoundsSeen = max(report.aggregate.rounds);

  console.table(
    [summaryA, summaryB].map((summary) => ({
      side: summary.label,
      wins: summary.wins,
      losses: summary.losses,
      draws: summary.draws,
      winRate: round(summary.winRate, 3),
      avgMaxChain: round(summary.avgMaxChain, 2),
      maxMaxChain: summary.maxMaxChain,
      avgSent: round(summary.avgSent, 2),
    })),
  );
  console.log(`avg rounds/game: ${round(avgRounds, 2)}, max rounds seen: ${maxRoundsSeen}`);

  return { summaryA, summaryB, avgRounds, maxRoundsSeen };
}

function runMain() {
  const args = parseArgs(process.argv.slice(2));

  console.log(
    `Running ${args.games} game(s), seed=${args.seed}, maxRounds=${args.maxRounds}`,
  );
  console.log(`A settings: ${JSON.stringify(args.aSettings)}`);
  console.log(`B settings: ${JSON.stringify(args.bSettings)}`);

  const startedAt = performance.now();
  const report = runGames(args);
  const elapsedMs = performance.now() - startedAt;

  const { summaryA, summaryB, avgRounds, maxRoundsSeen } = printTable(report);
  console.log(`elapsed: ${Math.round(elapsedMs)}ms`);

  const resultJson = {
    kind: "puyoai_battle_harness_report",
    version: 1,
    games: args.games,
    seed: args.seed,
    maxRounds: args.maxRounds,
    aSettings: args.aSettings,
    bSettings: args.bSettings,
    a: summaryA,
    b: summaryB,
    avgRoundsPerGame: avgRounds,
    maxRoundsSeen,
    elapsedMs,
  };
  console.log(`RESULT_JSON:${JSON.stringify(resultJson)}`);
}

const isMainModule = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  runMain();
}
