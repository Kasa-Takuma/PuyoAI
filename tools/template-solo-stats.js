#!/usr/bin/env node
// Solo chain-statistics tool: plays template-ai against an empty board (no
// opponent, no garbage) and records chain-firing stats. Useful for tuning a
// single template-ai configuration without battle-rule noise.
import { createEmptyBoard } from "../src/core/board.js";
import { resolveTurn } from "../src/core/engine.js";
import { createRng, fillQueue } from "../src/core/randomizer.js";
import { analyzeTemplateMove, resetTemplateOpeningState } from "../src/ai/template-ai.js";

const DEFAULT_GAMES = 60;
const DEFAULT_SEED = 42;
const DEFAULT_MAX_MOVES = 120;

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
    maxMoves: DEFAULT_MAX_MOVES,
    settings: {},
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
    } else if (arg === "--max-moves") {
      args.maxMoves = Math.max(1, parseIntArg(next, args.maxMoves));
      index += 1;
    } else if (arg === "--settings") {
      args.settings = parseJsonArg(next, "--settings");
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
  node tools/template-solo-stats.js [options]

Options:
  --games N      Number of solo games to play. Default: ${DEFAULT_GAMES}
  --seed N       Base seed. Game g uses seed <seed + g>. Default: ${DEFAULT_SEED}
  --max-moves N  Move cap per game. Default: ${DEFAULT_MAX_MOVES}
  --settings JSON template-ai settings object. Default: {}`);
}

function drawPair(rng, queue) {
  fillQueue(rng, queue, 3);
  const currentPair = queue.shift();
  const nextQueue = queue.slice(0, 2);
  return { currentPair, nextQueue };
}

// Plays one solo game to completion (topout, no legal action, or the move
// cap). Returns per-game stats; no opponent, no garbage.
export function runSoloGame({ seed, maxMoves = DEFAULT_MAX_MOVES, settings = {} }) {
  const rng = createRng(seed);
  const queue = [];
  let board = createEmptyBoard();

  resetTemplateOpeningState();

  let maxChain = 0;
  let moves = 0;
  let topout = false;

  for (; moves < maxMoves; moves += 1) {
    const { currentPair, nextQueue } = drawPair(rng, queue);
    const analysis = analyzeTemplateMove({
      board,
      currentPair,
      nextQueue,
      settings,
      pendingOjama: 0,
      opponent: null,
    });

    if (!analysis.bestAction) {
      break;
    }

    const result = resolveTurn(board, currentPair, analysis.bestAction);
    if (result.topout) {
      topout = true;
      break;
    }

    board = result.finalBoard;
    maxChain = Math.max(maxChain, result.totalChains);
  }

  return { seed, moves, maxChain, topout };
}

export function runSoloGames({ games = DEFAULT_GAMES, seed = DEFAULT_SEED, maxMoves = DEFAULT_MAX_MOVES, settings = {} }) {
  const perGame = [];
  const aggregate = {
    maxChains: [],
    moves: [],
    topouts: 0,
    atLeast5: 0,
    atLeast7: 0,
    atLeast10: 0,
  };

  for (let g = 0; g < games; g += 1) {
    const gameSeed = seed + g;
    const result = runSoloGame({ seed: gameSeed, maxMoves, settings });
    perGame.push({ index: g, ...result });

    aggregate.maxChains.push(result.maxChain);
    aggregate.moves.push(result.moves);
    if (result.topout) {
      aggregate.topouts += 1;
    }
    if (result.maxChain >= 5) {
      aggregate.atLeast5 += 1;
    }
    if (result.maxChain >= 7) {
      aggregate.atLeast7 += 1;
    }
    if (result.maxChain >= 10) {
      aggregate.atLeast10 += 1;
    }
  }

  return { games, seed, maxMoves, settings, perGame, aggregate };
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

function printTable(report) {
  const { aggregate, games } = report;
  const summary = {
    games,
    avgMaxChain: round(average(aggregate.maxChains), 2),
    maxMaxChain: max(aggregate.maxChains),
    avgMoves: round(average(aggregate.moves), 2),
    topouts: aggregate.topouts,
    "chains>=5": aggregate.atLeast5,
    "chains>=7": aggregate.atLeast7,
    "chains>=10": aggregate.atLeast10,
  };
  console.table([summary]);
  return summary;
}

function runMain() {
  const args = parseArgs(process.argv.slice(2));

  console.log(
    `Running ${args.games} solo game(s), seed=${args.seed}, maxMoves=${args.maxMoves}`,
  );
  console.log(`settings: ${JSON.stringify(args.settings)}`);

  const startedAt = performance.now();
  const report = runSoloGames(args);
  const elapsedMs = performance.now() - startedAt;

  const summary = printTable(report);
  console.log(`elapsed: ${Math.round(elapsedMs)}ms`);

  const resultJson = {
    kind: "puyoai_template_solo_stats_report",
    version: 1,
    games: args.games,
    seed: args.seed,
    maxMoves: args.maxMoves,
    settings: args.settings,
    summary,
    elapsedMs,
  };
  console.log(`RESULT_JSON:${JSON.stringify(resultJson)}`);
}

const isMainModule = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  runMain();
}
