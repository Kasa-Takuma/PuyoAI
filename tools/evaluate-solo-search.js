#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { isMainThread, parentPort, workerData, Worker } from "node:worker_threads";

import { searchBestMove } from "../src/ai/search.js";
import {
  pairedBootstrap,
  summarizeLatency,
  summarizeSoloGames,
} from "../src/ai/solo-evaluation.js";
import { searchSoloMove } from "../src/ai/solo-search.js";
import { hydrateSoloValueModel } from "../src/ai/solo-value.js";
import { createEmptyBoard, encodeAction } from "../src/core/board.js";
import { PLAYABLE_COLORS } from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";

const ENGINES = new Set(["solo", "v13", "v13-nosample"]);
const V13_BASE = Object.freeze({
  depth: 3,
  beamWidth: 48,
  searchProfile: "chain_builder_v13",
  dedupe: true,
  sampleDepth: 4,
  sampleBeamWidth: 8,
  sampleTopK: 12,
  sampleWeight: 1,
});

function parseArgs(argv) {
  const args = {
    engine: "solo",
    split: "dev",
    seedFile: null,
    model: "models/solo_value.web.json",
    output: null,
    compare: null,
    parallel: Math.max(1, Math.min(8, Number(process.env.SOLO_CPU_WORKERS) || 6)),
    limitGames: null,
    maxTurns: 1000,
    beamWidth: 32,
    preserveRootActions: true,
    valueAtLeafOnly: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const next = argv[index + 1];
    const arg = argv[index];
    if (arg === "--engine") {
      args.engine = next;
      index += 1;
    } else if (arg === "--split") {
      args.split = next;
      index += 1;
    } else if (arg === "--seeds") {
      args.seedFile = next;
      index += 1;
    } else if (arg === "--model") {
      args.model = next;
      index += 1;
    } else if (arg === "--out") {
      args.output = next;
      index += 1;
    } else if (arg === "--compare") {
      args.compare = next;
      index += 1;
    } else if (arg === "--parallel") {
      args.parallel = Math.max(1, Math.min(16, Number.parseInt(next, 10) || 1));
      index += 1;
    } else if (arg === "--limit-games") {
      args.limitGames = Math.max(1, Number.parseInt(next, 10) || 1);
      index += 1;
    } else if (arg === "--max-turns") {
      args.maxTurns = Math.max(1, Math.min(1000, Number.parseInt(next, 10) || 1000));
      index += 1;
    } else if (arg === "--beam") {
      args.beamWidth = Math.max(22, Math.min(64, Number.parseInt(next, 10) || 32));
      index += 1;
    } else if (arg === "--no-root-preservation") {
      args.preserveRootActions = false;
    } else if (arg === "--value-at-leaf-only") {
      args.valueAtLeafOnly = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!ENGINES.has(args.engine)) {
    throw new Error(`Unknown engine ${args.engine}. Use solo, v13, or v13-nosample.`);
  }
  args.seedFile ??= `training/solo_search/artifacts/seeds/${args.split}-pairs.json`;
  args.output ??= `training/solo_search/artifacts/evaluation/${args.split}-${args.engine}.json`;
  return args;
}

function printHelp() {
  console.log(`Usage: node tools/evaluate-solo-search.js [options]

Options:
  --engine NAME        solo, v13, or v13-nosample. Default: solo
  --split NAME         Seed split used in default paths. Default: dev
  --seeds FILE         Fixed pair file.
  --model FILE         Solo web model. Default: models/solo_value.web.json
  --out FILE           Atomic JSON report output.
  --compare FILE       Add paired bootstrap against another report.
  --parallel N         Worker threads. Default: 6
  --limit-games N      Evaluate only the first N games.
  --max-turns N        Planned and maximum turns. Default: 1000
  --beam N             Solo beam width: 32, 48, or 64. Default: 32
  --no-root-preservation  Use a global beam for the structural ablation.
  --value-at-leaf-only    Use the fixed simple score for intermediate pruning.`);
}

function sha256(content) {
  return createHash("sha256").update(content).digest("hex");
}

function decodePair(encoded) {
  return {
    axis: PLAYABLE_COLORS[encoded[0] - 1],
    child: PLAYABLE_COLORS[encoded[1] - 1],
  };
}

function engineSettings(args) {
  if (args.engine === "solo") {
    return {
      depth: 3,
      beamWidth: args.beamWidth,
      preserveRootActions: args.preserveRootActions,
      valueAtLeafOnly: args.valueAtLeafOnly,
      dedupe: true,
    };
  }
  return {
    ...V13_BASE,
    sampleCount: args.engine === "v13" ? 8 : 0,
  };
}

function runGame(game, config, rawModel) {
  const pairs = game.pairs.map(decodePair);
  const model = rawModel ? hydrateSoloValueModel(rawModel) : null;
  let board = createEmptyBoard();
  let totalScore = 0;
  let topout = false;
  let executedTurns = 0;
  const events = [];
  const actions = [];
  const searchMs = [];

  for (let turnIndex = 0; turnIndex < Math.min(config.maxTurns, pairs.length - 2); turnIndex += 1) {
    const currentPair = pairs[turnIndex];
    const nextQueue = pairs.slice(turnIndex + 1, turnIndex + 3);
    const analysis =
      config.engine === "solo"
        ? searchSoloMove({
            board,
            currentPair,
            nextQueue,
            settings: config.settings,
            model,
          })
        : searchBestMove({
            board,
            currentPair,
            nextQueue,
            settings: config.settings,
            turn: turnIndex + 1,
            totalScore,
          });
    if (!analysis.bestAction) break;
    const result = resolveTurn(board, currentPair, analysis.bestAction);
    executedTurns = turnIndex + 1;
    searchMs.push(analysis.elapsedMs);
    actions.push(encodeAction(analysis.bestAction));
    totalScore += result.totalScore;
    if (result.totalChains > 0) {
      events.push({ turn: executedTurns, chains: result.totalChains, score: result.totalScore });
    }
    if (result.topout) {
      topout = true;
      break;
    }
    board = result.finalBoard;
  }

  return {
    seed: game.seed,
    pairSha256: game.sha256,
    plannedTurns: config.maxTurns,
    executedTurns,
    topout,
    totalScore,
    events,
    actions,
    searchMs,
  };
}

function runWorker() {
  for (const game of workerData.games) {
    parentPort.postMessage({
      type: "game",
      game: runGame(game, workerData.config, workerData.rawModel),
    });
  }
}

async function runParallel(games, config, rawModel, parallel) {
  const workerCount = Math.min(games.length, parallel);
  const assignments = Array.from({ length: workerCount }, () => []);
  games.forEach((game, index) => assignments[index % workerCount].push(game));
  const results = [];
  let completed = 0;
  await Promise.all(
    assignments.map(
      (assignment) =>
        new Promise((resolve, reject) => {
          const worker = new Worker(new URL(import.meta.url), {
            workerData: { games: assignment, config, rawModel },
          });
          worker.on("message", (message) => {
            if (message.type !== "game") return;
            results.push(message.game);
            completed += 1;
            process.stderr.write(
              `${JSON.stringify({ stage: "game", completed, games: games.length, seed: message.game.seed })}\n`,
            );
          });
          worker.once("error", reject);
          worker.once("exit", (code) => {
            if (code === 0) resolve();
            else reject(new Error(`Evaluation worker exited with ${code}.`));
          });
        }),
    ),
  );
  const resultBySeed = new Map(results.map((game) => [game.seed, game]));
  return games.map((game) => resultBySeed.get(game.seed));
}

export function validateEvaluationGames(games, expectedGames, maxTurns) {
  if (games.length !== expectedGames || games.some((game) => !game)) {
    throw new Error(`Expected ${expectedGames} completed games, got ${games.filter(Boolean).length}.`);
  }
  const seeds = new Set();
  for (const game of games) {
    if (seeds.has(game.seed)) throw new Error(`Duplicate game seed: ${game.seed}`);
    seeds.add(game.seed);
    if (game.executedTurns !== game.actions.length || game.executedTurns !== game.searchMs.length) {
      throw new Error(`Non-contiguous turn log for ${game.seed}.`);
    }
    if (!game.topout && game.executedTurns !== maxTurns) {
      throw new Error(`Incomplete non-terminal game ${game.seed}: ${game.executedTurns}/${maxTurns}.`);
    }
    if (game.events.some((event) => event.turn < 1 || event.turn > game.executedTurns)) {
      throw new Error(`Event outside executed turn range for ${game.seed}.`);
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const seedText = await readFile(path.resolve(args.seedFile), "utf8");
  const seedPayload = JSON.parse(seedText);
  let games = seedPayload.games;
  if (args.limitGames !== null) games = games.slice(0, args.limitGames);
  const settings = engineSettings(args);
  let rawModel = null;
  let modelSha256 = null;
  if (args.engine === "solo") {
    const modelText = await readFile(path.resolve(args.model), "utf8");
    rawModel = JSON.parse(modelText);
    hydrateSoloValueModel(rawModel);
    modelSha256 = sha256(modelText);
  }
  const config = {
    engine: args.engine,
    split: args.split,
    maxTurns: args.maxTurns,
    settings,
  };
  const startedAt = performance.now();
  const perGame = await runParallel(games, config, rawModel, args.parallel);
  validateEvaluationGames(perGame, games.length, args.maxTurns);
  const summary = summarizeSoloGames(perGame, args.maxTurns);
  const latency = summarizeLatency(perGame.flatMap((game) => game.searchMs));
  let comparison = null;
  if (args.compare) {
    const baseline = JSON.parse(await readFile(path.resolve(args.compare), "utf8"));
    const baselineBySeed = new Map(baseline.perGame.map((game) => [game.seed, game]));
    const pairedBaseline = perGame.map((game) => baselineBySeed.get(game.seed));
    if (pairedBaseline.some((game) => !game)) {
      throw new Error("Comparison report does not contain every evaluated seed.");
    }
    comparison = pairedBootstrap(summary.perGame, summarizeSoloGames(pairedBaseline, args.maxTurns).perGame);
  }

  const report = {
    format: "puyoai-solo-evaluation-v1",
    config,
    seedFile: path.resolve(args.seedFile),
    seedFileSha256: sha256(seedText),
    modelPath: args.engine === "solo" ? path.resolve(args.model) : null,
    modelSha256,
    expectedGames: games.length,
    completedGames: perGame.length,
    elapsedMs: performance.now() - startedAt,
    latency,
    summary: { ...summary, perGame: undefined },
    comparison,
    perGame,
  };
  const outputPath = path.resolve(args.output);
  await mkdir(path.dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.tmp-${process.pid}`;
  await writeFile(temporaryPath, `${JSON.stringify(report, null, 2)}\n`);
  await rename(temporaryPath, outputPath);
  console.log(JSON.stringify({ output: outputPath, summary: report.summary, latency, comparison }, null, 2));
}

if (isMainThread) await main();
else runWorker();
