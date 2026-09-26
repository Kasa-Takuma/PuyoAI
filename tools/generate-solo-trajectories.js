#!/usr/bin/env node
import { createGzip } from "node:zlib";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, stat } from "node:fs/promises";
import { once } from "node:events";
import path from "node:path";
import { isMainThread, parentPort, workerData, Worker } from "node:worker_threads";

import { searchBestMove } from "../src/ai/search.js";
import { soloChainUtility } from "../src/ai/solo-search.js";
import { createEmptyBoard, encodeAction } from "../src/core/board.js";
import { PLAYABLE_COLORS } from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";
import { fromLegacyBoard } from "../src/core/fast-board.js";

const DEFAULT_SEEDS = "training/solo_search/artifacts/seeds/train-pairs.json";
const DEFAULT_OUTPUT = "training/solo_search/artifacts/initial-v13-trajectories.jsonl.gz";
const V13_SETTINGS = Object.freeze({
  depth: 3,
  beamWidth: 48,
  searchProfile: "chain_builder_v13",
  dedupe: true,
  sampleCount: 8,
  sampleDepth: 4,
  sampleBeamWidth: 8,
  sampleTopK: 12,
  sampleWeight: 1,
});

function parseArgs(argv) {
  const args = {
    seedFile: DEFAULT_SEEDS,
    output: DEFAULT_OUTPUT,
    parallel: Math.max(1, Math.min(8, Number(process.env.SOLO_CPU_WORKERS) || 6)),
    maxTurns: 1024,
    limitGames: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const next = argv[index + 1];
    if (argv[index] === "--seeds") {
      args.seedFile = next;
      index += 1;
    } else if (argv[index] === "--out") {
      args.output = next;
      index += 1;
    } else if (argv[index] === "--parallel") {
      args.parallel = Math.max(1, Math.min(16, Number.parseInt(next, 10) || 1));
      index += 1;
    } else if (argv[index] === "--max-turns") {
      args.maxTurns = Math.max(1, Number.parseInt(next, 10) || args.maxTurns);
      index += 1;
    } else if (argv[index] === "--limit-games") {
      args.limitGames = Math.max(1, Number.parseInt(next, 10) || 1);
      index += 1;
    } else if (argv[index] === "--help" || argv[index] === "-h") {
      console.log(`Usage: node tools/generate-solo-trajectories.js [options]

Options:
  --seeds FILE      Fixed train pair file. Default: ${DEFAULT_SEEDS}
  --out FILE        Gzipped JSONL output. Default: ${DEFAULT_OUTPUT}
  --parallel N      Worker threads. Default: 6
  --max-turns N     Placement cap per game. Default: 1024
  --limit-games N   Generate only the first N games for a smoke run.`);
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${argv[index]}`);
    }
  }
  return args;
}

function decodePair(encoded) {
  return {
    axis: PLAYABLE_COLORS[encoded[0] - 1],
    child: PLAYABLE_COLORS[encoded[1] - 1],
  };
}

function boardBase64(board) {
  return Buffer.from(fromLegacyBoard(board)).toString("base64");
}

function encodedPair(pair) {
  return [PLAYABLE_COLORS.indexOf(pair.axis) + 1, PLAYABLE_COLORS.indexOf(pair.child) + 1];
}

function observePending(pending, reward, topout) {
  const finished = [];
  for (const sample of pending) {
    sample.steps += 1;
    sample.targetValue += 0.99 ** (sample.steps - 1) * reward;
    if (topout) sample.deathWithin128 = true;
  }
  while (pending.length > 0 && (topout || pending[0].steps >= 128)) {
    finished.push(pending.shift());
  }
  return finished;
}

function runGame(game, maxTurns) {
  const pairs = game.pairs.map(decodePair);
  let board = createEmptyBoard();
  let totalScore = 0;
  let topout = false;
  const events = [];
  const actions = [];
  const samples = [];
  const pending = [];
  const startedAt = performance.now();
  let executedTurns = 0;

  for (let turnIndex = 0; turnIndex < Math.min(maxTurns, pairs.length - 2); turnIndex += 1) {
    const currentPair = pairs[turnIndex];
    const nextQueue = pairs.slice(turnIndex + 1, turnIndex + 3);
    const analysis = searchBestMove({
      board,
      currentPair,
      nextQueue,
      settings: V13_SETTINGS,
      turn: turnIndex + 1,
      totalScore,
    });
    const result = resolveTurn(board, currentPair, analysis.bestAction);
    executedTurns = turnIndex + 1;
    actions.push(encodeAction(analysis.bestAction));
    totalScore += result.totalScore;
    const reward = result.topout ? -4 : soloChainUtility(result.totalChains);
    samples.push(...observePending(pending, reward, result.topout));
    if (result.totalChains > 0) {
      events.push({
        turn: executedTurns,
        chains: result.totalChains,
        score: result.totalScore,
      });
    }
    if (result.topout) {
      topout = true;
      break;
    }

    board = result.finalBoard;
    pending.push({
      kind: "sample",
      seed: game.seed,
      turn: executedTurns,
      board: boardBase64(board),
      nextPairs: nextQueue.map(encodedPair),
      targetValue: 0,
      deathWithin128: false,
      steps: 0,
    });
  }

  return {
    game: {
      kind: "game",
      seed: game.seed,
      pairSha256: game.sha256,
      executedTurns,
      plannedTurns: maxTurns,
      topout,
      totalScore,
      actions,
      events,
      droppedTrailingSamples: pending.length,
      elapsedMs: performance.now() - startedAt,
    },
    samples,
  };
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, "drain");
}

function runWorker(games, maxTurns) {
  for (const game of games) {
    const result = runGame(game, maxTurns);
    parentPort.postMessage({ type: "game", result });
  }
  parentPort.postMessage({ type: "done" });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const seedPayload = JSON.parse(await readFile(path.resolve(args.seedFile), "utf8"));
  let games = seedPayload.games;
  if (args.limitGames !== null) games = games.slice(0, args.limitGames);
  const workerCount = Math.min(args.parallel, games.length);
  const assignments = Array.from({ length: workerCount }, () => []);
  games.forEach((game, index) => assignments[index % workerCount].push(game));

  const outputPath = path.resolve(args.output);
  await mkdir(path.dirname(outputPath), { recursive: true });
  const gzip = createGzip({ level: 6 });
  const file = createWriteStream(outputPath);
  gzip.pipe(file);
  await writeLine(gzip, {
    kind: "manifest",
    format: "puyoai-solo-trajectories-v1",
    source: "chain_builder_v13",
    settings: V13_SETTINGS,
    horizon: 128,
    discount: 0.99,
    maxTurns: args.maxTurns,
    expectedGames: games.length,
  });

  let completed = 0;
  let sampleCount = 0;
  let highChainEvents = 0;
  let highChainGames = 0;
  const startedAt = performance.now();

  await Promise.all(
    assignments.map(
      (assignment) =>
        new Promise((resolve, reject) => {
          const worker = new Worker(new URL(import.meta.url), {
            workerData: { games: assignment, maxTurns: args.maxTurns },
          });
          let writeQueue = Promise.resolve();
          worker.on("message", (message) => {
            if (message.type !== "game") return;
            writeQueue = writeQueue.then(async () => {
              for (const sample of message.result.samples) await writeLine(gzip, sample);
              await writeLine(gzip, message.result.game);
              completed += 1;
              sampleCount += message.result.samples.length;
              const high = message.result.game.events.filter((event) => event.chains >= 10).length;
              highChainEvents += high;
              if (high > 0) highChainGames += 1;
              process.stderr.write(
                `${JSON.stringify({ stage: "game", completed, games: games.length, sampleCount, highChainEvents })}\n`,
              );
            });
            writeQueue.catch(reject);
          });
          worker.once("error", reject);
          worker.once("exit", async (code) => {
            if (code !== 0) {
              reject(new Error(`Trajectory worker exited with ${code}.`));
              return;
            }
            try {
              await writeQueue;
              resolve();
            } catch (error) {
              reject(error);
            }
          });
        }),
    ),
  );

  await writeLine(gzip, {
    kind: "summary",
    completedGames: completed,
    sampleCount,
    highChainEvents,
    highChainGames,
    elapsedMs: performance.now() - startedAt,
  });
  gzip.end();
  await once(file, "close");
  const outputStat = await stat(outputPath);
  console.log(
    JSON.stringify({
      output: outputPath,
      bytes: outputStat.size,
      completedGames: completed,
      sampleCount,
      highChainEvents,
      highChainGames,
      elapsedMs: performance.now() - startedAt,
    }),
  );
}

if (isMainThread) {
  await main();
} else {
  runWorker(workerData.games, workerData.maxTurns);
}
