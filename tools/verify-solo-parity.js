#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { searchSoloMove } from "../src/ai/solo-search.js";
import {
  encodeSoloValueInputFast,
  evaluateSoloValueInput,
  hydrateSoloValueModel,
} from "../src/ai/solo-value.js";
import { createEmptyBoard } from "../src/core/board.js";
import { PLAYABLE_COLORS } from "../src/core/constants.js";
import { resolveTurn } from "../src/core/engine.js";
import { fromLegacyBoard } from "../src/core/fast-board.js";

const ACTIONS = [
  ...Array.from({ length: 6 }, (_, column) => [
    { column, orientation: "UP" },
    { column, orientation: "DOWN" },
  ]).flat(),
  ...Array.from({ length: 5 }, (_, column) => ({ column, orientation: "RIGHT" })),
  ...Array.from({ length: 5 }, (_, index) => ({ column: index + 1, orientation: "LEFT" })),
];

function parseArgs(argv) {
  const args = {
    model: "models/solo_value.web.json",
    seeds: "training/solo_search/artifacts/seeds/dev-pairs.json",
    games: 32,
    turns: 1000,
    predictionInputs: 10_000,
    python: ".venv/bin/python",
    beamWidth: 32,
    requireSearchParity: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const next = argv[index + 1];
    if (argv[index] === "--model") args.model = next;
    else if (argv[index] === "--seeds") args.seeds = next;
    else if (argv[index] === "--games") args.games = Number.parseInt(next, 10);
    else if (argv[index] === "--turns") args.turns = Number.parseInt(next, 10);
    else if (argv[index] === "--prediction-inputs") args.predictionInputs = Number.parseInt(next, 10);
    else if (argv[index] === "--python") args.python = next;
    else if (argv[index] === "--beam") args.beamWidth = Number.parseInt(next, 10);
    else if (argv[index] === "--require-search-parity") args.requireSearchParity = true;
    else if (argv[index] === "--help" || argv[index] === "-h") {
      console.log("Usage: node tools/verify-solo-parity.js [--model FILE] [--games 32] [--turns 1000] [--require-search-parity]");
      process.exit(0);
    } else throw new Error(`Unknown argument: ${argv[index]}`);
    if (!argv[index].startsWith("--") || ["--help", "-h"].includes(argv[index])) continue;
    index += 1;
  }
  return args;
}

function decodePair(pair) {
  return { axis: PLAYABLE_COLORS[pair[0] - 1], child: PLAYABLE_COLORS[pair[1] - 1] };
}

function boardBase64(board) {
  return Buffer.from(fromLegacyBoard(board)).toString("base64");
}

function pairCodes(pair) {
  return [PLAYABLE_COLORS.indexOf(pair.axis) + 1, PLAYABLE_COLORS.indexOf(pair.child) + 1];
}

async function createBridge(args) {
  const child = spawn(args.python, ["-m", "training.solo_search.bridge", "--model", path.resolve(args.model)], {
    cwd: process.cwd(),
    env: { ...process.env, JAX_PLATFORM_NAME: process.env.JAX_PLATFORM_NAME ?? "cpu" },
    stdio: ["pipe", "pipe", "inherit"],
  });
  const lines = createInterface({ input: child.stdout });
  const pending = new Map();
  let nextId = 1;
  lines.on("line", (line) => {
    const message = JSON.parse(line);
    if (message.type === "ready") {
      pending.get("ready")?.resolve(message);
      pending.delete("ready");
      return;
    }
    const waiter = pending.get(message.id);
    if (!waiter) return;
    pending.delete(message.id);
    if (message.error) waiter.reject(new Error(message.error));
    else waiter.resolve(message);
  });
  child.once("exit", (code) => {
    for (const waiter of pending.values()) waiter.reject(new Error(`JAX bridge exited with ${code}.`));
  });
  const ready = await new Promise((resolve, reject) => {
    pending.set("ready", { resolve, reject });
  });
  return {
    ready,
    request(payload) {
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        child.stdin.write(`${JSON.stringify({ id, ...payload })}\n`);
      });
    },
    close() {
      child.stdin.end();
    },
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const model = hydrateSoloValueModel(JSON.parse(await readFile(args.model, "utf8")));
  const seedPayload = JSON.parse(await readFile(args.seeds, "utf8"));
  const games = seedPayload.games.slice(0, args.games).map((game) => ({
    seed: game.seed,
    pairs: game.pairs.map(decodePair),
    board: createEmptyBoard(),
    active: true,
  }));
  const bridge = await createBridge(args);
  const predictionCases = [];
  let decisions = 0;
  let firstDifference = null;
  let maxSearchScoreAbsError = 0;
  let maxSearchScoreToleranceRatio = 0;

  for (let turnIndex = 0; turnIndex < args.turns && games.some((game) => game.active); turnIndex += 1) {
    const jsAnalyses = games.map((game) => {
      const pairs = game.pairs.slice(turnIndex, turnIndex + 3);
      if (!game.active || pairs.length < 3) return null;
      return searchSoloMove({
        board: game.board,
        currentPair: pairs[0],
        nextQueue: pairs.slice(1),
        settings: { beamWidth: args.beamWidth },
        model,
      });
    });
    const response = await bridge.request({
      type: "search",
      boards: games.map((game) => boardBase64(game.board)),
      pairs: games.map((game) =>
        game.pairs.slice(turnIndex, turnIndex + 3).map(pairCodes),
      ),
      beamWidth: args.beamWidth,
    });

    for (let gameIndex = 0; gameIndex < games.length; gameIndex += 1) {
      const game = games[gameIndex];
      const analysis = jsAnalyses[gameIndex];
      if (!game.active || !analysis) continue;
      decisions += 1;
      const jaxAction = response.actions[gameIndex];
      const jsAction = ACTIONS.findIndex(
        (action) =>
          action.column === analysis.bestAction.column &&
          action.orientation === analysis.bestAction.orientation,
      );
      if (jsAction !== jaxAction && firstDifference === null) {
        firstDifference = {
          seed: game.seed,
          turn: turnIndex + 1,
          board: boardBase64(game.board),
          pairs: game.pairs.slice(turnIndex, turnIndex + 3),
          jsAction,
          jaxAction,
          jsCandidates: analysis.candidates.slice(0, 5),
          jaxRootScores: response.rootScores[gameIndex],
        };
      }
      for (const candidate of analysis.candidates) {
        const candidateAction = ACTIONS.findIndex(
          (action) =>
            action.column === candidate.action.column &&
            action.orientation === candidate.action.orientation,
        );
        const reference = response.rootScores[gameIndex][candidateAction];
        const error = Math.abs(candidate.searchScore - reference);
        const tolerance = 0.05 + 1e-4 * Math.abs(reference);
        maxSearchScoreAbsError = Math.max(maxSearchScoreAbsError, error);
        maxSearchScoreToleranceRatio = Math.max(
          maxSearchScoreToleranceRatio,
          error / tolerance,
        );
      }
      if (predictionCases.length < args.predictionInputs) {
        predictionCases.push({
          board: fromLegacyBoard(game.board),
          nextPairs: game.pairs.slice(turnIndex + 1, turnIndex + 3),
          known: predictionCases.length % 3,
        });
      }
      const result = resolveTurn(game.board, game.pairs[turnIndex], analysis.bestAction);
      if (result.topout) game.active = false;
      else game.board = result.finalBoard;
    }
  }

  let predictionCount = 0;
  let maxAbsError = 0;
  let maxToleranceRatio = 0;
  for (let start = 0; start < predictionCases.length; start += 2048) {
    const batch = predictionCases.slice(start, start + 2048);
    const response = await bridge.request({
      type: "predict",
      boards: batch.map((item) => Buffer.from(item.board).toString("base64")),
      nextPairs: batch.map((item) => {
        const result = [
          [0, 0],
          [0, 0],
        ];
        item.nextPairs.slice(0, item.known).forEach((pair, index) => {
          result[index] = pairCodes(pair);
        });
        return result;
      }),
      knownCounts: batch.map((item) => item.known),
    });
    batch.forEach((item, index) => {
      const paddedPairs = [
        ...item.nextPairs.slice(0, item.known),
      ];
      const js = evaluateSoloValueInput(
        model,
        encodeSoloValueInputFast(item.board, paddedPairs),
      );
      for (let outputIndex = 0; outputIndex < 2; outputIndex += 1) {
        const jsValue = outputIndex === 0 ? js.value : js.deathLogit;
        const reference = response.outputs[index][outputIndex];
        const error = Math.abs(jsValue - reference);
        const tolerance = 1e-4 + 1e-4 * Math.abs(reference);
        maxAbsError = Math.max(maxAbsError, error);
        maxToleranceRatio = Math.max(maxToleranceRatio, error / tolerance);
      }
      predictionCount += 1;
    });
  }
  bridge.close();
  const searchActionParity = firstDifference === null ? "passed" : "failed";
  const searchScoreParity =
    maxSearchScoreToleranceRatio <= 1 ? "passed" : "failed";
  const predictionParity = maxToleranceRatio <= 1 ? "passed" : "failed";
  const report = {
    status:
      predictionParity === "passed" &&
      (!args.requireSearchParity ||
        (searchActionParity === "passed" && searchScoreParity === "passed"))
        ? "passed"
        : "failed",
    predictionParity,
    searchActionParity,
    searchScoreParity,
    searchParityRequired: args.requireSearchParity,
    model: path.resolve(args.model),
    bridge: bridge.ready,
    games: games.length,
    decisions,
    firstDifference,
    predictionCount,
    maxAbsError,
    maxToleranceRatio,
    maxSearchScoreAbsError,
    maxSearchScoreToleranceRatio,
  };
  console.log(JSON.stringify(report, null, 2));
  if (report.status !== "passed") process.exitCode = 1;
}

await main();
