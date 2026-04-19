#!/usr/bin/env node
import { once } from "node:events";

import { searchBestMove } from "../src/ai/search.js";
import {
  createAiSnapshot,
  createValueTrainingSample,
} from "../src/ai/dataset.js";
import { extractBoardFeatures } from "../src/ai/features.js";
import { applyAction, createGameState } from "../src/app/state.js";

const DEFAULT_HORIZONS = Object.freeze([12, 24, 48]);
const BENCHMARK_FEATURE_KEYS = Object.freeze([
  "stackCells",
  "maxHeight",
  "hiddenCells",
  "dangerCells",
  "surfaceRoughness",
  "steepWalls",
  "valleyPenalty",
  "adjacency",
  "group2Count",
  "group3Count",
  "surfaceExtendableGroup2Count",
  "surfaceReadyGroup3Count",
  "isolatedSingles",
  "colorBalance",
  "columnsUsed",
  "bestVirtualChain",
  "bestVirtualScore",
  "virtualChainCount2Plus",
  "virtualChainCount3Plus",
  "topVirtualChainSum",
  "topVirtualScoreSum",
]);

function parseInteger(value, fallback, min = 1, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function parseHorizons(value) {
  if (!value) {
    return [...DEFAULT_HORIZONS];
  }
  const horizons = value
    .split(",")
    .map((entry) => Number.parseInt(entry.trim(), 10))
    .filter((entry) => Number.isInteger(entry) && entry > 0);
  return horizons.length > 0 ? [...new Set(horizons)].sort((a, b) => a - b) : [...DEFAULT_HORIZONS];
}

function parseArgs(argv) {
  const args = {
    turns: 100_000,
    games: 0,
    depth: 3,
    beamWidth: 16,
    seed: "value-stream",
    searchProfile: "chain_builder_v11",
    presetId: "sandbox",
    horizons: [...DEFAULT_HORIZONS],
    reportEvery: 10_000,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--turns") {
      args.turns = parseInteger(next, args.turns);
      index += 1;
    } else if (arg === "--games") {
      args.games = parseInteger(next, args.games, 0);
      index += 1;
    } else if (arg === "--depth") {
      args.depth = parseInteger(next, args.depth, 1, 4);
      index += 1;
    } else if (arg === "--beam-width") {
      args.beamWidth = parseInteger(next, args.beamWidth, 4, 96);
      index += 1;
    } else if (arg === "--seed") {
      args.seed = next || args.seed;
      index += 1;
    } else if (arg === "--search-profile") {
      args.searchProfile = next || args.searchProfile;
      index += 1;
    } else if (arg === "--preset") {
      args.presetId = next || args.presetId;
      index += 1;
    } else if (arg === "--horizons") {
      args.horizons = parseHorizons(next);
      index += 1;
    } else if (arg === "--report-every") {
      args.reportEvery = parseInteger(next, args.reportEvery);
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
  node tools/generate-value-stream.js [options] > value.jsonl

Options:
  --turns N            Total searched turns to generate. Default: 100000
  --games N            Split turns across N games. 0 means keep starting games until turns are reached. Default: 0
  --depth N            Search depth. Default: 3
  --beam-width N       Search beam width. Default: 16
  --search-profile ID  Search profile used as the teacher. Default: chain_builder_v11
  --seed TEXT          Deterministic seed base. Default: value-stream
  --horizons LIST      Future label horizons, for example 12,24,48. Default: 12,24,48
  --report-every N     Progress interval in generated turns. Default: 10000`);
}

function pickBenchmarkFeatures(features) {
  return Object.fromEntries(
    BENCHMARK_FEATURE_KEYS.map((key) => [key, features[key] ?? 0]),
  );
}

function createEmptyFutureLabel() {
  return {
    complete: false,
    stepsObserved: 0,
    totalScore: 0,
    maxChains: 0,
    chainEvents: 0,
    chains7Plus: 0,
    chains10Plus: 0,
    chains11Plus: 0,
    chains12Plus: 0,
    chains13Plus: 0,
    topout: false,
    topoutAt: null,
  };
}

function createFutureLabels(horizons) {
  return Object.fromEntries(horizons.map((horizon) => [horizon, createEmptyFutureLabel()]));
}

function updateFutureLabel(label, result, stepOffset) {
  label.stepsObserved = stepOffset;
  label.totalScore += result.totalScore;
  label.maxChains = Math.max(label.maxChains, result.totalChains);
  if (result.totalChains > 0) {
    label.chainEvents += 1;
  }
  if (result.totalChains >= 7) {
    label.chains7Plus += 1;
  }
  if (result.totalChains >= 10) {
    label.chains10Plus += 1;
  }
  if (result.totalChains >= 11) {
    label.chains11Plus += 1;
  }
  if (result.totalChains >= 12) {
    label.chains12Plus += 1;
  }
  if (result.totalChains >= 13) {
    label.chains13Plus += 1;
  }
  if (result.topout && !label.topout) {
    label.topout = true;
    label.topoutAt = stepOffset;
  }
}

function finalizeValueEntry(entry, horizons, terminal = false) {
  const future = Object.fromEntries(
    horizons.map((horizon) => {
      const label = { ...entry.future[horizon] };
      label.complete = terminal || label.stepsObserved >= horizon;
      return [horizon, label];
    }),
  );

  return createValueTrainingSample({
    snapshot: entry.snapshot,
    analysis: entry.analysis,
    workerId: 0,
    gameSeed: entry.gameSeed,
    features: entry.features,
    immediate: entry.immediate,
    future,
  });
}

function observeValueFuture(pendingEntries, horizons, maxHorizon, result, terminal = false) {
  const completed = [];

  for (const entry of pendingEntries) {
    entry.stepsObserved += 1;
    for (const horizon of horizons) {
      if (entry.stepsObserved <= horizon) {
        updateFutureLabel(entry.future[horizon], result, entry.stepsObserved);
      }
    }
  }

  while (
    pendingEntries.length > 0 &&
    (terminal || pendingEntries[0].stepsObserved >= maxHorizon)
  ) {
    completed.push(finalizeValueEntry(pendingEntries.shift(), horizons, terminal));
  }

  return completed;
}

async function writeJsonLine(payload) {
  const line = `${JSON.stringify(payload)}\n`;
  if (!process.stdout.write(line)) {
    await once(process.stdout, "drain");
  }
}

function reportProgress(payload) {
  process.stderr.write(`${JSON.stringify(payload)}\n`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const horizons = args.horizons;
  const maxHorizon = Math.max(...horizons);
  const aiSettings = {
    depth: args.depth,
    beamWidth: args.beamWidth,
    searchProfile: args.searchProfile,
  };
  const turnsPerGame =
    args.games > 0 ? Math.ceil(args.turns / Math.max(args.games, 1)) : Number.MAX_SAFE_INTEGER;
  const startedAt = performance.now();
  let totalTurns = 0;
  let emittedSamples = 0;
  let droppedPending = 0;
  let topouts = 0;
  let bestChain = 0;
  let nextReportTurn = args.reportEvery;
  let gameIndex = 0;

  reportProgress({
    stage: "start",
    turns: args.turns,
    games: args.games,
    depth: args.depth,
    beamWidth: args.beamWidth,
    searchProfile: args.searchProfile,
    horizons,
    seed: args.seed,
  });

  while (totalTurns < args.turns && (args.games === 0 || gameIndex < args.games)) {
    gameIndex += 1;
    const gameSeed = `${args.seed}:game-${gameIndex}`;
    const state = createGameState({
      presetId: args.presetId,
      seed: gameSeed,
      aiSettings,
    });
    const pendingValueEntries = [];
    let gameTurns = 0;

    while (!state.gameOver && totalTurns < args.turns && gameTurns < turnsPerGame) {
      const snapshot = createAiSnapshot(state);
      const analysis = searchBestMove({
        board: state.board,
        currentPair: state.currentPair,
        nextQueue: state.nextQueue,
        settings: state.aiSettings,
      });
      const preActionFeatures = extractBoardFeatures(state.board, {
        includeVirtualChains: true,
      });
      const result = applyAction(state, analysis.bestAction, "value-stream");
      if (!result) {
        break;
      }

      totalTurns += 1;
      gameTurns += 1;
      bestChain = Math.max(bestChain, result.totalChains);

      pendingValueEntries.push({
        gameSeed,
        snapshot,
        analysis,
        features: pickBenchmarkFeatures(preActionFeatures),
        immediate: {
          chains: result.totalChains,
          score: result.totalScore,
          topout: result.topout,
          allClear: result.allClear,
          actionKey: analysis.bestActionKey,
        },
        future: createFutureLabels(horizons),
        stepsObserved: 0,
      });

      const completedSamples = observeValueFuture(
        pendingValueEntries,
        horizons,
        maxHorizon,
        result,
        state.gameOver,
      );
      for (const sample of completedSamples) {
        await writeJsonLine(sample);
        emittedSamples += 1;
      }

      if (state.gameOver) {
        topouts += 1;
      }

      if (totalTurns >= nextReportTurn) {
        reportProgress({
          stage: "progress",
          totalTurns,
          emittedSamples,
          game: gameIndex,
          gameTurns,
          topouts,
          bestChain,
          elapsedMs: Math.round(performance.now() - startedAt),
        });
        nextReportTurn += args.reportEvery;
      }
    }

    droppedPending += pendingValueEntries.length;
  }

  reportProgress({
    stage: "complete",
    totalTurns,
    emittedSamples,
    droppedPending,
    gamesStarted: gameIndex,
    topouts,
    bestChain,
    elapsedMs: Math.round(performance.now() - startedAt),
  });
}

main().catch((error) => {
  reportProgress({
    stage: "error",
    message: error instanceof Error ? error.message : String(error),
  });
  process.exitCode = 1;
});
