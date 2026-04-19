#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

import { getBoardProfileWeights } from "../src/ai/features.js";
import { getTurnResultProfileWeights, searchBestMove } from "../src/ai/search.js";
import { applyAction, createGameState } from "../src/app/state.js";
import { createRng } from "../src/core/randomizer.js";

const BASE_PROFILE_ID = "chain_builder_v9b";
const DEFAULT_OUTPUT_DIR = "log";

const TUNABLES = Object.freeze([
  { group: "turnWeights", key: "chainValueBase", spread: 0.08 },
  { group: "turnWeights", key: "chainExponent", spread: 0.025 },
  { group: "turnWeights", key: "scoreScale", spread: 0.06 },
  { group: "turnWeights", key: "smallChainPenaltyStep", spread: 0.16 },
  { group: "turnWeights", key: "midChainPenalty", spread: 0.16 },
  { group: "turnWeights", key: "sevenChainPenalty", spread: 0.18 },
  { group: "turnWeights", key: "eightChainPenalty", spread: 0.18 },
  { group: "turnWeights", key: "nineChainPenalty", spread: 0.18 },
  { group: "turnWeights", key: "tenPlusBonus", spread: 0.16 },
  { group: "turnWeights", key: "elevenPlusBonus", spread: 0.18 },
  { group: "turnWeights", key: "twelvePlusBonus", spread: 0.2 },
  { group: "boardWeights", key: "bestVirtualChain", spread: 0.08 },
  { group: "boardWeights", key: "topVirtualChainSum", spread: 0.1 },
  { group: "boardWeights", key: "bestVirtualScore", spread: 0.1 },
  { group: "boardWeights", key: "topVirtualScoreSum", spread: 0.12 },
  { group: "boardWeights", key: "virtualChainCount3Plus", spread: 0.12 },
  { group: "boardWeights", key: "surfaceReadyGroup3Count", spread: 0.1 },
  { group: "boardWeights", key: "surfaceExtendableGroup2Count", spread: 0.1 },
  { group: "boardWeights", key: "dangerCells", spread: 0.12 },
  { group: "boardWeights", key: "surfaceRoughness", spread: 0.12 },
  { group: "boardWeights", key: "steepWalls", spread: 0.12 },
  { group: "bonusScales", key: "largeChain", base: 1, spread: 0.1 },
  { group: "bonusScales", key: "v9b", base: 1, spread: 0.14 },
]);

function parseArgs(argv) {
  const args = {
    candidates: 12,
    turns: 1200,
    games: 2,
    depth: 3,
    beamWidth: 16,
    seed: "v9b-tune",
    output: null,
    top: 5,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--candidates") {
      args.candidates = Math.max(1, Number.parseInt(next, 10) || args.candidates);
      index += 1;
    } else if (arg === "--turns") {
      args.turns = Math.max(1, Number.parseInt(next, 10) || args.turns);
      index += 1;
    } else if (arg === "--games") {
      args.games = Math.max(1, Number.parseInt(next, 10) || args.games);
      index += 1;
    } else if (arg === "--depth") {
      args.depth = Math.max(1, Math.min(4, Number.parseInt(next, 10) || args.depth));
      index += 1;
    } else if (arg === "--beam-width") {
      args.beamWidth = Math.max(
        4,
        Math.min(96, Number.parseInt(next, 10) || args.beamWidth),
      );
      index += 1;
    } else if (arg === "--seed") {
      args.seed = next || args.seed;
      index += 1;
    } else if (arg === "--output") {
      args.output = next || args.output;
      index += 1;
    } else if (arg === "--top") {
      args.top = Math.max(1, Number.parseInt(next, 10) || args.top);
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
  node tools/tune-search-profile.js [options]

Options:
  --candidates N     Number of mutated v9b candidates to test. Default: 12
  --turns N          Target turns per candidate including baseline. Default: 1200
  --games N          Fixed seed games per candidate. Default: 2
  --depth N          Search depth for the benchmark. Default: 3
  --beam-width N     Beam width for the benchmark. Default: 16
  --seed TEXT        Deterministic tuning seed. Default: v9b-tune
  --output PATH      JSON report path. Default: log/puyoai-tuning-report-<iso>.json
  --top N            Number of top candidates printed at the end. Default: 5`);
}

function randomFloat(rng) {
  return rng.nextUint32() / 0xffffffff;
}

function roundLikeBase(value, base) {
  if (Number.isInteger(base) || Math.abs(base) >= 10) {
    return Math.round(value);
  }
  return Number(value.toFixed(5));
}

function jitterValue(base, spread, rng) {
  const factor = 1 + (randomFloat(rng) * 2 - 1) * spread;
  return roundLikeBase(base * factor, base);
}

function createCandidate(index, rng, baseTurnWeights, baseBoardWeights) {
  const profileConfig = {
    id: `tuned_v9b_${String(index).padStart(3, "0")}`,
    label: `Tuned v9b ${index}`,
    baseProfileId: BASE_PROFILE_ID,
    turnWeights: {},
    boardWeights: {},
    bonusScales: {},
  };

  for (const tunable of TUNABLES) {
    const base =
      tunable.base ??
      (tunable.group === "turnWeights"
        ? baseTurnWeights[tunable.key]
        : baseBoardWeights[tunable.key]);
    if (typeof base !== "number") {
      continue;
    }
    profileConfig[tunable.group][tunable.key] = jitterValue(
      base,
      tunable.spread,
      rng,
    );
  }

  return {
    id: profileConfig.id,
    baseProfileId: BASE_PROFILE_ID,
    profileConfig,
  };
}

function countAtLeast(histogram, threshold) {
  return Object.entries(histogram).reduce((sum, [chains, count]) => {
    return Number(chains) >= threshold ? sum + count : sum;
  }, 0);
}

function scoreSummary(summary) {
  return (
    summary.tenPlusPer10k * 12 +
    summary.elevenPlusPer10k * 42 +
    summary.twelvePlusPer10k * 88 +
    summary.thirteenPlusPer10k * 180 +
    summary.scorePerTurn * 0.04 -
    summary.below7Per10k * 0.18 -
    summary.sevenToNinePer10k * 0.24 -
    summary.topouts * 650
  );
}

function summarizeRun(candidate, stats) {
  const chains7Plus = countAtLeast(stats.chainHistogram, 7);
  const chains10Plus = countAtLeast(stats.chainHistogram, 10);
  const chains11Plus = countAtLeast(stats.chainHistogram, 11);
  const chains12Plus = countAtLeast(stats.chainHistogram, 12);
  const chains13Plus = countAtLeast(stats.chainHistogram, 13);
  const chains7To9 = chains7Plus - chains10Plus;
  const chainsBelow7 = Math.max(0, stats.chainEventsTotal - chains7Plus);
  const per10k = (value) =>
    stats.totalTurns > 0 ? (value / stats.totalTurns) * 10_000 : 0;
  const summary = {
    id: candidate.id,
    baseProfileId: candidate.baseProfileId,
    totalTurns: stats.totalTurns,
    topouts: stats.topouts,
    totalScore: stats.totalScore,
    scorePerTurn:
      stats.totalTurns > 0 ? stats.totalScore / stats.totalTurns : 0,
    bestChain: stats.bestChain,
    chainEventsTotal: stats.chainEventsTotal,
    chainsBelow7,
    chains7Plus,
    chains10Plus,
    chains11Plus,
    chains12Plus,
    chains13Plus,
    below7Per10k: per10k(chainsBelow7),
    sevenToNinePer10k: per10k(chains7To9),
    tenPlusPer10k: per10k(chains10Plus),
    elevenPlusPer10k: per10k(chains11Plus),
    twelvePlusPer10k: per10k(chains12Plus),
    thirteenPlusPer10k: per10k(chains13Plus),
    elevenShareOf10Plus:
      chains10Plus > 0 ? chains11Plus / chains10Plus : 0,
    chainHistogram: stats.chainHistogram,
  };
  return {
    ...summary,
    objectiveScore: scoreSummary(summary),
  };
}

function runCandidate(candidate, args) {
  const stats = {
    totalTurns: 0,
    totalScore: 0,
    topouts: 0,
    bestChain: 0,
    chainEventsTotal: 0,
    chainHistogram: {},
  };
  const turnsPerGame = Math.ceil(args.turns / args.games);
  const aiSettings = {
    depth: args.depth,
    beamWidth: args.beamWidth,
    searchProfile: BASE_PROFILE_ID,
    profileConfig: candidate.profileConfig ?? null,
  };

  for (let game = 0; game < args.games && stats.totalTurns < args.turns; game += 1) {
    const state = createGameState({
      presetId: "sandbox",
      seed: `${args.seed}:game-${game + 1}`,
      aiSettings,
    });
    let gameTurns = 0;

    while (
      !state.gameOver &&
      gameTurns < turnsPerGame &&
      stats.totalTurns < args.turns
    ) {
      const analysis = searchBestMove({
        board: state.board,
        currentPair: state.currentPair,
        nextQueue: state.nextQueue,
        settings: aiSettings,
      });
      const result = applyAction(state, analysis.bestAction, "tune");
      if (!result) {
        break;
      }

      gameTurns += 1;
      stats.totalTurns += 1;
      stats.totalScore += result.totalScore;
      stats.bestChain = Math.max(stats.bestChain, result.totalChains);
      if (result.totalChains > 0) {
        stats.chainEventsTotal += 1;
      }
      if (result.totalChains >= 7) {
        const key = String(result.totalChains);
        stats.chainHistogram[key] = (stats.chainHistogram[key] ?? 0) + 1;
      }
      if (result.topout) {
        stats.topouts += 1;
      }
    }
  }

  return summarizeRun(candidate, stats);
}

function defaultOutputPath() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return path.join(DEFAULT_OUTPUT_DIR, `puyoai-tuning-report-${iso}.json`);
}

function roundMetrics(summary) {
  return {
    ...summary,
    objectiveScore: Number(summary.objectiveScore.toFixed(3)),
    scorePerTurn: Number(summary.scorePerTurn.toFixed(3)),
    below7Per10k: Number(summary.below7Per10k.toFixed(3)),
    sevenToNinePer10k: Number(summary.sevenToNinePer10k.toFixed(3)),
    tenPlusPer10k: Number(summary.tenPlusPer10k.toFixed(3)),
    elevenPlusPer10k: Number(summary.elevenPlusPer10k.toFixed(3)),
    twelvePlusPer10k: Number(summary.twelvePlusPer10k.toFixed(3)),
    thirteenPlusPer10k: Number(summary.thirteenPlusPer10k.toFixed(3)),
    elevenShareOf10Plus: Number((summary.elevenShareOf10Plus * 100).toFixed(2)),
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const rng = createRng(args.seed);
  const baseTurnWeights = getTurnResultProfileWeights(BASE_PROFILE_ID);
  const baseBoardWeights = getBoardProfileWeights(BASE_PROFILE_ID);
  const candidates = [
    {
      id: BASE_PROFILE_ID,
      baseProfileId: BASE_PROFILE_ID,
      profileConfig: null,
    },
  ];

  for (let index = 1; index <= args.candidates; index += 1) {
    candidates.push(createCandidate(index, rng, baseTurnWeights, baseBoardWeights));
  }

  const startedAt = performance.now();
  const results = [];

  console.log(
    JSON.stringify({
      stage: "start",
      baseProfileId: BASE_PROFILE_ID,
      candidates: candidates.length,
      turns: args.turns,
      games: args.games,
      depth: args.depth,
      beamWidth: args.beamWidth,
      seed: args.seed,
    }),
  );

  for (const candidate of candidates) {
    const summary = runCandidate(candidate, args);
    results.push({
      summary,
      profileConfig: candidate.profileConfig,
    });
    console.log(JSON.stringify({ stage: "candidate", ...roundMetrics(summary) }));
  }

  results.sort(
    (left, right) => right.summary.objectiveScore - left.summary.objectiveScore,
  );

  const report = {
    kind: "puyoai_search_profile_tuning_report",
    version: 1,
    createdAt: new Date().toISOString(),
    elapsedMs: performance.now() - startedAt,
    objective:
      "reward high 11+/12+/13+ frequency, lightly reward 10+ and score/turn, penalize <7, 7-9, and topouts",
    settings: args,
    baseProfileId: BASE_PROFILE_ID,
    tunables: TUNABLES,
    results,
  };
  const outputPath = args.output ?? defaultOutputPath();
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);

  console.log(
    JSON.stringify({
      stage: "complete",
      output: outputPath,
      top: results.slice(0, args.top).map((result, rank) => ({
        rank: rank + 1,
        ...roundMetrics(result.summary),
      })),
    }),
  );
}

main();
