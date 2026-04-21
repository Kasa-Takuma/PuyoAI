#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { isMainThread, parentPort, Worker } from "node:worker_threads";

import { getBoardProfileWeights } from "../src/ai/features.js";
import { getTurnResultProfileWeights, searchBestMove } from "../src/ai/search.js";
import { applyAction, createGameState } from "../src/app/state.js";
import { createRng } from "../src/core/randomizer.js";

const DEFAULT_BASE_PROFILE_ID = "chain_builder_v11";
const DEFAULT_OUTPUT_DIR = "log";
const DEFAULT_STAGE_TURNS = Object.freeze([3000, 5000, 10000]);
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
    baseProfile: DEFAULT_BASE_PROFILE_ID,
    generations: 10,
    population: 36,
    depth: 3,
    beamWidth: 24,
    parallelProfiles: 4,
    stage1Turns: DEFAULT_STAGE_TURNS[0],
    stage2Turns: DEFAULT_STAGE_TURNS[1],
    stage3Turns: DEFAULT_STAGE_TURNS[2],
    stage1Keep: 12,
    stage2Keep: 4,
    stage1Games: 2,
    stage2Games: 2,
    stage3Games: 3,
    top: 8,
    hallOfFame: 8,
    minImprovementPct: 0.03,
    minDistance: 0.006,
    seed: "auto",
    output: null,
    resumeReport: null,
    baseProfileExplicit: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--base-profile") {
      args.baseProfile = next || args.baseProfile;
      args.baseProfileExplicit = true;
      index += 1;
    } else if (arg === "--generations") {
      args.generations = Math.max(1, Number.parseInt(next, 10) || args.generations);
      index += 1;
    } else if (arg === "--population") {
      args.population = Math.max(4, Number.parseInt(next, 10) || args.population);
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
    } else if (arg === "--parallel-profiles") {
      args.parallelProfiles = Math.max(
        1,
        Math.min(16, Number.parseInt(next, 10) || args.parallelProfiles),
      );
      index += 1;
    } else if (arg === "--stage1-turns") {
      args.stage1Turns = Math.max(1, Number.parseInt(next, 10) || args.stage1Turns);
      index += 1;
    } else if (arg === "--stage2-turns") {
      args.stage2Turns = Math.max(1, Number.parseInt(next, 10) || args.stage2Turns);
      index += 1;
    } else if (arg === "--stage3-turns") {
      args.stage3Turns = Math.max(1, Number.parseInt(next, 10) || args.stage3Turns);
      index += 1;
    } else if (arg === "--stage1-keep") {
      args.stage1Keep = Math.max(1, Number.parseInt(next, 10) || args.stage1Keep);
      index += 1;
    } else if (arg === "--stage2-keep") {
      args.stage2Keep = Math.max(1, Number.parseInt(next, 10) || args.stage2Keep);
      index += 1;
    } else if (arg === "--stage1-games") {
      args.stage1Games = Math.max(1, Number.parseInt(next, 10) || args.stage1Games);
      index += 1;
    } else if (arg === "--stage2-games") {
      args.stage2Games = Math.max(1, Number.parseInt(next, 10) || args.stage2Games);
      index += 1;
    } else if (arg === "--stage3-games") {
      args.stage3Games = Math.max(1, Number.parseInt(next, 10) || args.stage3Games);
      index += 1;
    } else if (arg === "--top") {
      args.top = Math.max(1, Number.parseInt(next, 10) || args.top);
      index += 1;
    } else if (arg === "--hall-of-fame") {
      args.hallOfFame = Math.max(0, Number.parseInt(next, 10) || args.hallOfFame);
      index += 1;
    } else if (arg === "--min-improvement-pct") {
      args.minImprovementPct = Math.max(
        0,
        Number.parseFloat(next) || args.minImprovementPct,
      );
      index += 1;
    } else if (arg === "--min-distance") {
      args.minDistance = Math.max(0, Number.parseFloat(next) || args.minDistance);
      index += 1;
    } else if (arg === "--seed") {
      args.seed = next || args.seed;
      index += 1;
    } else if (arg === "--output") {
      args.output = next || args.output;
      index += 1;
    } else if (arg === "--resume-report") {
      args.resumeReport = next || args.resumeReport;
      index += 1;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  args.stage1Keep = Math.min(args.stage1Keep, args.population);
  args.stage2Keep = Math.min(args.stage2Keep, args.stage1Keep);
  return args;
}

function printHelp() {
  console.log(`Usage:
  node tools/evolve-search-profile.js [options]

Options:
  --base-profile ID          Starting profile. Default: chain_builder_v11
  --generations N           Number of evolution generations. With --resume-report,
                            this many additional generations are run. Default: 10
  --population N            New mutated candidates per generation. Default: 36
  --stage1-turns N          Turns per candidate in stage 1. Default: 3000
  --stage2-turns N          Turns per candidate in stage 2. Default: 5000
  --stage3-turns N          Turns per candidate in stage 3. Default: 10000
  --stage1-keep N           Candidates kept after stage 1. Default: 12
  --stage2-keep N           Candidates kept after stage 2. Default: 4
  --stage1-games N          Random seed games used in stage 1. Default: 2
  --stage2-games N          Random seed games used in stage 2. Default: 2
  --stage3-games N          Random seed games used in stage 3. Default: 3
  --parallel-profiles N     Candidate evaluations to run in parallel. Default: 4
  --depth N                 Search depth. Default: 3
  --beam-width N            Beam width. Default: 24
  --seed TEXT|auto          Run seed. auto creates a random seed. Default: auto
  --resume-report PATH      Continue from a previous evolution report.
  --output PATH             JSON report path. Default: log/puyoai-evolution-report-<iso>.json`);
}

function printJson(payload) {
  console.log(JSON.stringify(payload));
}

function randomFloat(rng) {
  return rng.nextUint32() / 0xffffffff;
}

function randomHex(rng, bytes = 4) {
  let out = "";
  for (let index = 0; index < bytes; index += 1) {
    out += Math.floor(randomFloat(rng) * 256)
      .toString(16)
      .padStart(2, "0");
  }
  return out;
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

function effectiveTunableValues(profileConfig, baseProfileId) {
  const effectiveBaseProfileId = profileConfig?.baseProfileId ?? baseProfileId;
  const turnWeights = {
    ...getTurnResultProfileWeights(effectiveBaseProfileId),
    ...(profileConfig?.turnWeights ?? {}),
  };
  const boardWeights = {
    ...getBoardProfileWeights(effectiveBaseProfileId),
    ...(profileConfig?.boardWeights ?? {}),
  };
  const bonusScales = {
    largeChain: 1,
    v9b: 1,
    ...(profileConfig?.bonusScales ?? {}),
  };

  return Object.fromEntries(
    TUNABLES.map((tunable) => {
      const base =
        tunable.base ??
        (tunable.group === "turnWeights"
          ? turnWeights[tunable.key]
          : boardWeights[tunable.key]);
      const value =
        tunable.group === "turnWeights"
          ? turnWeights[tunable.key]
          : tunable.group === "boardWeights"
            ? boardWeights[tunable.key]
            : bonusScales[tunable.key];
      return [tunableKey(tunable), typeof value === "number" ? value : base];
    }),
  );
}

function tunableKey(tunable) {
  return `${tunable.group}.${tunable.key}`;
}

function profileVector(profileConfig, baseProfileId) {
  const values = effectiveTunableValues(profileConfig, baseProfileId);
  return TUNABLES.map((tunable) => values[tunableKey(tunable)] ?? 0);
}

function profileHash(profileConfig, baseProfileId) {
  const values = effectiveTunableValues(profileConfig, baseProfileId);
  return JSON.stringify(
    TUNABLES.map((tunable) => {
      const value = values[tunableKey(tunable)] ?? 0;
      return Number(value.toFixed(Math.abs(value) >= 10 ? 0 : 5));
    }),
  );
}

function vectorDistance(left, right) {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) {
    const scale = Math.max(Math.abs(left[index]), Math.abs(right[index]), 1);
    const diff = (left[index] - right[index]) / scale;
    sum += diff * diff;
  }
  return Math.sqrt(sum / Math.max(1, left.length));
}

function createProfileConfig({
  id,
  label,
  baseProfileId,
  sourceProfileConfig,
  spreadMultiplier,
  rng,
}) {
  const sourceValues = effectiveTunableValues(sourceProfileConfig, baseProfileId);
  const profileConfig = {
    id,
    label,
    baseProfileId,
    turnWeights: {},
    boardWeights: {},
    bonusScales: {},
  };

  for (const tunable of TUNABLES) {
    const base = sourceValues[tunableKey(tunable)] ?? tunable.base;
    if (typeof base !== "number" || !Number.isFinite(base)) {
      continue;
    }
    profileConfig[tunable.group][tunable.key] = jitterValue(
      base,
      tunable.spread * spreadMultiplier,
      rng,
    );
  }

  return profileConfig;
}

function candidateFromConfig(profileConfig, parentId, source) {
  return {
    id: profileConfig.id,
    baseProfileId: profileConfig.baseProfileId,
    parentId,
    source,
    profileConfig,
  };
}

function createBaselineCandidate(baseProfileId) {
  return {
    id: baseProfileId,
    baseProfileId,
    parentId: null,
    source: "baseline",
    profileConfig: null,
  };
}

function candidateKey(candidate) {
  return candidate.id;
}

function uniqueCandidates(candidates) {
  const seen = new Set();
  const unique = [];
  for (const candidate of candidates) {
    const key = candidateKey(candidate);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(candidate);
  }
  return unique;
}

function chooseMutationSource({ slot, population, champion, hallOfFame, rng }) {
  const smallLimit = Math.floor(population * 0.5);
  const broadLimit = smallLimit + Math.floor(population * 0.23);
  const eliteLimit = broadLimit + Math.floor(population * 0.17);

  if (slot <= smallLimit) {
    return { source: champion, sourceKind: "champion_small", spreadMultiplier: 0.75 };
  }
  if (slot <= broadLimit) {
    return { source: champion, sourceKind: "champion_broad", spreadMultiplier: 1.25 };
  }
  if (slot <= eliteLimit && hallOfFame.length > 0) {
    const elite = hallOfFame[Math.floor(randomFloat(rng) * hallOfFame.length)];
    return { source: elite, sourceKind: "hall_of_fame", spreadMultiplier: 1.0 };
  }
  return { source: champion, sourceKind: "explore", spreadMultiplier: 1.85 };
}

function createGenerationCandidates({
  args,
  generation,
  rng,
  champion,
  hallOfFame,
  seenHashes,
  seenVectors,
  stagnation,
}) {
  const candidates = [];
  const generationVectors = [];
  const spreadBoost = Math.min(2.3, 1 + Math.max(0, stagnation - 1) * 0.18);
  let attempts = 0;

  while (candidates.length < args.population && attempts < args.population * 250) {
    attempts += 1;
    const slot = candidates.length + 1;
    const mutation = chooseMutationSource({
      slot,
      population: args.population,
      champion,
      hallOfFame,
      rng,
    });
    const id = `evolve_${args.baseProfile}_g${String(generation).padStart(
      3,
      "0",
    )}_c${String(slot).padStart(3, "0")}_${randomHex(rng, 3)}`;
    const profileConfig = createProfileConfig({
      id,
      label: `Evolve ${args.baseProfile} G${generation} C${slot}`,
      baseProfileId: args.baseProfile,
      sourceProfileConfig: mutation.source.profileConfig,
      spreadMultiplier: mutation.spreadMultiplier * spreadBoost,
      rng,
    });
    const hash = profileHash(profileConfig, args.baseProfile);
    if (seenHashes.has(hash)) {
      continue;
    }

    const vector = profileVector(profileConfig, args.baseProfile);
    const tooClose =
      attempts < args.population * 150 &&
      [...seenVectors, ...generationVectors].some(
        (existing) => vectorDistance(vector, existing) < args.minDistance,
      );
    if (tooClose) {
      continue;
    }

    seenHashes.add(hash);
    seenVectors.push(vector);
    generationVectors.push(vector);
    candidates.push(
      candidateFromConfig(
        profileConfig,
        mutation.source.id,
        `${mutation.sourceKind}@x${Number(
          (mutation.spreadMultiplier * spreadBoost).toFixed(3),
        )}`,
      ),
    );
  }

  if (candidates.length < args.population) {
    printJson({
      stage: "generation_warning",
      generation,
      message: "Could not create the requested number of sufficiently distant candidates.",
      requested: args.population,
      created: candidates.length,
      attempts,
    });
  }

  return candidates;
}

function countAtLeast(histogram, threshold) {
  return Object.entries(histogram).reduce((sum, [chains, count]) => {
    return Number(chains) >= threshold ? sum + count : sum;
  }, 0);
}

function scoreSummary(summary) {
  return (
    summary.tenPlusPer10k * 10 +
    summary.elevenPlusPer10k * 48 +
    summary.twelvePlusPer10k * 120 +
    summary.thirteenPlusPer10k * 260 +
    summary.scorePerTurn * 0.04 -
    summary.below7Per10k * 0.22 -
    summary.sevenToNinePer10k * 0.23 -
    summary.topouts * 900 -
    summary.wallMsPerTurn * 0.08
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
    parentId: candidate.parentId,
    source: candidate.source,
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
    wallMsPerTurn:
      stats.totalTurns > 0 ? stats.wallMs / stats.totalTurns : 0,
    searchMsPerTurn:
      stats.totalTurns > 0 ? stats.searchMs / stats.totalTurns : 0,
    chainHistogram: stats.chainHistogram,
  };
  return {
    ...summary,
    objectiveScore: scoreSummary(summary),
  };
}

function runCandidate(candidate, job) {
  const stats = {
    totalTurns: 0,
    totalScore: 0,
    topouts: 0,
    bestChain: 0,
    chainEventsTotal: 0,
    chainHistogram: {},
    wallMs: 0,
    searchMs: 0,
  };
  const startedAt = performance.now();
  const turnsPerGame = Math.ceil(job.turns / job.seeds.length);
  const aiSettings = {
    depth: job.depth,
    beamWidth: job.beamWidth,
    searchProfile: candidate.baseProfileId,
    profileConfig: candidate.profileConfig ?? null,
  };

  for (
    let game = 0;
    game < job.seeds.length && stats.totalTurns < job.turns;
    game += 1
  ) {
    const state = createGameState({
      presetId: "sandbox",
      seed: job.seeds[game],
      aiSettings,
    });
    let gameTurns = 0;

    while (
      !state.gameOver &&
      gameTurns < turnsPerGame &&
      stats.totalTurns < job.turns
    ) {
      const analysis = searchBestMove({
        board: state.board,
        currentPair: state.currentPair,
        nextQueue: state.nextQueue,
        settings: aiSettings,
      });
      const result = applyAction(state, analysis.bestAction, "evolve");
      if (!result) {
        break;
      }

      gameTurns += 1;
      stats.totalTurns += 1;
      stats.totalScore += result.totalScore;
      stats.searchMs += analysis.elapsedMs ?? 0;
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

  stats.wallMs = performance.now() - startedAt;
  return summarizeRun(candidate, stats);
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
    wallMsPerTurn: Number(summary.wallMsPerTurn.toFixed(3)),
    searchMsPerTurn: Number(summary.searchMsPerTurn.toFixed(3)),
  };
}

function sortResults(results) {
  return [...results].sort(
    (left, right) => right.summary.objectiveScore - left.summary.objectiveScore,
  );
}

function makeStageSeeds({ args, rng, generation, stageName, games }) {
  const token = randomHex(rng, 6);
  return Array.from({ length: games }, (_, index) =>
    [
      "evolve",
      args.baseProfile,
      `g${generation}`,
      stageName,
      token,
      `game-${index + 1}`,
    ].join(":"),
  );
}

function compactResult(result) {
  return {
    summary: roundMetrics(result.summary),
    profileConfig: result.candidate.profileConfig,
  };
}

async function runCandidatesInParallel({ candidates, job, parallelProfiles }) {
  if (candidates.length === 0) {
    return [];
  }

  const workerCount = Math.min(parallelProfiles, candidates.length);
  const results = [];
  let nextIndex = 0;

  await Promise.all(
    Array.from({ length: workerCount }, () =>
      new Promise((resolve, reject) => {
        const worker = new Worker(new URL(import.meta.url), {
          workerData: { mode: "evaluate" },
        });
        let stopping = false;

        function sendNext() {
          if (nextIndex >= candidates.length) {
            worker.postMessage({ type: "stop" });
            return;
          }
          const candidate = candidates[nextIndex];
          nextIndex += 1;
          worker.postMessage({
            type: "run",
            candidate,
            job,
          });
        }

        worker.on("message", (message) => {
          if (message.type === "result") {
            results.push({
              candidate: message.candidate,
              summary: message.summary,
            });
            printJson({
              stage: "candidate",
              generation: job.generation,
              tier: job.stageName,
              ...roundMetrics(message.summary),
            });
            sendNext();
          } else if (message.type === "stopped") {
            stopping = true;
            worker.terminate().then(resolve, reject);
          } else if (message.type === "error") {
            reject(new Error(message.error));
          }
        });
        worker.on("error", reject);
        worker.on("exit", (code) => {
          if (!stopping && code !== 0) {
            reject(new Error(`Worker exited with code ${code}`));
          }
        });

        sendNext();
      }),
    ),
  );

  return sortResults(results);
}

function protectedCandidates({ baseline, champion, hallOfFame }) {
  return uniqueCandidates([
    baseline,
    champion,
    ...hallOfFame.slice(0, 3),
  ]);
}

function selectForNextStage(results, keep, protectedIds) {
  const selected = [];
  for (const result of results) {
    if (selected.length < keep || protectedIds.has(result.candidate.id)) {
      selected.push(result.candidate);
    }
  }
  return uniqueCandidates(selected);
}

function isChampionImprovement(candidateSummary, championSummary, args) {
  if (!candidateSummary || !championSummary) {
    return false;
  }
  if (candidateSummary.id === championSummary.id) {
    return false;
  }
  const objectiveRequired =
    championSummary.objectiveScore * (1 + args.minImprovementPct);
  const enoughObjective = candidateSummary.objectiveScore >= objectiveRequired;
  const keepsLargeChains =
    candidateSummary.elevenPlusPer10k >= championSummary.elevenPlusPer10k * 0.95;
  const keepsSafety = candidateSummary.topouts <= championSummary.topouts;
  return enoughObjective && keepsLargeChains && keepsSafety;
}

function updateHallOfFame(hallOfFame, results, limit) {
  const merged = new Map(hallOfFame.map((candidate) => [candidate.id, candidate]));
  for (const result of results) {
    if (result.candidate.profileConfig) {
      merged.set(result.candidate.id, {
        ...result.candidate,
        lastObjectiveScore: result.summary.objectiveScore,
      });
    }
  }
  return [...merged.values()]
    .sort(
      (left, right) =>
        (right.lastObjectiveScore ?? -Infinity) -
        (left.lastObjectiveScore ?? -Infinity),
    )
    .slice(0, limit);
}

function defaultOutputPath() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return path.join(DEFAULT_OUTPUT_DIR, `puyoai-evolution-report-${iso}.json`);
}

function writeReport(outputPath, report) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
}

function createRootSeed(seedArg) {
  return seedArg === "auto" ? crypto.randomUUID() : seedArg;
}

function reportSettings(args, rootSeed) {
  const { baseProfileExplicit, ...settings } = args;
  return { ...settings, seed: rootSeed };
}

function ensureOutputDoesNotOverwriteResume({ resumeReportPath, outputPath }) {
  if (!resumeReportPath) {
    return;
  }
  if (path.resolve(resumeReportPath) === path.resolve(outputPath)) {
    throw new Error(
      "--output must be different from --resume-report so the source report is preserved.",
    );
  }
}

function readResumeReport(reportPath) {
  if (!reportPath) {
    return null;
  }
  const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
  if (report.kind !== "puyoai_search_profile_evolution_report") {
    throw new Error(`Unsupported resume report kind: ${report.kind ?? "unknown"}`);
  }
  if (!report.champion) {
    throw new Error("Resume report does not include a champion.");
  }
  return report;
}

function candidateFromReportEntry(entry, fallbackBaseProfileId) {
  if (!entry) {
    return null;
  }
  const profileConfig = entry.profileConfig ?? null;
  const baseProfileId =
    entry.baseProfileId ??
    profileConfig?.baseProfileId ??
    fallbackBaseProfileId;
  return {
    id: entry.id ?? profileConfig?.id ?? baseProfileId,
    baseProfileId,
    parentId: entry.parentId ?? null,
    source: entry.source ?? "resume",
    profileConfig,
    lastObjectiveScore: entry.lastObjectiveScore,
  };
}

function lastCompletedGeneration(report) {
  return Math.max(
    0,
    ...(report.generations ?? []).map((record) => record.generation ?? 0),
  );
}

function resumedStagnation(report) {
  const generations = report.generations ?? [];
  let count = 0;
  for (let index = generations.length - 1; index >= 0; index -= 1) {
    if (generations[index].championChanged) {
      break;
    }
    count += 1;
  }
  return count;
}

function addProfileToSeen({ profileConfig, baseProfileId, seenHashes, seenVectors }) {
  const hash = profileHash(profileConfig, baseProfileId);
  if (!seenHashes.has(hash)) {
    seenHashes.add(hash);
    seenVectors.push(profileVector(profileConfig, baseProfileId));
  }
}

function restoreSeenProfiles({ report, baseProfileId, seenHashes, seenVectors }) {
  addProfileToSeen({
    profileConfig: null,
    baseProfileId,
    seenHashes,
    seenVectors,
  });

  const addEntry = (entry) => {
    const profileConfig = entry?.profileConfig ?? null;
    if (!profileConfig) {
      return;
    }
    addProfileToSeen({
      profileConfig,
      baseProfileId: profileConfig.baseProfileId ?? baseProfileId,
      seenHashes,
      seenVectors,
    });
  };

  addEntry(report.champion);
  for (const entry of report.hallOfFame ?? []) {
    addEntry(entry);
  }
  for (const generation of report.generations ?? []) {
    for (const entry of generation.generated ?? []) {
      addEntry(entry);
    }
    for (const stage of generation.stages ?? []) {
      for (const result of stage.top ?? []) {
        addEntry(result);
      }
    }
  }
}

async function runMain() {
  const args = parseArgs(process.argv.slice(2));
  const resumeReport = readResumeReport(args.resumeReport);
  if (resumeReport && !args.baseProfileExplicit) {
    args.baseProfile =
      resumeReport.settings?.baseProfile ??
      resumeReport.champion?.baseProfileId ??
      resumeReport.champion?.profileConfig?.baseProfileId ??
      args.baseProfile;
  }
  const rootSeed = createRootSeed(args.seed);
  const rng = createRng(rootSeed);
  const outputPath = args.output ?? defaultOutputPath();
  ensureOutputDoesNotOverwriteResume({
    resumeReportPath: args.resumeReport,
    outputPath,
  });
  const startedAt = performance.now();
  const baseline = createBaselineCandidate(args.baseProfile);
  let champion =
    candidateFromReportEntry(resumeReport?.champion, args.baseProfile) ?? baseline;
  let championStage3Summary =
    resumeReport?.champion?.lastStage3Summary ??
    resumeReport?.generations?.at(-1)?.stage3ChampionSummary ??
    null;
  let stagnation = resumeReport ? resumedStagnation(resumeReport) : 0;
  let hallOfFame = (resumeReport?.hallOfFame ?? [])
    .map((entry) => candidateFromReportEntry(entry, args.baseProfile))
    .filter(Boolean)
    .slice(0, args.hallOfFame);
  const seenHashes = new Set();
  const seenVectors = [];
  if (resumeReport) {
    restoreSeenProfiles({
      report: resumeReport,
      baseProfileId: args.baseProfile,
      seenHashes,
      seenVectors,
    });
  } else {
    addProfileToSeen({
      profileConfig: null,
      baseProfileId: args.baseProfile,
      seenHashes,
      seenVectors,
    });
  }
  const completedGeneration = resumeReport ? lastCompletedGeneration(resumeReport) : 0;
  const startGeneration = completedGeneration + 1;
  const endGeneration = completedGeneration + args.generations;
  const generations = resumeReport ? [...(resumeReport.generations ?? [])] : [];
  const report = {
    kind: "puyoai_search_profile_evolution_report",
    version: 1,
    createdAt: new Date().toISOString(),
    rootSeed,
    objective:
      "evolve search profile weights with random fair seed sets, staged elimination, protected baselines, and strict champion promotion",
    settings: reportSettings(args, rootSeed),
    resumedFrom: resumeReport
      ? {
          path: args.resumeReport,
          rootSeed: resumeReport.rootSeed,
          completedGeneration,
          champion: resumeReport.champion?.id ?? null,
          createdAt: resumeReport.createdAt ?? null,
        }
      : null,
    tunables: TUNABLES,
    champion: resumeReport?.champion ?? null,
    hallOfFame: resumeReport?.hallOfFame ?? [],
    generations,
  };

  printJson({
    stage: "start",
    baseProfileId: args.baseProfile,
    rootSeed,
    generations: args.generations,
    startGeneration,
    endGeneration,
    population: args.population,
    stageTurns: [args.stage1Turns, args.stage2Turns, args.stage3Turns],
    stageKeeps: [args.stage1Keep, args.stage2Keep],
    parallelProfiles: args.parallelProfiles,
    resumeReport: args.resumeReport,
    resumedChampion: resumeReport?.champion?.id ?? null,
    output: outputPath,
  });

  for (let generation = startGeneration; generation <= endGeneration; generation += 1) {
    const generated = createGenerationCandidates({
      args,
      generation,
      rng,
      champion,
      hallOfFame,
      seenHashes,
      seenVectors,
      stagnation,
    });
    const protectedList = protectedCandidates({ baseline, champion, hallOfFame });
    const allStage1Candidates = uniqueCandidates([...protectedList, ...generated]);
    const generationRecord = {
      generation,
      championAtStart: champion.id,
      stagnation,
      generated: generated.map((candidate) => ({
        id: candidate.id,
        parentId: candidate.parentId,
        source: candidate.source,
        profileConfig: candidate.profileConfig,
      })),
      stages: [],
      championChanged: false,
      championAtEnd: null,
    };

    printJson({
      stage: "generation_start",
      generation,
      champion: champion.id,
      stagnation,
      generated: generated.length,
      protected: protectedList.map((candidate) => candidate.id),
    });

    const stageDefinitions = [
      {
        name: "stage1",
        turns: args.stage1Turns,
        games: args.stage1Games,
        keep: args.stage1Keep,
        candidates: allStage1Candidates,
      },
      {
        name: "stage2",
        turns: args.stage2Turns,
        games: args.stage2Games,
        keep: args.stage2Keep,
        candidates: null,
      },
      {
        name: "stage3",
        turns: args.stage3Turns,
        games: args.stage3Games,
        keep: args.stage2Keep,
        candidates: null,
      },
    ];

    let previousResults = null;
    let stage3Results = null;
    for (const stageDefinition of stageDefinitions) {
      const protectedIds = new Set(protectedList.map((candidate) => candidate.id));
      const candidates =
        stageDefinition.candidates ??
        selectForNextStage(
          previousResults,
          stageDefinition.keep,
          protectedIds,
        );
      const seeds = makeStageSeeds({
        args,
        rng,
        generation,
        stageName: stageDefinition.name,
        games: stageDefinition.games,
      });
      const job = {
        generation,
        stageName: stageDefinition.name,
        turns: stageDefinition.turns,
        seeds,
        depth: args.depth,
        beamWidth: args.beamWidth,
      };

      printJson({
        stage: "evaluation_start",
        generation,
        tier: stageDefinition.name,
        candidates: candidates.length,
        turns: stageDefinition.turns,
        games: stageDefinition.games,
        seeds,
      });

      const results = await runCandidatesInParallel({
        candidates,
        job,
        parallelProfiles: args.parallelProfiles,
      });
      generationRecord.stages.push({
        name: stageDefinition.name,
        turns: stageDefinition.turns,
        games: stageDefinition.games,
        seeds,
        candidates: candidates.map((candidate) => candidate.id),
        top: results.slice(0, args.top).map(compactResult),
      });
      previousResults = results;
      if (stageDefinition.name === "stage3") {
        stage3Results = results;
      }

      printJson({
        stage: "evaluation_complete",
        generation,
        tier: stageDefinition.name,
        top: results.slice(0, args.top).map((result, rank) => ({
          rank: rank + 1,
          ...roundMetrics(result.summary),
        })),
      });
    }

    const championResult = stage3Results.find(
      (result) => result.candidate.id === champion.id,
    );
    const bestResult = stage3Results[0];
    const promoted = isChampionImprovement(
      bestResult?.summary,
      championResult?.summary ?? championStage3Summary,
      args,
    );

    if (promoted) {
      champion = bestResult.candidate;
      championStage3Summary = bestResult.summary;
      generationRecord.championChanged = true;
      stagnation = 0;
    } else {
      championStage3Summary = championResult?.summary ?? championStage3Summary;
      stagnation += 1;
    }

    hallOfFame = updateHallOfFame(
      hallOfFame,
      stage3Results,
      args.hallOfFame,
    );

    generationRecord.championAtEnd = champion.id;
    generationRecord.stage3ChampionSummary = championStage3Summary
      ? roundMetrics(championStage3Summary)
      : null;
    generations.push(generationRecord);
    report.champion = {
      id: champion.id,
      baseProfileId: champion.baseProfileId,
      parentId: champion.parentId,
      source: champion.source,
      profileConfig: champion.profileConfig,
      lastStage3Summary: championStage3Summary
        ? roundMetrics(championStage3Summary)
        : null,
    };
    report.hallOfFame = hallOfFame.map((candidate) => ({
      id: candidate.id,
      parentId: candidate.parentId,
      source: candidate.source,
      lastObjectiveScore: candidate.lastObjectiveScore,
      profileConfig: candidate.profileConfig,
    }));
    report.elapsedMs = performance.now() - startedAt;
    writeReport(outputPath, report);

    printJson({
      stage: "generation_complete",
      generation,
      championChanged: generationRecord.championChanged,
      champion: report.champion,
      output: outputPath,
    });
  }

  report.completedAt = new Date().toISOString();
  report.elapsedMs = performance.now() - startedAt;
  writeReport(outputPath, report);
  printJson({
    stage: "complete",
    output: outputPath,
    champion: report.champion,
    elapsedMs: Math.round(report.elapsedMs),
  });
}

if (isMainThread) {
  runMain().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
} else {
  parentPort.on("message", (message) => {
    if (message.type === "stop") {
      parentPort.postMessage({ type: "stopped" });
      return;
    }
    if (message.type !== "run") {
      return;
    }
    try {
      const summary = runCandidate(message.candidate, message.job);
      parentPort.postMessage({
        type: "result",
        candidate: message.candidate,
        summary,
      });
    } catch (error) {
      parentPort.postMessage({
        type: "error",
        error: error instanceof Error ? error.stack : String(error),
      });
    }
  });
}
