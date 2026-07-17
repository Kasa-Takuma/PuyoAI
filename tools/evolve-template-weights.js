#!/usr/bin/env node
// Evolves template-ai's tunable evaluation weights (src/ai/template-ai.js's
// DEFAULT_TEMPLATE_WEIGHTS, exposed via settings.evalWeights) against a fixed
// default-weight baseline, using the existing battle-harness.js/template-
// solo-stats.js simulators (reused as-is, not duplicated). Every candidate in
// a generation is scored with common random numbers (CRN): the same battle
// seeds and the same solo seeds within a generation, so fitness differences
// reflect the weight change rather than seed luck.
import fs from "node:fs";
import path from "node:path";

import { DEFAULT_TEMPLATE_WEIGHTS } from "../src/ai/template-ai.js";
import { createRng } from "../src/core/randomizer.js";
import { runGames } from "./battle-harness.js";
import { runSoloGames } from "./template-solo-stats.js";

const WEIGHT_KEYS = Object.keys(DEFAULT_TEMPLATE_WEIGHTS);

const DEFAULT_GENERATIONS = 6;
const DEFAULT_POPULATION = 8;
const DEFAULT_BATTLE_GAMES = 20;
const DEFAULT_SOLO_GAMES = 15;
const DEFAULT_SEED = 1000;
const DEFAULT_SIGMA = 0.15;
const DEFAULT_MUTATION_PROBABILITY = 0.5;
const DEFAULT_OUTPUT_PATH = "log/template-evolution.json";

function parseIntArg(text, fallback) {
  const parsed = Number.parseInt(text, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseFloatArg(text, fallback) {
  const parsed = Number.parseFloat(text);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseArgs(argv) {
  const args = {
    generations: DEFAULT_GENERATIONS,
    population: DEFAULT_POPULATION,
    battleGames: DEFAULT_BATTLE_GAMES,
    soloGames: DEFAULT_SOLO_GAMES,
    seed: DEFAULT_SEED,
    sigma: DEFAULT_SIGMA,
    soloWeight: 1,
    out: DEFAULT_OUTPUT_PATH,
    maxRounds: undefined,
    maxMoves: undefined,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--generations") {
      args.generations = Math.max(1, parseIntArg(next, args.generations));
      index += 1;
    } else if (arg === "--population") {
      args.population = Math.max(2, parseIntArg(next, args.population));
      index += 1;
    } else if (arg === "--battle-games") {
      args.battleGames = Math.max(1, parseIntArg(next, args.battleGames));
      index += 1;
    } else if (arg === "--solo-games") {
      args.soloGames = Math.max(1, parseIntArg(next, args.soloGames));
      index += 1;
    } else if (arg === "--seed") {
      args.seed = parseIntArg(next, args.seed);
      index += 1;
    } else if (arg === "--sigma") {
      args.sigma = Math.max(0, parseFloatArg(next, args.sigma));
      index += 1;
    } else if (arg === "--solo-weight") {
      args.soloWeight = Math.max(0, parseFloatArg(next, args.soloWeight));
      index += 1;
    } else if (arg === "--out") {
      args.out = next || args.out;
      index += 1;
    } else if (arg === "--max-rounds") {
      args.maxRounds = Math.max(1, parseIntArg(next, args.maxRounds));
      index += 1;
    } else if (arg === "--max-moves") {
      args.maxMoves = Math.max(1, parseIntArg(next, args.maxMoves));
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
  node tools/evolve-template-weights.js [options]

Options:
  --generations N     Number of evolution generations. Default: ${DEFAULT_GENERATIONS}
  --population N      Candidates per generation (champion + mutants). Default: ${DEFAULT_POPULATION}
  --battle-games N    Battle games (candidate vs default-weight baseline) per candidate. Default: ${DEFAULT_BATTLE_GAMES}
  --solo-games N      Solo games per candidate. Default: ${DEFAULT_SOLO_GAMES}
  --seed N            Base seed. Default: ${DEFAULT_SEED}
  --sigma X           Mutation log-normal sigma. Default: ${DEFAULT_SIGMA}
  --solo-weight X     Fitness weight on solo avg max chain (fitness = winRate*100 + soloAvgMaxChain*X). Default: 1
  --out PATH          JSON report path. Default: ${DEFAULT_OUTPUT_PATH}
  --max-rounds N      Round cap per battle game (passed through to battle-harness). Default: harness default
  --max-moves N       Move cap per solo game (passed through to template-solo-stats). Default: tool default`);
}

function printJson(payload) {
  console.log(JSON.stringify(payload));
}

// Deterministic numeric seed derived from arbitrary string parts, for
// feeding battle-harness/template-solo-stats's own numeric `seed` (they add
// small per-game offsets to it, so it must be a plain integer).
function numericSeedFrom(...parts) {
  return createRng(parts.join(":")).nextUint32();
}

function uniformFloat(rng) {
  return rng.nextUint32() / 0xffffffff;
}

// src/core/randomizer's createRng only exposes nextUint32/nextInt (no
// floats), so a standard Box-Muller transform turns two of its uniforms into
// one N(0,1) sample.
function nextGaussian(rng) {
  let u1 = uniformFloat(rng);
  while (u1 <= Number.EPSILON) {
    u1 = uniformFloat(rng);
  }
  const u2 = uniformFloat(rng);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Mutates each weight independently with probability 0.5 by multiplying its
// magnitude by exp(N(0, sigma)) - always positive, so the sign of every
// weight is preserved regardless of how large the mutation draw is.
function mutateWeights(baseWeights, rng, sigma) {
  const mutated = { ...baseWeights };
  for (const key of WEIGHT_KEYS) {
    if (uniformFloat(rng) < DEFAULT_MUTATION_PROBABILITY) {
      mutated[key] = mutated[key] * Math.exp(nextGaussian(rng) * sigma);
    }
  }
  return mutated;
}

function buildGenerationCandidates({ championWeights, population, seed, generation, sigma }) {
  const candidates = [{ id: "champion", weights: championWeights }];
  for (let index = 1; index < population; index += 1) {
    const rng = createRng(`${seed}:mutate:g${generation}:c${index}`);
    candidates.push({
      id: `g${generation}c${index}`,
      weights: mutateWeights(championWeights, rng, sigma),
    });
  }
  return candidates;
}

function average(values) {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

// Battle: candidate as side A vs the fixed default-weight baseline as side B.
// Solo: candidate alone. Both use the SAME seed base for every candidate in
// the generation (common random numbers), so fitness differences come from
// the weight change, not from which seeds got drawn.
function evaluateCandidate(candidate, { battleSeedBase, soloSeedBase, battleGames, soloGames, maxRounds, maxMoves, soloWeight = 1 }) {
  const battleReport = runGames({
    games: battleGames,
    seed: battleSeedBase,
    aSettings: { evalWeights: candidate.weights },
    bSettings: { evalWeights: {} },
    ...(maxRounds !== undefined ? { maxRounds } : {}),
  });
  const soloReport = runSoloGames({
    games: soloGames,
    seed: soloSeedBase,
    settings: { evalWeights: candidate.weights },
    ...(maxMoves !== undefined ? { maxMoves } : {}),
  });

  const wins = battleReport.aggregate.a.wins;
  const losses = battleReport.aggregate.a.losses;
  const draws = battleReport.aggregate.a.draws;
  const decidedWinRate = wins / Math.max(1, wins + losses);
  const soloAvgMaxChain = average(soloReport.aggregate.maxChains);
  const fitness = decidedWinRate * 100 + soloAvgMaxChain * soloWeight;

  return { fitness, wins, losses, draws, decidedWinRate, soloAvgMaxChain };
}

function writeReport(outputPath, report) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
}

export async function runEvolution(args) {
  const startedAt = performance.now();
  let championWeights = { ...DEFAULT_TEMPLATE_WEIGHTS };
  let championFitness = null;

  const report = {
    kind: "puyoai_template_weight_evolution_report",
    version: 1,
    createdAt: new Date().toISOString(),
    settings: args,
    baselineWeights: DEFAULT_TEMPLATE_WEIGHTS,
    champion: { weights: championWeights, fitness: null },
    generations: [],
  };

  for (let generation = 0; generation < args.generations; generation += 1) {
    const candidates = buildGenerationCandidates({
      championWeights,
      population: args.population,
      seed: args.seed,
      generation,
      sigma: args.sigma,
    });

    const battleSeedBase = numericSeedFrom(args.seed, "battle", generation);
    const soloSeedBase = numericSeedFrom(args.seed, "solo", generation);

    printJson({
      stage: "generation_start",
      generation,
      population: candidates.length,
      battleSeedBase,
      soloSeedBase,
    });

    const evaluated = candidates.map((candidate) => {
      const result = evaluateCandidate(candidate, {
        battleSeedBase,
        soloSeedBase,
        battleGames: args.battleGames,
        soloGames: args.soloGames,
        maxRounds: args.maxRounds,
        maxMoves: args.maxMoves,
        soloWeight: args.soloWeight,
      });
      printJson({ stage: "candidate", generation, id: candidate.id, ...result });
      return { id: candidate.id, weights: candidate.weights, ...result };
    });

    // argmax with ties won by the earliest candidate; the incumbent champion
    // is always candidate index 0, so a tie against it keeps the incumbent.
    let bestIndex = 0;
    for (let index = 1; index < evaluated.length; index += 1) {
      if (evaluated[index].fitness > evaluated[bestIndex].fitness) {
        bestIndex = index;
      }
    }
    const best = evaluated[bestIndex];
    championWeights = best.weights;
    championFitness = best.fitness;

    report.champion = { weights: championWeights, fitness: championFitness, stats: best };
    report.generations.push({
      generation,
      battleSeedBase,
      soloSeedBase,
      candidates: evaluated,
      champion: report.champion,
    });
    report.elapsedMs = performance.now() - startedAt;
    writeReport(args.out, report);

    printJson({
      stage: "generation_complete",
      generation,
      championId: best.id,
      championFitness,
      output: args.out,
    });
  }

  report.completedAt = new Date().toISOString();
  report.elapsedMs = performance.now() - startedAt;
  writeReport(args.out, report);

  return report;
}

function runMain() {
  const args = parseArgs(process.argv.slice(2));

  printJson({
    stage: "start",
    generations: args.generations,
    population: args.population,
    battleGames: args.battleGames,
    soloGames: args.soloGames,
    seed: args.seed,
    sigma: args.sigma,
    output: args.out,
  });

  runEvolution(args).then((report) => {
    const resultJson = {
      kind: "puyoai_template_weight_evolution_result",
      version: 1,
      generations: args.generations,
      championWeights: report.champion.weights,
      championFitness: report.champion.fitness,
      championStats: report.champion.stats,
      baselineWeights: DEFAULT_TEMPLATE_WEIGHTS,
      output: args.out,
      elapsedMs: Math.round(report.elapsedMs),
    };
    console.log(`RESULT_JSON:${JSON.stringify(resultJson)}`);
  }).catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

const isMainModule = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  runMain();
}
