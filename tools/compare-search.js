#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { isMainThread, parentPort, Worker } from "node:worker_threads";

import { searchBestMove } from "../src/ai/search.js";
import { hydrateModel } from "../src/ai/value.js";
import { applyAction, createGameState } from "../src/app/state.js";

const DEFAULT_OUTPUT_DIR = "log";

const CONFIGS = [
  {
    label: "sampled",
    settings: {
      dedupe: true,
      sampleCount: 4,
      sampleDepth: 4,
      sampleBeamWidth: 6,
      sampleTopK: 8,
      sampleWeight: 1,
    },
  },
  {
    label: "wide_samples",
    settings: {
      dedupe: true,
      sampleCount: 8,
      sampleDepth: 4,
      sampleBeamWidth: 6,
      sampleTopK: 8,
      sampleWeight: 1,
    },
  },
  {
    label: "wide_topk",
    settings: {
      dedupe: true,
      sampleCount: 4,
      sampleDepth: 4,
      sampleBeamWidth: 6,
      sampleTopK: 12,
      sampleWeight: 1,
    },
  },
  {
    label: "beam48",
    settings: {
      dedupe: true,
      beamWidth: 48,
      sampleCount: 4,
      sampleDepth: 4,
      sampleBeamWidth: 6,
      sampleTopK: 8,
      sampleWeight: 1,
    },
  },
  {
    label: "rollout_beam10",
    settings: {
      dedupe: true,
      sampleCount: 4,
      sampleDepth: 4,
      sampleBeamWidth: 10,
      sampleTopK: 8,
      sampleWeight: 1,
    },
  },
  {
    label: "weight_half",
    settings: {
      dedupe: true,
      sampleCount: 4,
      sampleDepth: 4,
      sampleBeamWidth: 6,
      sampleTopK: 8,
      sampleWeight: 0.5,
    },
  },
  {
    label: "refine",
    settings: {
      dedupe: true,
      sampleCount: 4,
      sampleDepth: 4,
      sampleBeamWidth: 6,
      sampleTopK: 8,
      sampleWeight: 1,
      sampleRefineLeaf: true,
    },
  },
  {
    label: "refine_value",
    settings: {
      dedupe: true,
      sampleCount: 4,
      sampleDepth: 4,
      sampleBeamWidth: 6,
      sampleTopK: 8,
      sampleWeight: 1,
      sampleRefineLeaf: true,
      sampleValueWeight: 80000,
    },
  },
  { label: "baseline", settings: { dedupe: true, sampleCount: 0 } },
];

function parseIntArg(text, fallback) {
  const parsed = Number.parseInt(text, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseArgs(argv) {
  const args = {
    turns: 2400,
    games: 2,
    seed: "compare-v1",
    depth: 3,
    beamWidth: 24,
    profile: "chain_builder_v13",
    visibleNexts: 2,
    configLabels: null,
    parallel: 4,
    output: null,
    valueModelPath: null,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--turns") {
      args.turns = Math.max(1, parseIntArg(next, args.turns));
      index += 1;
    } else if (arg === "--games") {
      args.games = Math.max(1, parseIntArg(next, args.games));
      index += 1;
    } else if (arg === "--seed") {
      args.seed = next || args.seed;
      index += 1;
    } else if (arg === "--depth") {
      args.depth = Math.max(1, Math.min(4, parseIntArg(next, args.depth)));
      index += 1;
    } else if (arg === "--beam") {
      args.beamWidth = Math.max(4, Math.min(96, parseIntArg(next, args.beamWidth)));
      index += 1;
    } else if (arg === "--profile") {
      args.profile = next || args.profile;
      index += 1;
    } else if (arg === "--visible-nexts") {
      args.visibleNexts = Math.max(0, parseIntArg(next, args.visibleNexts));
      index += 1;
    } else if (arg === "--configs") {
      args.configLabels = (next ?? "")
        .split(",")
        .map((label) => label.trim())
        .filter(Boolean);
      index += 1;
    } else if (arg === "--parallel") {
      args.parallel = Math.max(1, Math.min(16, parseIntArg(next, args.parallel)));
      index += 1;
    } else if (arg === "--out") {
      args.output = next || args.output;
      index += 1;
    } else if (arg === "--value-model") {
      args.valueModelPath = next || args.valueModelPath;
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
  node tools/compare-search.js [options]

Options:
  --turns N           Total turn budget per config across all games. Default: 2400
  --games N           Number of games per config. Default: 2
  --seed TEXT         Root seed. Games use <seed>:game-<index>. Default: compare-v1
  --depth N           Search depth. Default: 3
  --beam N            Beam width. Default: 24
  --profile ID        Search profile id. Default: chain_builder_v13
  --visible-nexts N   Next queue length passed to the search. Default: 2
  --configs a,b,c     Comma separated config labels to run. Default: all
  --parallel N        Worker count. Default: 4
  --out PATH          JSON report path. Default: log/puyoai-search-compare-<iso>.json
  --value-model PATH  Value model web export JSON, loaded once per worker and
                       passed to searchBestMove. Default: none`);
}

function filterConfigs(labels) {
  if (!labels) {
    return CONFIGS;
  }
  const known = new Set(CONFIGS.map((config) => config.label));
  const unknown = labels.filter((label) => !known.has(label));
  if (unknown.length > 0) {
    console.error(
      `Unknown config label(s): ${unknown.join(", ")}. Available: ${[...known].join(", ")}`,
    );
    process.exit(1);
  }
  return CONFIGS.filter((config) => labels.includes(config.label));
}

function countAtLeast(histogram, threshold) {
  return Object.entries(histogram).reduce((sum, [chains, count]) => {
    return Number(chains) >= threshold ? sum + count : sum;
  }, 0);
}

function countBetween(histogram, min, max) {
  return Object.entries(histogram).reduce((sum, [chains, count]) => {
    const chainCount = Number(chains);
    return chainCount >= min && chainCount <= max ? sum + count : sum;
  }, 0);
}

function createEmptyStats() {
  return {
    gameCount: 0,
    totalTurns: 0,
    totalScore: 0,
    topouts: 0,
    allClears: 0,
    bestChain: 0,
    chainHistogram: {},
    searchMs: 0,
    wallMs: 0,
  };
}

function recordChainResult(stats, result) {
  stats.totalTurns += 1;
  stats.totalScore += result.totalScore;
  stats.bestChain = Math.max(stats.bestChain, result.totalChains);

  if (result.allClear) {
    stats.allClears += 1;
  }

  if (result.totalChains > 0) {
    const key = String(result.totalChains);
    stats.chainHistogram[key] = (stats.chainHistogram[key] ?? 0) + 1;
  }

  if (result.topout) {
    stats.topouts += 1;
  }
}

function buildFullHistogram(rawHistogram, maxChain) {
  const histogram = {};
  for (let chain = 1; chain <= maxChain; chain += 1) {
    histogram[String(chain)] = rawHistogram[String(chain)] ?? 0;
  }
  return histogram;
}

function summarizeRun(config, stats) {
  const per10k = (value) =>
    stats.totalTurns > 0 ? (value / stats.totalTurns) * 10_000 : 0;
  const chains1 = countBetween(stats.chainHistogram, 1, 1);
  const chains2to6 = countBetween(stats.chainHistogram, 2, 6);
  const chains7Plus = countAtLeast(stats.chainHistogram, 7);
  const chains10Plus = countAtLeast(stats.chainHistogram, 10);
  const chains11Plus = countAtLeast(stats.chainHistogram, 11);
  const chains12Plus = countAtLeast(stats.chainHistogram, 12);
  const chains13Plus = countAtLeast(stats.chainHistogram, 13);
  const chains7to9 = chains7Plus - chains10Plus;

  return {
    label: config.label,
    settings: config.settings,
    gameCount: stats.gameCount,
    totalTurns: stats.totalTurns,
    totalScore: stats.totalScore,
    scorePerTurn: stats.totalTurns > 0 ? stats.totalScore / stats.totalTurns : 0,
    topouts: stats.topouts,
    allClears: stats.allClears,
    bestChain: stats.bestChain,
    chainHistogram: buildFullHistogram(stats.chainHistogram, stats.bestChain),
    per10k: {
      chains1: per10k(chains1),
      chains2to6: per10k(chains2to6),
      chains7to9: per10k(chains7to9),
      chains10Plus: per10k(chains10Plus),
      chains11Plus: per10k(chains11Plus),
      chains12Plus: per10k(chains12Plus),
      chains13Plus: per10k(chains13Plus),
    },
    searchMsPerTurn: stats.totalTurns > 0 ? stats.searchMs / stats.totalTurns : 0,
    wallMs: stats.wallMs,
  };
}

let cachedValueModel = null;
let cachedValueModelPath = null;

function loadValueModel(valueModelPath) {
  if (!valueModelPath) {
    return null;
  }
  if (cachedValueModelPath !== valueModelPath) {
    const raw = fs.readFileSync(valueModelPath, "utf8");
    cachedValueModel = hydrateModel(JSON.parse(raw));
    cachedValueModelPath = valueModelPath;
  }
  return cachedValueModel;
}

function runConfig(config, job) {
  const stats = createEmptyStats();
  const startedAt = performance.now();
  const turnsPerGame = Math.ceil(job.turns / job.seeds.length);
  const aiSettings = {
    depth: job.depth,
    beamWidth: job.beamWidth,
    searchProfile: job.profile,
    ...config.settings,
  };
  const valueModel = loadValueModel(job.valueModelPath);

  for (
    let game = 0;
    game < job.seeds.length && stats.totalTurns < job.turns;
    game += 1
  ) {
    const seed = job.seeds[game];
    const state = createGameState({ presetId: "sandbox", seed, aiSettings });
    stats.gameCount += 1;
    let gameTurns = 0;

    while (
      !state.gameOver &&
      gameTurns < turnsPerGame &&
      stats.totalTurns < job.turns
    ) {
      const analysis = searchBestMove({
        board: state.board,
        currentPair: state.currentPair,
        nextQueue: state.nextQueue.slice(0, job.visibleNexts),
        settings: aiSettings,
        turn: state.turn,
        valueModel,
      });
      const result = applyAction(state, analysis.bestAction, "compare");
      if (!result) {
        break;
      }

      gameTurns += 1;
      stats.searchMs += analysis.elapsedMs ?? 0;
      recordChainResult(stats, result);
    }
  }

  stats.wallMs = performance.now() - startedAt;
  return summarizeRun(config, stats);
}

async function runConfigsInParallel({ configs, job, parallel }) {
  if (configs.length === 0) {
    return [];
  }

  const workerCount = Math.min(parallel, configs.length);
  const resultsByLabel = new Map();
  let nextIndex = 0;

  await Promise.all(
    Array.from({ length: workerCount }, () =>
      new Promise((resolve, reject) => {
        const worker = new Worker(new URL(import.meta.url), {
          workerData: { mode: "evaluate" },
        });
        let stopping = false;

        function sendNext() {
          if (nextIndex >= configs.length) {
            worker.postMessage({ type: "stop" });
            return;
          }
          const config = configs[nextIndex];
          nextIndex += 1;
          worker.postMessage({ type: "run", config, job });
        }

        worker.on("message", (message) => {
          if (message.type === "result") {
            resultsByLabel.set(message.summary.label, message.summary);
            console.log(
              `  [${message.summary.label}] done in ${Math.round(message.summary.wallMs)}ms`,
            );
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

  return configs.map((config) => resultsByLabel.get(config.label));
}

function round(value, digits = 2) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : value;
}

function printTable(results) {
  const rows = results.map((summary) => ({
    config: summary.label,
    turns: summary.totalTurns,
    score: Math.round(summary.totalScore),
    scorePerTurn: round(summary.scorePerTurn, 3),
    topouts: summary.topouts,
    allClears: summary.allClears,
    bestChain: summary.bestChain,
    "1/10k": round(summary.per10k.chains1, 1),
    "2-6/10k": round(summary.per10k.chains2to6, 1),
    "7-9/10k": round(summary.per10k.chains7to9, 1),
    "10+/10k": round(summary.per10k.chains10Plus, 1),
    "11+/10k": round(summary.per10k.chains11Plus, 1),
    "12+/10k": round(summary.per10k.chains12Plus, 1),
    "13+/10k": round(summary.per10k.chains13Plus, 1),
    searchMsPerTurn: round(summary.searchMsPerTurn, 2),
    wallMs: Math.round(summary.wallMs),
  }));
  console.table(rows);
}

function defaultOutputPath() {
  const iso = new Date().toISOString().replaceAll(":", "-");
  return path.join(DEFAULT_OUTPUT_DIR, `puyoai-search-compare-${iso}.json`);
}

function writeReport(outputPath, report) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
}

async function runMain() {
  const args = parseArgs(process.argv.slice(2));
  const selectedConfigs = filterConfigs(args.configLabels);
  const seeds = Array.from(
    { length: args.games },
    (_, index) => `${args.seed}:game-${index}`,
  );
  const job = {
    turns: args.turns,
    seeds,
    depth: args.depth,
    beamWidth: args.beamWidth,
    profile: args.profile,
    visibleNexts: args.visibleNexts,
    valueModelPath: args.valueModelPath,
  };
  const outputPath = args.output ?? defaultOutputPath();

  console.log(
    `Running ${selectedConfigs.length} config(s) x ${args.games} game(s) x up to ${args.turns} turns (profile=${args.profile}, depth=${args.depth}, beam=${args.beamWidth}, visibleNexts=${args.visibleNexts})`,
  );
  console.log(`seeds: ${seeds.join(", ")}`);

  const startedAt = performance.now();
  const results = await runConfigsInParallel({
    configs: selectedConfigs,
    job,
    parallel: args.parallel,
  });
  const elapsedMs = performance.now() - startedAt;

  printTable(results);

  const report = {
    kind: "puyoai_search_compare_report",
    version: 1,
    createdAt: new Date().toISOString(),
    settings: {
      turns: args.turns,
      games: args.games,
      seed: args.seed,
      depth: args.depth,
      beamWidth: args.beamWidth,
      profile: args.profile,
      visibleNexts: args.visibleNexts,
      parallel: args.parallel,
      valueModelPath: args.valueModelPath,
    },
    seeds,
    results,
    elapsedMs,
  };
  writeReport(outputPath, report);
  console.log(`Report written to ${outputPath}`);
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
      const summary = runConfig(message.config, message.job);
      parentPort.postMessage({ type: "result", summary });
    } catch (error) {
      parentPort.postMessage({
        type: "error",
        error: error instanceof Error ? error.stack : String(error),
      });
    }
  });
}
