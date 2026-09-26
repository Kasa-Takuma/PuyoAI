#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { createRng, nextPair } from "../src/core/randomizer.js";
import { PLAYABLE_COLORS } from "../src/core/constants.js";

const COLOR_CODE = Object.freeze(
  Object.fromEntries(PLAYABLE_COLORS.map((color, index) => [color, index + 1])),
);
const DEFAULT_OUTPUT = "training/solo_search/artifacts/seeds";
const SPLITS = Object.freeze({
  train: { count: 64, pairs: 1026 },
  dev: { count: 64, pairs: 1002 },
  test: { count: 128, pairs: 1002 },
  latency: { count: 64, pairs: 1002 },
});

function parseArgs(argv) {
  const args = { output: DEFAULT_OUTPUT };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--out") {
      args.output = argv[index + 1] || args.output;
      index += 1;
    } else if (argv[index] === "--help" || argv[index] === "-h") {
      console.log(`Usage: node tools/prepare-solo-seeds.js [--out DIRECTORY]\n\nDefault output: ${DEFAULT_OUTPUT}`);
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${argv[index]}`);
    }
  }
  return args;
}

function sha256(content) {
  return createHash("sha256").update(content).digest("hex");
}

function generateGame(split, index, pairCount) {
  const seed = `solo-search-20260914/${split}/${index}`;
  const rng = createRng(seed);
  const pairs = Array.from({ length: pairCount }, () => {
    const pair = nextPair(rng);
    return [COLOR_CODE[pair.axis], COLOR_CODE[pair.child]];
  });
  const pairBytes = Uint8Array.from(pairs.flat());
  return { seed, pairCount, pairs, sha256: sha256(pairBytes) };
}

export async function prepareSoloSeeds(outputDirectory) {
  await mkdir(outputDirectory, { recursive: true });
  const files = [];
  const allNames = new Set();
  for (const [split, config] of Object.entries(SPLITS)) {
    const games = Array.from({ length: config.count }, (_, index) =>
      generateGame(split, index, config.pairs),
    );
    for (const game of games) {
      if (allNames.has(game.seed)) throw new Error(`Duplicate seed name: ${game.seed}`);
      allNames.add(game.seed);
    }
    const payload = `${JSON.stringify({
      format: "puyoai-solo-pairs-v1",
      split,
      colorCodes: PLAYABLE_COLORS,
      games,
    })}\n`;
    const filename = `${split}-pairs.json`;
    await writeFile(path.join(outputDirectory, filename), payload);
    files.push({
      split,
      filename,
      games: games.length,
      pairsPerGame: config.pairs,
      sha256: sha256(payload),
    });
  }
  const manifest = {
    format: "puyoai-solo-seed-manifest-v1",
    generator: "src/core/randomizer.js:createRng+nextPair",
    namespace: "solo-search-20260914",
    files,
  };
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  await writeFile(path.join(outputDirectory, "manifest.json"), manifestText);
  return { ...manifest, manifestSha256: sha256(manifestText) };
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  const args = parseArgs(process.argv.slice(2));
  const result = await prepareSoloSeeds(path.resolve(args.output));
  console.log(JSON.stringify(result, null, 2));
}
