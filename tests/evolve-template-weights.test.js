import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { runEvolution } from "../tools/evolve-template-weights.js";

function smokeArgs(outputPath) {
  return {
    generations: 1,
    population: 2,
    battleGames: 2,
    soloGames: 2,
    seed: 9000,
    sigma: 0.2,
    out: outputPath,
    maxRounds: undefined,
    maxMoves: undefined,
  };
}

test("runEvolution completes a minimal smoke config and produces a finite-fitness champion", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "template-evolution-"));
  const outputPath = path.join(dir, "report.json");

  const report = await runEvolution(smokeArgs(outputPath));

  assert.equal(report.generations.length, 1);
  assert.equal(report.generations[0].candidates.length, 2);
  assert.ok(Number.isFinite(report.champion.fitness));
  assert.ok(Number.isFinite(report.champion.stats.decidedWinRate));
  assert.ok(Number.isFinite(report.champion.stats.soloAvgMaxChain));

  assert.ok(fs.existsSync(outputPath));
  const written = JSON.parse(fs.readFileSync(outputPath, "utf8"));
  assert.equal(written.kind, "puyoai_template_weight_evolution_report");
  assert.deepEqual(written.champion.weights, report.champion.weights);
});

test("runEvolution is deterministic across two runs with the same seed", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "template-evolution-"));

  const first = await runEvolution(smokeArgs(path.join(dir, "a.json")));
  const second = await runEvolution(smokeArgs(path.join(dir, "b.json")));

  assert.deepEqual(first.champion.weights, second.champion.weights);
  assert.equal(first.champion.fitness, second.champion.fitness);
  assert.deepEqual(
    first.generations.map((g) => g.candidates.map((c) => c.fitness)),
    second.generations.map((g) => g.candidates.map((c) => c.fitness)),
  );
});
