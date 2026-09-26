import test from "node:test";
import assert from "node:assert/strict";

import {
  pairedBootstrap,
  summarizeLatency,
  summarizeSoloGames,
} from "../src/ai/solo-evaluation.js";

test("solo evaluation keeps dead games in all planned-turn denominators", () => {
  const summary = summarizeSoloGames([
    {
      seed: "dead",
      executedTurns: 100,
      topout: true,
      events: [
        { turn: 20, chains: 10, score: 1 },
        { turn: 40, chains: 12, score: 2 },
      ],
    },
    {
      seed: "alive",
      executedTurns: 1000,
      topout: false,
      events: [{ turn: 900, chains: 14, score: 3 }],
    },
  ]);

  assert.equal(summary.totalPlannedTurns, 2000);
  assert.equal(summary.totalExecutedTurns, 1100);
  assert.equal(summary.totalUtility, 4.5);
  assert.equal(summary.u1000, 2.25);
  assert.equal(summary.firesPer1000[10 + "Plus"], 1.5);
  assert.equal(summary.deathRate, 0.5);
  assert.equal(summary.longestLargeChainGap, 960);
});

test("solo evaluation measures repeated segments and 60-turn refires", () => {
  const summary = summarizeSoloGames([
    {
      seed: "repeat",
      executedTurns: 1000,
      topout: false,
      events: [
        { turn: 10, chains: 10 },
        { turn: 50, chains: 11 },
        { turn: 100, chains: 12 },
        { turn: 950, chains: 10 },
      ],
    },
  ]);

  assert.equal(summary.repeatedSegmentRate, 0.2);
  assert.deepEqual(summary.refireWithin60, {
    eligible: 3,
    successful: 2,
    rate: 2 / 3,
  });
  assert.equal(summary.firstLargeChain.meanTurn, 10);
});

test("paired bootstrap is paired by seed and deterministic", () => {
  const candidate = [
    { seed: "a", utility: 2 },
    { seed: "b", utility: 4 },
  ];
  const baseline = [
    { seed: "a", utility: 1 },
    { seed: "b", utility: 2 },
  ];
  const first = pairedBootstrap(candidate, baseline, { iterations: 1000, seed: 7 });
  const second = pairedBootstrap(candidate, baseline, { iterations: 1000, seed: 7 });
  assert.deepEqual(first, second);
  assert.deepEqual(first.ratio95, [2, 2]);
  assert.ok(first.difference95[0] > 0);
  assert.throws(
    () => pairedBootstrap(candidate, [...baseline].reverse()),
    /seed mismatch/,
  );
});

test("latency summary includes upper tail and maximum", () => {
  const summary = summarizeLatency(Array.from({ length: 100 }, (_, index) => index + 1));
  assert.equal(summary.count, 100);
  assert.equal(summary.medianMs, 50.5);
  assert.equal(summary.maxMs, 100);
  assert.ok(summary.p95Ms > 95 && summary.p95Ms < 96);
});
