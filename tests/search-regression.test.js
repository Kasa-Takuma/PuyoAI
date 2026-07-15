import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { searchBestMove } from "../src/ai/search.js";
import { applyAction, createGameState } from "../src/app/state.js";

const fixturePath = fileURLToPath(
  new URL("./fixtures/search-regression.json", import.meta.url),
);
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));

function round3(value) {
  return Math.round(value * 1000) / 1000;
}

for (const testCase of fixture.cases) {
  test(`search regression fixture: ${testCase.label}`, () => {
    const state = createGameState({ presetId: "sandbox", seed: testCase.seed });

    for (let t = 0; t < testCase.turns.length; t += 1) {
      const expectedTurn = testCase.turns[t];
      const analysis = searchBestMove({
        board: state.board,
        currentPair: state.currentPair,
        nextQueue: state.nextQueue.slice(0, 2),
        settings: testCase.settings,
        turn: t + 1,
      });

      assert.equal(
        analysis.bestActionKey,
        expectedTurn.bestActionKey,
        `bestActionKey mismatch at turn ${t}`,
      );
      assert.equal(
        analysis.candidateCount,
        expectedTurn.candidateCount,
        `candidateCount mismatch at turn ${t}`,
      );
      assert.equal(
        round3(analysis.bestScore),
        expectedTurn.bestScore,
        `bestScore mismatch at turn ${t}`,
      );

      applyAction(state, analysis.bestAction, "fixture");
    }
  });
}
