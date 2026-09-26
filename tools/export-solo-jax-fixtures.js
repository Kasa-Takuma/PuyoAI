#!/usr/bin/env node
import { createGzip } from "node:zlib";
import { createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { once } from "node:events";
import path from "node:path";

import { createEmptyBoard } from "../src/core/board.js";
import { PLAYABLE_COLORS } from "../src/core/constants.js";
import {
  fastColumnHeights,
  fastEnumerateLegalActions,
  fastResolveTurn,
  fromLegacyBoard,
  pairToCodes,
} from "../src/core/fast-board.js";
import { createRng, nextPair } from "../src/core/randomizer.js";

function parseArgs(argv) {
  const args = {
    boards: 100,
    output: "training/solo_search/artifacts/jax-fixtures.jsonl.gz",
    seed: "solo-search-jax-fixtures-v1",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const next = argv[index + 1];
    if (argv[index] === "--boards") {
      args.boards = Math.max(1, Number.parseInt(next, 10) || args.boards);
      index += 1;
    } else if (argv[index] === "--out") {
      args.output = next;
      index += 1;
    } else if (argv[index] === "--seed") {
      args.seed = next || args.seed;
      index += 1;
    } else if (argv[index] === "--help" || argv[index] === "-h") {
      console.log("Usage: node tools/export-solo-jax-fixtures.js [--boards N] [--out FILE] [--seed TEXT]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${argv[index]}`);
    }
  }
  return args;
}

function base64(board) {
  return Buffer.from(board).toString("base64");
}

function collectBoards(count, seed) {
  const rng = createRng(seed);
  const boards = [];
  let board = fromLegacyBoard(createEmptyBoard());
  while (boards.length < count) {
    boards.push(new Uint8Array(board));
    const pair = nextPair(rng);
    const codes = pairToCodes(pair);
    const actions = fastEnumerateLegalActions(board, codes.axis, codes.child);
    const action = actions[rng.nextInt(actions.length)];
    const result = fastResolveTurn(board, codes.axis, codes.child, action);
    board = result.topout ? fromLegacyBoard(createEmptyBoard()) : result.board;
  }
  return boards;
}

function boardFixture(board, index) {
  const cases = [];
  let chainCases = 0;
  for (const axis of PLAYABLE_COLORS) {
    for (const child of PLAYABLE_COLORS) {
      const codes = pairToCodes({ axis, child });
      const actions = fastEnumerateLegalActions(board, codes.axis, codes.child);
      for (let actionId = 0; actionId < 22; actionId += 1) {
        const action = actions.find((candidate) => {
          const orientationOrder = { UP: 0, DOWN: 1, RIGHT: 2, LEFT: 3 };
          if (actionId < 12) {
            return (
              candidate.column === Math.floor(actionId / 2) &&
              orientationOrder[candidate.orientation] === actionId % 2
            );
          }
          if (actionId < 17) {
            return candidate.orientation === "RIGHT" && candidate.column === actionId - 12;
          }
          return candidate.orientation === "LEFT" && candidate.column === actionId - 16;
        });
        if (!action) continue;
        const result = fastResolveTurn(board, codes.axis, codes.child, action);
        if (result.totalChains > 0) chainCases += 1;
        cases.push({
          axis: codes.axis,
          child: codes.child,
          actionId,
          board: base64(result.board),
          topout: result.topout,
          chains: result.totalChains,
          score: result.totalScore,
          allClear: result.allClear,
        });
      }
    }
  }
  return {
    kind: "board",
    index,
    board: base64(board),
    maxHeight: Math.max(...fastColumnHeights(board)),
    chainCases,
    cases,
  };
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, "drain");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const output = path.resolve(args.output);
  await mkdir(path.dirname(output), { recursive: true });
  const gzip = createGzip({ level: 6 });
  const file = createWriteStream(output);
  gzip.pipe(file);
  await writeLine(gzip, {
    kind: "manifest",
    format: "puyoai-solo-jax-fixtures-v1",
    seed: args.seed,
    boards: args.boards,
    expectedCasesPerDifferentPair: 22,
    expectedCasesPerSamePair: 11,
  });
  const boards = collectBoards(args.boards, args.seed);
  let cases = 0;
  let chainBoards = 0;
  for (let index = 0; index < boards.length; index += 1) {
    const fixture = boardFixture(boards[index], index);
    cases += fixture.cases.length;
    if (fixture.chainCases > 0) chainBoards += 1;
    await writeLine(gzip, fixture);
  }
  await writeLine(gzip, { kind: "summary", boards: boards.length, cases, chainBoards });
  gzip.end();
  await once(file, "close");
  console.log(JSON.stringify({ output, boards: boards.length, cases, chainBoards }));
}

await main();
