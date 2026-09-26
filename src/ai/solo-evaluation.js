import { soloChainUtility } from "./solo-search.js";

export const SOLO_EVALUATION_TURNS = 1000;

function percentile(sorted, probability) {
  if (sorted.length === 0) return null;
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function ratePer1000(count, plannedTurns) {
  return plannedTurns > 0 ? (1000 * count) / plannedTurns : 0;
}

export function summarizeSoloGame(game, plannedTurns = SOLO_EVALUATION_TURNS) {
  const events = [...(game.events ?? [])].sort((left, right) => left.turn - right.turn);
  const largeEvents = events.filter((event) => event.chains >= 10);
  const histogram = {};
  let utility = 0;
  let totalScore = 0;
  let maxChain = 0;
  for (const event of events) {
    histogram[event.chains] = (histogram[event.chains] ?? 0) + 1;
    utility += soloChainUtility(event.chains);
    totalScore += event.score ?? 0;
    maxChain = Math.max(maxChain, event.chains);
  }

  let repeatedSegments = 0;
  for (let segment = 0; segment < 5; segment += 1) {
    const firstTurn = segment * 200 + 1;
    const lastTurn = (segment + 1) * 200;
    const count = largeEvents.filter(
      (event) => event.turn >= firstTurn && event.turn <= lastTurn,
    ).length;
    if (count >= 3) repeatedSegments += 1;
  }

  const eligibleRefires = largeEvents.filter((event) => event.turn <= plannedTurns - 60);
  const successfulRefires = eligibleRefires.filter((event) =>
    largeEvents.some(
      (candidate) =>
        candidate.turn > event.turn && candidate.turn <= event.turn + 60,
    ),
  ).length;

  const boundaries = [0, ...largeEvents.map((event) => event.turn), plannedTurns];
  let longestLargeChainGap = 0;
  for (let index = 1; index < boundaries.length; index += 1) {
    longestLargeChainGap = Math.max(
      longestLargeChainGap,
      boundaries[index] - boundaries[index - 1],
    );
  }

  return {
    seed: game.seed,
    plannedTurns,
    executedTurns: game.executedTurns ?? events.at(-1)?.turn ?? 0,
    topout: Boolean(game.topout),
    utility,
    events,
    histogram,
    totalScore,
    maxChain,
    largeChainTurns: largeEvents.map((event) => event.turn),
    repeatedSegments,
    eligibleRefires: eligibleRefires.length,
    successfulRefires,
    longestLargeChainGap,
    firstLargeChainTurn: largeEvents[0]?.turn ?? null,
  };
}

export function summarizeSoloGames(games, plannedTurns = SOLO_EVALUATION_TURNS) {
  const perGame = games.map((game) => summarizeSoloGame(game, plannedTurns));
  const totalPlannedTurns = perGame.length * plannedTurns;
  const totalUtility = perGame.reduce((sum, game) => sum + game.utility, 0);
  const counts = {};
  for (const threshold of [10, 11, 12, 13, 14]) {
    counts[threshold] = perGame.reduce(
      (sum, game) =>
        sum + game.events.filter((event) => event.chains >= threshold).length,
      0,
    );
  }
  const eligibleRefires = perGame.reduce(
    (sum, game) => sum + game.eligibleRefires,
    0,
  );
  const successfulRefires = perGame.reduce(
    (sum, game) => sum + game.successfulRefires,
    0,
  );
  const reached = perGame
    .map((game) => game.firstLargeChainTurn)
    .filter((turn) => turn !== null);
  const allEvents = perGame.flatMap((game) => game.events);

  return {
    games: perGame.length,
    plannedTurns,
    totalPlannedTurns,
    totalExecutedTurns: perGame.reduce(
      (sum, game) => sum + game.executedTurns,
      0,
    ),
    u1000: ratePer1000(totalUtility, totalPlannedTurns),
    totalUtility,
    firesPer1000: Object.fromEntries(
      Object.entries(counts).map(([threshold, count]) => [
        `${threshold}Plus`,
        ratePer1000(count, totalPlannedTurns),
      ]),
    ),
    fireCounts: Object.fromEntries(
      Object.entries(counts).map(([threshold, count]) => [`${threshold}Plus`, count]),
    ),
    repeatedSegmentRate:
      perGame.length > 0
        ? perGame.reduce((sum, game) => sum + game.repeatedSegments, 0) /
          (perGame.length * 5)
        : 0,
    refireWithin60: {
      eligible: eligibleRefires,
      successful: successfulRefires,
      rate: eligibleRefires > 0 ? successfulRefires / eligibleRefires : null,
    },
    longestLargeChainGap: Math.max(
      0,
      ...perGame.map((game) => game.longestLargeChainGap),
    ),
    deathRate:
      perGame.length > 0
        ? perGame.filter((game) => game.topout).length / perGame.length
        : 0,
    firstLargeChain: {
      reachedGames: reached.length,
      unreachedGames: perGame.length - reached.length,
      meanTurn:
        reached.length > 0
          ? reached.reduce((sum, turn) => sum + turn, 0) / reached.length
          : null,
      medianTurn:
        reached.length > 0 ? percentile([...reached].sort((a, b) => a - b), 0.5) : null,
    },
    diagnostic: {
      smallChainCounts: Object.fromEntries(
        Array.from({ length: 9 }, (_, index) => {
          const chains = index + 1;
          return [
            chains,
            allEvents.filter((event) => event.chains === chains).length,
          ];
        }),
      ),
      totalScore: perGame.reduce((sum, game) => sum + game.totalScore, 0),
      maxChain: Math.max(0, ...perGame.map((game) => game.maxChain)),
    },
    perGame,
  };
}

function xorshift32(state) {
  let next = state >>> 0;
  next ^= next << 13;
  next ^= next >>> 17;
  next ^= next << 5;
  return next >>> 0;
}

export function pairedBootstrap(
  candidateGames,
  baselineGames,
  { iterations = 10_000, seed = 0x51f15e } = {},
) {
  if (candidateGames.length !== baselineGames.length || candidateGames.length === 0) {
    throw new Error("Paired bootstrap requires two non-empty equally-sized game arrays.");
  }
  for (let index = 0; index < candidateGames.length; index += 1) {
    if (candidateGames[index].seed !== baselineGames[index].seed) {
      throw new Error(`Paired bootstrap seed mismatch at index ${index}.`);
    }
  }

  const differences = new Array(iterations);
  const ratios = new Array(iterations);
  let state = seed >>> 0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let candidate = 0;
    let baseline = 0;
    for (let draw = 0; draw < candidateGames.length; draw += 1) {
      state = xorshift32(state);
      const index = state % candidateGames.length;
      candidate += candidateGames[index].utility;
      baseline += baselineGames[index].utility;
    }
    candidate /= candidateGames.length;
    baseline /= baselineGames.length;
    differences[iteration] = candidate - baseline;
    ratios[iteration] = baseline > 0 ? candidate / baseline : null;
  }
  differences.sort((a, b) => a - b);
  const finiteRatios = ratios.filter((value) => value !== null).sort((a, b) => a - b);
  return {
    iterations,
    difference95: [percentile(differences, 0.025), percentile(differences, 0.975)],
    ratio95:
      finiteRatios.length === iterations
        ? [percentile(finiteRatios, 0.025), percentile(finiteRatios, 0.975)]
        : null,
    zeroBaselineReplicates: iterations - finiteRatios.length,
  };
}

export function summarizeLatency(samples) {
  const sorted = samples.map(Number).sort((a, b) => a - b);
  return {
    count: sorted.length,
    medianMs: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    p99Ms: percentile(sorted, 0.99),
    maxMs: sorted.at(-1) ?? null,
  };
}
