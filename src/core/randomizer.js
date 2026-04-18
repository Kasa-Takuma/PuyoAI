import { PLAYABLE_COLORS } from "./constants.js";

function hashSeed(seed) {
  const text = String(seed);
  let state = 2166136261;

  for (let index = 0; index < text.length; index += 1) {
    state ^= text.charCodeAt(index);
    state = Math.imul(state, 16777619);
  }

  return state >>> 0;
}

export function createRng(seed) {
  let state = hashSeed(seed) || 0x12345678;

  return {
    getSeed() {
      return seed;
    },
    nextUint32() {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      state >>>= 0;
      return state;
    },
    nextInt(max) {
      return this.nextUint32() % max;
    },
  };
}

export function nextPair(rng) {
  return {
    axis: PLAYABLE_COLORS[rng.nextInt(PLAYABLE_COLORS.length)],
    child: PLAYABLE_COLORS[rng.nextInt(PLAYABLE_COLORS.length)],
  };
}

export function fillQueue(rng, queue, minimumLength) {
  while (queue.length < minimumLength) {
    queue.push(nextPair(rng));
  }
}
