// Opening book ported from gata272/puyoAI2's puyoAI.cpp buildLockedPlan.
// Classifies the first three pairs into an abstract A/B/C/D color pattern and
// returns three fixed "chigiri-avoidance" placements when the pattern is
// covered by the table.
import { ORIENTATIONS } from "../core/constants.js";

function abstractColors(p1, p2, p3) {
  let charA = p1.axis;

  if (p1.axis !== p1.child) {
    const p1Colors = [p1.axis, p1.child];
    const p2Colors = [p2.axis, p2.child];
    const intersect = p1Colors.filter((color) => p2Colors.includes(color));
    if (intersect.length === 1) {
      charA = intersect[0];
    } else if (intersect.length === 2) {
      charA = p1.axis;
    }
  }

  const colorMap = new Map();
  colorMap.set(charA, "A");

  const priorityColors = [p1.axis, p1.child, p2.axis, p2.child, p3.axis, p3.child];
  const uniqueColors = [];
  const seen = new Set();
  for (const color of priorityColors) {
    if (!seen.has(color)) {
      uniqueColors.push(color);
      seen.add(color);
    }
  }

  let nextAlpha = "B".charCodeAt(0);
  for (const color of uniqueColors) {
    if (color !== charA) {
      colorMap.set(color, String.fromCharCode(nextAlpha));
      nextAlpha += 1;
    }
  }

  return colorMap;
}

function getPatternLetters(pair, colorMap) {
  return [colorMap.get(pair.axis), colorMap.get(pair.child)].sort().join("");
}

function buildPatternKey(p1, p2, p3, colorMap) {
  return [p1, p2, p3].map((pair) => getPatternLetters(pair, colorMap)).join("-");
}

function detectPatternType(patternKey) {
  if (patternKey.startsWith("AA-AB")) return "AAAB";
  if (patternKey.startsWith("AA-BB")) return "AABB";
  if (patternKey.startsWith("AB-AB")) return "ABAB";
  if (patternKey.startsWith("AB-AC")) return "ABAC";
  if (patternKey.startsWith("AA-BC")) return "AABC";
  if (patternKey.startsWith("AB-CC")) return "ABCC";
  if (patternKey.startsWith("AA-AA")) return "AAAA";
  if (patternKey.startsWith("AB-AA")) return "ABAA";
  return null;
}

function resolveVerticalMove(col1Based, target, pair, colorMap) {
  const column = col1Based - 1;
  if (colorMap.get(pair.axis) === target) {
    return { column, orientation: ORIENTATIONS.UP };
  }
  if (colorMap.get(pair.child) === target) {
    return { column, orientation: ORIENTATIONS.DOWN };
  }
  return { column, orientation: ORIENTATIONS.UP };
}

function resolveHorizontalMove(left1Based, right1Based, target, pair, colorMap) {
  if (target === null) {
    return { column: left1Based - 1, orientation: ORIENTATIONS.RIGHT };
  }
  if (colorMap.get(pair.axis) === target) {
    return { column: left1Based - 1, orientation: ORIENTATIONS.RIGHT };
  }
  if (colorMap.get(pair.child) === target) {
    return { column: right1Based - 1, orientation: ORIENTATIONS.LEFT };
  }
  return { column: left1Based - 1, orientation: ORIENTATIONS.RIGHT };
}

function V(col, target = null) {
  return { kind: "V", col, target };
}

function H(left, right, target = null) {
  return { kind: "H", left, right, target };
}

// Transcribed verbatim from puyoAI.cpp buildLockedPlan (all patternKey
// branches across the AAAB/AABB/ABAB/ABAC/AABC/ABCC/AAAA/ABAA groups).
const PATTERN_TABLE = {
  "AA-AB-AA": [H(1, 2, null), V(3, "B"), H(4, 5, null)],
  "AA-AB-AB": [H(1, 2, null), V(3, "B"), V(4, "A")],
  "AA-AB-AC": [H(1, 2, null), V(3, "B"), V(2, "C")],
  "AA-AB-BB": [H(1, 2, null), V(2, "B"), V(1, "B")],
  "AA-AB-BC": [H(1, 2, null), V(3, "B"), V(4, "C")],
  "AA-AB-CC": [H(1, 2, null), V(3, "B"), H(1, 2, null)],
  "AA-AB-CD": [H(1, 2, null), V(3, "B"), V(6, "D")],

  "AA-BB-AA": [H(1, 2, null), H(1, 2, null), H(4, 5, null)],
  "AA-BB-AB": [H(1, 2, null), H(1, 2, null), H(1, 2, "B")],
  "AA-BB-AC": [H(1, 2, null), H(1, 2, null), V(3, "C")],
  "AA-BB-BB": [H(1, 2, null), H(1, 2, null), H(4, 5, null)],
  "AA-BB-BC": [H(1, 2, null), H(1, 2, null), V(1, "B")],
  "AA-BB-CC": [H(1, 2, null), H(1, 2, null), H(4, 5, null)],
  "AA-BB-CD": [H(1, 2, null), H(1, 2, null), H(5, 6, "C")],

  "AB-AB-AA": [V(1, "A"), V(2, "A"), H(4, 5, null)],
  "AB-AB-AB": [V(1, "A"), V(2, "A"), H(1, 2, "B")],
  "AB-AB-AC": [V(1, "A"), V(2, "A"), V(3, "C")],
  "AB-AB-BB": [V(1, "B"), V(2, "B"), H(4, 5, null)],
  "AB-AB-BC": [V(1, "A"), V(2, "A"), V(1, "B")],
  "AB-AB-CC": [V(1, "A"), V(2, "A"), H(4, 5, null)],
  "AB-AB-CD": [V(1, "A"), V(2, "A"), H(5, 6, null)],

  "AB-AC-AA": [H(2, 3, "A"), V(1, "A"), H(3, 4, null)],
  "AB-AC-AB": [V(1, "A"), H(2, 3, "A"), H(2, 3, "B")],
  "AB-AC-AC": [H(2, 3, "A"), V(1, "A"), H(2, 3, "C")],
  "AB-AC-AD": [V(1, "A"), H(2, 3, "A"), H(3, 4, "A")],
  "AB-AC-BB": [V(1, "A"), H(2, 3, "A"), H(1, 2, null)],
  "AB-AC-BC": [H(2, 3, "A"), V(1, "A"), V(4, "B")],
  "AB-AC-BD": [H(2, 3, "A"), V(1, "A"), V(4, "D")],
  "AB-AC-CC": [V(4, "B"), V(3, "A"), H(1, 2, null)],
  "AB-AC-CD": [V(1, "A"), H(2, 3, "A"), V(4, "C")],

  "AA-BC-AA": [H(1, 2, null), H(2, 3, "B"), H(2, 3, null)],
  "AA-BC-AB": [H(1, 2, null), H(3, 4, "B"), H(5, 6, "B")],
  "AA-BC-AC": [H(1, 2, null), H(3, 4, "C"), H(5, 6, "C")],
  "AA-BC-AD": [H(1, 2, null), H(3, 4, "B"), H(2, 3, "A")],
  "AA-BC-BB": [H(1, 2, null), H(3, 4, "B"), H(5, 6, null)],
  "AA-BC-BC": [H(1, 2, null), H(3, 4, "B"), V(5, "B")],
  "AA-BC-BD": [H(1, 2, null), V(1, "B"), H(2, 3, "B")],
  "AA-BC-CC": [H(1, 2, null), H(2, 3, "C"), V(1, "C")],
  "AA-BC-CD": [H(1, 2, null), H(4, 3, "C"), H(5, 6, "C")],

  "AB-CC-AA": [H(3, 4, "A"), H(1, 2, null), H(5, 6, null)],
  "AB-CC-AB": [V(4, "A"), H(1, 2, null), V(5, "A")],
  "AB-CC-AC": [H(3, 4, "B"), H(1, 2, null), H(3, 4, "A")],
  "AB-CC-AD": [H(3, 4, "A"), H(1, 2, null), H(5, 6, "A")],
  "AB-CC-BB": [H(3, 4, "B"), H(1, 2, null), H(5, 6, null)],
  "AB-CC-BC": [H(3, 4, "A"), H(1, 2, null), H(3, 4, "B")],
  "AB-CC-BD": [H(3, 4, "B"), H(1, 2, null), H(5, 6, "B")],
  "AB-CC-CC": [H(3, 4, "B"), H(1, 2, null), H(5, 6, null)],
  "AB-CC-CD": [H(3, 4, "B"), H(1, 2, null), H(2, 3, "C")],
  "AB-CC-DD": [H(3, 4, "B"), H(1, 2, null), H(1, 2, null)],

  "AA-AA-AA": [V(2, null), V(4, null), V(3, null)],
  "AA-AA-AB": [V(3, null), V(3, null), H(1, 2, "A")],
  "AA-AA-BB": [V(3, null), V(3, null), H(1, 2, null)],
  "AA-AA-BC": [V(3, null), V(3, null), H(1, 2, "B")],

  "AB-AA-AA": [V(3, "B"), H(1, 2, null), H(4, 5, null)],
  "AB-AA-AB": [V(3, "B"), H(1, 2, null), V(4, "A")],
  "AB-AA-AC": [V(3, "B"), H(1, 2, null), V(2, "C")],
  "AB-AA-BB": [V(3, "B"), H(1, 2, null), H(1, 2, null)],
  "AB-AA-BC": [V(3, "B"), H(1, 2, null), V(4, "C")],
  "AB-AA-CC": [V(3, "B"), H(1, 2, null), H(1, 2, null)],
  "AB-AA-CD": [V(3, "B"), H(1, 2, null), V(6, "D")],
};

function isValidPair(pair) {
  return Boolean(pair) && typeof pair.axis === "string" && typeof pair.child === "string";
}

export function buildOpeningPlan(pairs) {
  if (!Array.isArray(pairs) || pairs.length !== 3 || !pairs.every(isValidPair)) {
    return null;
  }

  const [p1, p2, p3] = pairs;
  const colorMap = abstractColors(p1, p2, p3);
  const patternKey = buildPatternKey(p1, p2, p3, colorMap);
  const spec = PATTERN_TABLE[patternKey];
  if (!spec) {
    return null;
  }

  const orderedPairs = [p1, p2, p3];
  const actions = spec.map((descriptor, index) => {
    const pair = orderedPairs[index];
    return descriptor.kind === "V"
      ? resolveVerticalMove(descriptor.col, descriptor.target, pair, colorMap)
      : resolveHorizontalMove(descriptor.left, descriptor.right, descriptor.target, pair, colorMap);
  });

  return {
    patternKey,
    patternType: detectPatternType(patternKey),
    actions,
  };
}
