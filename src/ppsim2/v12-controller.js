import { searchBestMove } from "../ai/search.js";
import {
  convertActionToPpsimPlacement,
  convertBoard,
  convertCurrentPair,
  convertNextQueue,
  summarizeAction,
} from "./adapter.js";

const SEARCH_SETTINGS = Object.freeze({
  depth: 3,
  beamWidth: 24,
  searchProfile: "chain_builder_v12",
});
const AUTO_INTERVAL_MS = 160;
const LIMITED_HORIZONTAL_MOVE_DELAY_MS = 90;

let autoEnabled = false;
let autoTimer = null;
let aiBusy = false;
let horizontalMoveLimited = false;

class MovementError extends Error {
  constructor(message, phase) {
    super(message);
    this.name = "MovementError";
    this.phase = phase;
  }
}

function aiStatus(text) {
  const element = document.getElementById("ai-status");
  if (element) {
    element.textContent = text;
  }
}

function setAutoButton(on) {
  const button = document.getElementById("ai-auto-button");
  if (button) {
    button.textContent = on ? "AI自動: ON (v12)" : "AI自動: OFF (v12)";
  }
}

function setStepButtonDisabled(disabled) {
  const button = document.getElementById("ai-step-button");
  if (button) {
    button.disabled = Boolean(disabled);
  }
}

function setMoveLimitButton(on) {
  const button = document.getElementById("ai-move-limit-button");
  if (button) {
    button.textContent = on ? "横移動制限: ON" : "横移動制限: OFF";
  }
}

function getGameState() {
  return typeof window.getGameState === "function" ? window.getGameState() : null;
}

function getCurrentPuyo() {
  return typeof window.getCurrentPuyoState === "function"
    ? window.getCurrentPuyoState()
    : null;
}

function buildSearchPayload() {
  const ppsimBoard =
    typeof window.getBoardSnapshot === "function" ? window.getBoardSnapshot() : null;
  const currentPair = convertCurrentPair(getCurrentPuyo());
  const nextQueue = convertNextQueue(
    typeof window.getUpcomingPairs === "function" ? window.getUpcomingPairs(5) : [],
  );

  if (!ppsimBoard || !currentPair) {
    return null;
  }

  return {
    board: convertBoard(ppsimBoard),
    currentPair,
    nextQueue,
    settings: SEARCH_SETTINGS,
  };
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function nextFrame() {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });
}

async function waitForLimitedHorizontalMove() {
  await nextFrame();
  await sleep(LIMITED_HORIZONTAL_MOVE_DELAY_MS);
}

function ensurePpsimOperationHooks() {
  if (typeof window.__aiMoveHorizontal !== "function") {
    throw new Error("ppsim2 horizontal movement hook is not available.");
  }
  if (
    typeof window.rotatePuyoCW !== "function" ||
    typeof window.rotatePuyoCCW !== "function"
  ) {
    throw new Error("ppsim2 rotation hooks are not available.");
  }
  if (typeof window.hardDrop !== "function") {
    throw new Error("ppsim2 hardDrop hook is not available.");
  }
}

function getActionKey(action) {
  return action ? `${action.orientation}:${action.column}` : "";
}

function getBestCandidate(analysis) {
  return (
    analysis.candidates.find(
      (candidate) => candidate.actionKey === analysis.bestActionKey,
    ) ??
    analysis.candidates[0] ??
    null
  );
}

function getOrderedCandidates(analysis) {
  const ordered = [];
  const seen = new Set();

  const add = (action, candidate) => {
    if (!action) {
      return;
    }
    const key = getActionKey(action);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    ordered.push({ action, candidate });
  };

  add(analysis.bestAction, getBestCandidate(analysis));
  for (const candidate of analysis.candidates) {
    add(candidate.action, candidate);
  }

  return ordered;
}

async function applyAction(action) {
  const placement = convertActionToPpsimPlacement(action);
  if (!placement) {
    throw new Error(
      `PuyoAI v12 returned an unusable placement: ${summarizeAction(action)}`,
    );
  }

  rotateToTarget(placement.rotation);
  await moveToTargetX(placement.mainX);

  window.hardDrop();
}

async function applyReachableCandidate(analysis) {
  ensurePpsimOperationHooks();

  const candidates = getOrderedCandidates(analysis);
  if (candidates.length === 0) {
    throw new Error("PuyoAI v12 did not return any usable candidates.");
  }

  const failures = [];
  for (const entry of candidates) {
    try {
      if (failures.length > 0) {
        aiStatus(`PuyoAI v12 再探索中... ${summarizeAction(entry.action)}`);
      }
      await applyAction(entry.action);
      return {
        action: entry.action,
        candidate: entry.candidate,
        fallbackCount: failures.length,
      };
    } catch (error) {
      if (!(error instanceof MovementError) || error.phase !== "horizontal") {
        throw error;
      }
      if (getGameState() !== "playing" || !getCurrentPuyo()) {
        throw error;
      }
      console.warn(
        "[PuyoAI v12] candidate blocked, trying fallback",
        summarizeAction(entry.action),
        error,
      );
      failures.push({ action: entry.action, error });
    }
  }

  const lastFailure = failures.at(-1)?.error;
  throw new Error(
    `Reachable placement was not found${
      lastFailure ? `: ${lastFailure.message}` : "."
    }`,
  );
}

function rotateToTarget(targetRotation) {
  let current = getCurrentPuyo();
  if (!current) {
    throw new MovementError("No active puyo is available for rotation.", "state");
  }

  const normalizedTarget = ((targetRotation % 4) + 4) % 4;
  let guard = 0;
  while (current.rotation !== normalizedTarget) {
    const clockwiseDistance = (normalizedTarget - current.rotation + 4) % 4;
    const counterClockwiseDistance = (current.rotation - normalizedTarget + 4) % 4;
    const didRotate =
      clockwiseDistance <= counterClockwiseDistance
        ? window.rotatePuyoCW()
        : window.rotatePuyoCCW();

    if (!didRotate) {
      throw new MovementError(
        `Could not rotate from ${current.rotation} to ${normalizedTarget}.`,
        "rotation",
      );
    }

    current = getCurrentPuyo();
    guard += 1;
    if (!current || guard > 4) {
      throw new MovementError("Rotation did not converge.", "rotation");
    }
  }
}

async function moveToTargetX(targetX) {
  let current = getCurrentPuyo();
  if (!current) {
    throw new MovementError(
      "No active puyo is available for horizontal movement.",
      "state",
    );
  }

  let guard = 0;
  while (current.mainX !== targetX) {
    const step = targetX > current.mainX ? 1 : -1;
    if (!window.__aiMoveHorizontal(step)) {
      throw new MovementError(
        `Could not move horizontally to column ${targetX + 1}.`,
        "horizontal",
      );
    }

    current = getCurrentPuyo();
    guard += 1;
    if (!current || guard > 12) {
      throw new MovementError(
        "Horizontal movement did not converge.",
        "horizontal",
      );
    }

    if (horizontalMoveLimited) {
      await waitForLimitedHorizontalMove();
    }
  }
}

async function runPuyoAIInternal() {
  if (aiBusy) {
    return false;
  }

  const gameState = getGameState();
  if (gameState !== "playing") {
    aiStatus(gameState === "gameover" ? "ゲームオーバー" : "PuyoAI v12 待機中");
    return false;
  }
  if (!getCurrentPuyo()) {
    aiStatus("操作ぷよ待ち");
    return false;
  }

  const payload = buildSearchPayload();
  if (!payload) {
    aiStatus("PuyoAI v12 入力待ち");
    return false;
  }

  aiBusy = true;
  setStepButtonDisabled(true);
  aiStatus("PuyoAI v12 思考中...");

  try {
    const analysis = searchBestMove(payload);
    const applied = await applyReachableCandidate(analysis);
    const chains = applied.candidate?.immediateChains ?? 0;
    const fallbackText =
      applied.fallbackCount > 0 ? ` / 再探索${applied.fallbackCount}回` : "";
    aiStatus(
      `PuyoAI v12 / ${summarizeAction(applied.action)} / ${chains}連${fallbackText} / ${Math.round(
        analysis.elapsedMs,
      )}ms`,
    );
    return true;
  } catch (error) {
    console.error("[PuyoAI v12] failed to run", error);
    aiStatus(
      `PuyoAI v12 エラー: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    autoEnabled = false;
    setAutoButton(false);
    return false;
  } finally {
    aiBusy = false;
    setStepButtonDisabled(false);
  }
}

function scheduleAutoTick() {
  if (!autoEnabled) {
    return;
  }

  if (autoTimer !== null) {
    window.clearTimeout(autoTimer);
    autoTimer = null;
  }

  autoTimer = window.setTimeout(async () => {
    autoTimer = null;
    if (!autoEnabled) {
      return;
    }

    const gameState = getGameState();
    if (gameState === "gameover") {
      autoEnabled = false;
      setAutoButton(false);
      aiStatus("ゲームオーバー");
      return;
    }

    if (!aiBusy && gameState === "playing" && getCurrentPuyo()) {
      await runPuyoAIInternal();
    } else if (gameState !== "playing") {
      aiStatus("PuyoAI v12 待機中");
    }

    scheduleAutoTick();
  }, AUTO_INTERVAL_MS);
}

window.PuyoAI = {
  runOnce: runPuyoAIInternal,
  settings: SEARCH_SETTINGS,
};

window.runPuyoAI = function runPuyoAI() {
  return runPuyoAIInternal();
};

window.toggleAIMoveLimit = function toggleAIMoveLimit() {
  horizontalMoveLimited = !horizontalMoveLimited;
  setMoveLimitButton(horizontalMoveLimited);
  aiStatus(
    horizontalMoveLimited
      ? "横移動制限 ON: 1マスずつ描画します"
      : "横移動制限 OFF: 最速で横移動します",
  );
};

window.toggleAIAuto = function toggleAIAuto() {
  autoEnabled = !autoEnabled;
  setAutoButton(autoEnabled);

  if (autoEnabled) {
    aiStatus("PuyoAI v12 自動実行中");
    scheduleAutoTick();
  } else {
    if (autoTimer !== null) {
      window.clearTimeout(autoTimer);
      autoTimer = null;
    }
    aiStatus("PuyoAI v12 待機中");
  }
};

function initializeAiControls() {
  setAutoButton(false);
  setMoveLimitButton(false);
  setStepButtonDisabled(false);
  aiStatus("PuyoAI v12 待機中");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeAiControls);
} else {
  initializeAiControls();
}
