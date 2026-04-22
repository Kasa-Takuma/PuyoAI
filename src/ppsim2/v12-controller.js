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

let autoEnabled = false;
let autoTimer = null;
let aiBusy = false;

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

function applyAnalysis(analysis) {
  const placement = convertActionToPpsimPlacement(analysis.bestAction);
  if (!placement) {
    throw new Error("PuyoAI v12 did not return a usable placement.");
  }
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

  rotateToTarget(placement.rotation);
  moveToTargetX(placement.mainX);

  window.hardDrop();
}

function rotateToTarget(targetRotation) {
  let current = getCurrentPuyo();
  if (!current) {
    throw new Error("No active puyo is available for rotation.");
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
      throw new Error(
        `Could not rotate from ${current.rotation} to ${normalizedTarget}.`,
      );
    }

    current = getCurrentPuyo();
    guard += 1;
    if (!current || guard > 4) {
      throw new Error("Rotation did not converge.");
    }
  }
}

function moveToTargetX(targetX) {
  let current = getCurrentPuyo();
  if (!current) {
    throw new Error("No active puyo is available for horizontal movement.");
  }

  let guard = 0;
  while (current.mainX !== targetX) {
    const step = targetX > current.mainX ? 1 : -1;
    if (!window.__aiMoveHorizontal(step)) {
      throw new Error(`Could not move horizontally to column ${targetX + 1}.`);
    }

    current = getCurrentPuyo();
    guard += 1;
    if (!current || guard > 12) {
      throw new Error("Horizontal movement did not converge.");
    }
  }
}

function runPuyoAIInternal() {
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
    applyAnalysis(analysis);

    const bestCandidate =
      analysis.candidates.find(
        (candidate) => candidate.actionKey === analysis.bestActionKey,
      ) ?? analysis.candidates[0];
    const chains = bestCandidate?.immediateChains ?? 0;
    aiStatus(
      `PuyoAI v12 / ${summarizeAction(analysis.bestAction)} / ${chains}連 / ${Math.round(
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

  autoTimer = window.setTimeout(() => {
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
      runPuyoAIInternal();
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
  setStepButtonDisabled(false);
  aiStatus("PuyoAI v12 待機中");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeAiControls);
} else {
  initializeAiControls();
}
