import { SEARCH_PROFILES, searchBestMove } from "../ai/search.js";
import {
  convertActionToPpsimPlacement,
  convertBoard,
  convertCurrentPair,
  convertNextQueue,
  summarizeAction,
} from "./adapter.js";

const PROFILE_STORAGE_KEY = "puyoai.ppsim2.searchProfile";
const DEFAULT_PROFILE_ID = "chain_builder_v12";
const SEARCH_SETTINGS = {
  depth: 3,
  beamWidth: 24,
  searchProfile: DEFAULT_PROFILE_ID,
};
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

function isPpsimSelectableProfile(profile) {
  const match = /^chain_builder_v(\d+)(?:_|$)/.exec(profile.id);
  return match ? Number.parseInt(match[1], 10) >= 12 : false;
}

const PPSIM_PROFILE_OPTIONS = SEARCH_PROFILES.filter(isPpsimSelectableProfile);

function getStoredProfileId() {
  try {
    return window.localStorage?.getItem(PROFILE_STORAGE_KEY) ?? null;
  } catch {
    return null;
  }
}

function storeProfileId(profileId) {
  try {
    window.localStorage?.setItem(PROFILE_STORAGE_KEY, profileId);
  } catch {
    // Storage can be unavailable in private browsing or file contexts.
  }
}

function normalizeProfileId(profileId) {
  return PPSIM_PROFILE_OPTIONS.some((profile) => profile.id === profileId)
    ? profileId
    : DEFAULT_PROFILE_ID;
}

function getActiveProfile() {
  return (
    PPSIM_PROFILE_OPTIONS.find((profile) => profile.id === SEARCH_SETTINGS.searchProfile) ??
    PPSIM_PROFILE_OPTIONS.find((profile) => profile.id === DEFAULT_PROFILE_ID) ??
    PPSIM_PROFILE_OPTIONS[0]
  );
}

function getActiveProfileLabel() {
  return getActiveProfile()?.label?.replace("Chain Builder ", "") ?? "v12";
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
    const label = getActiveProfileLabel();
    button.textContent = on ? `AI自動: ON (${label})` : `AI自動: OFF (${label})`;
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

function updateProfileDescription() {
  const description = document.getElementById("ai-profile-description");
  if (description) {
    description.textContent = getActiveProfile()?.description ?? "";
  }
}

function renderProfileSelect() {
  const select = document.getElementById("ai-profile-select");
  if (!select) {
    return;
  }

  select.innerHTML = "";
  for (const profile of PPSIM_PROFILE_OPTIONS) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.label.replace("Chain Builder ", "");
    select.appendChild(option);
  }
  select.value = SEARCH_SETTINGS.searchProfile;
  updateProfileDescription();
}

function setSearchProfile(profileId) {
  SEARCH_SETTINGS.searchProfile = normalizeProfileId(profileId);
  storeProfileId(SEARCH_SETTINGS.searchProfile);
  const select = document.getElementById("ai-profile-select");
  if (select && select.value !== SEARCH_SETTINGS.searchProfile) {
    select.value = SEARCH_SETTINGS.searchProfile;
  }
  updateProfileDescription();
  setAutoButton(autoEnabled);
  aiStatus(`PuyoAI ${getActiveProfileLabel()} を選択しました`);
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
    settings: { ...SEARCH_SETTINGS },
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
      `PuyoAI ${getActiveProfileLabel()} returned an unusable placement: ${summarizeAction(action)}`,
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
    throw new Error(`PuyoAI ${getActiveProfileLabel()} did not return any usable candidates.`);
  }

  const failures = [];
  for (const entry of candidates) {
    try {
      if (failures.length > 0) {
        aiStatus(`PuyoAI ${getActiveProfileLabel()} 再探索中... ${summarizeAction(entry.action)}`);
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
        `[PuyoAI ${getActiveProfileLabel()}] candidate blocked, trying fallback`,
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
    aiStatus(
      gameState === "gameover"
        ? "ゲームオーバー"
        : `PuyoAI ${getActiveProfileLabel()} 待機中`,
    );
    return false;
  }
  if (!getCurrentPuyo()) {
    aiStatus("操作ぷよ待ち");
    return false;
  }

  const payload = buildSearchPayload();
  if (!payload) {
    aiStatus(`PuyoAI ${getActiveProfileLabel()} 入力待ち`);
    return false;
  }

  aiBusy = true;
  setStepButtonDisabled(true);
  aiStatus(`PuyoAI ${getActiveProfileLabel()} 思考中...`);

  try {
    const analysis = searchBestMove(payload);
    const applied = await applyReachableCandidate(analysis);
    const chains = applied.candidate?.immediateChains ?? 0;
    const fallbackText =
      applied.fallbackCount > 0 ? ` / 再探索${applied.fallbackCount}回` : "";
    aiStatus(
      `PuyoAI ${getActiveProfileLabel()} / ${summarizeAction(applied.action)} / ${chains}連${fallbackText} / ${Math.round(
        analysis.elapsedMs,
      )}ms`,
    );
    return true;
  } catch (error) {
    console.error(`[PuyoAI ${getActiveProfileLabel()}] failed to run`, error);
    aiStatus(
      `PuyoAI ${getActiveProfileLabel()} エラー: ${
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
      aiStatus(`PuyoAI ${getActiveProfileLabel()} 待機中`);
    }

    scheduleAutoTick();
  }, AUTO_INTERVAL_MS);
}

window.PuyoAI = {
  runOnce: runPuyoAIInternal,
  settings: SEARCH_SETTINGS,
  profiles: PPSIM_PROFILE_OPTIONS,
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

window.setPuyoAIProfile = function setPuyoAIProfile(profileId) {
  setSearchProfile(profileId);
};

window.toggleAIAuto = function toggleAIAuto() {
  autoEnabled = !autoEnabled;
  setAutoButton(autoEnabled);

  if (autoEnabled) {
    aiStatus(`PuyoAI ${getActiveProfileLabel()} 自動実行中`);
    scheduleAutoTick();
  } else {
    if (autoTimer !== null) {
      window.clearTimeout(autoTimer);
      autoTimer = null;
    }
    aiStatus(`PuyoAI ${getActiveProfileLabel()} 待機中`);
  }
};

function initializeAiControls() {
  SEARCH_SETTINGS.searchProfile = normalizeProfileId(getStoredProfileId());
  renderProfileSelect();
  setAutoButton(false);
  setMoveLimitButton(false);
  setStepButtonDisabled(false);
  aiStatus(`PuyoAI ${getActiveProfileLabel()} 待機中`);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeAiControls);
} else {
  initializeAiControls();
}
