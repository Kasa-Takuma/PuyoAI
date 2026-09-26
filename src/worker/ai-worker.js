import { analyzeLearnedMove } from "../ai/learned.js";
import { searchBestMove } from "../ai/search.js";
import { searchSoloMove } from "../ai/solo-search.js";
import { loadSoloValueModel } from "../ai/solo-value.js";
import { loadSearchValueModel } from "../ai/value.js";

async function analyzeSearchMove(payload) {
  if (!payload?.settings?.useValueModel) {
    return searchBestMove(payload);
  }

  const valueModel = await loadSearchValueModel();
  return searchBestMove({
    ...payload,
    valueModel,
  });
}

async function analyzeSoloMove(payload) {
  const model = await loadSoloValueModel();
  return searchSoloMove({
    ...payload,
    settings: {
      ...payload.settings,
      beamWidth: payload.settings?.soloBeamWidth ?? 22,
    },
    model,
  });
}

self.addEventListener("message", (event) => {
  const { type, requestId, payload } = event.data ?? {};

  if (type !== "analyze") {
    return;
  }

  const work =
    payload?.mode === "learned"
      ? analyzeLearnedMove(payload)
      : payload?.mode === "solo"
        ? analyzeSoloMove(payload)
        : analyzeSearchMove(payload);

  work
    .then((analysis) => {
      self.postMessage({
        type: "analysis-result",
        requestId,
        analysis,
      });
    })
    .catch((error) => {
      self.postMessage({
        type: "analysis-error",
        requestId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
});
