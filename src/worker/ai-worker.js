import { analyzeLearnedMove } from "../ai/learned.js";
import { searchBestMove } from "../ai/search.js";

self.addEventListener("message", (event) => {
  const { type, requestId, payload } = event.data ?? {};

  if (type !== "analyze") {
    return;
  }

  const work =
    payload?.mode === "learned"
      ? analyzeLearnedMove(payload)
      : Promise.resolve(searchBestMove(payload));

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
