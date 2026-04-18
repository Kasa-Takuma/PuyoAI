import { searchBestMove } from "../ai/search.js";

self.addEventListener("message", (event) => {
  const { type, requestId, payload } = event.data ?? {};

  if (type !== "analyze") {
    return;
  }

  try {
    const analysis = searchBestMove(payload);
    self.postMessage({
      type: "analysis-result",
      requestId,
      analysis,
    });
  } catch (error) {
    self.postMessage({
      type: "analysis-error",
      requestId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
