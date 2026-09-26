#!/usr/bin/env node
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(new URL("..", import.meta.url).pathname);
const output = path.resolve(
  process.argv[2] ?? "training/solo_search/artifacts/v12ac-evolution-kaggle",
);
const datasetOutput = path.join(output, "dataset");
const kernelOutput = path.join(output, "kernel");
await rm(output, { recursive: true, force: true });
await mkdir(path.join(datasetOutput, "solo_search"), { recursive: true });
await mkdir(kernelOutput, { recursive: true });

for (const filename of [
  "__init__.py",
  "jax_board.py",
  "jax_search.py",
  "jax_structure.py",
  "model.py",
  "jax_profile_search.py",
  "evolve_profiles.py",
  "verify_profile_parity.py",
]) {
  await cp(
    path.join(root, "training/solo_search", filename),
    path.join(datasetOutput, "solo_search", filename),
  );
}
await cp(
  path.join(root, "training/solo_search/kaggle/evolution-dataset-metadata.json"),
  path.join(datasetOutput, "dataset-metadata.json"),
);
await cp(
  path.join(root, "training/solo_search/kaggle/evolve-kernel.py"),
  path.join(kernelOutput, "evolve-kernel.py"),
);
await cp(
  path.join(root, "training/solo_search/kaggle/evolution-kernel-metadata.json"),
  path.join(kernelOutput, "kernel-metadata.json"),
);

const resumeInput = process.env.V12AC_RESUME_REPORT;
let resumeOutput = null;
if (resumeInput) {
  resumeOutput = path.join(output, "resume-dataset");
  await mkdir(resumeOutput, { recursive: true });
  await cp(path.resolve(resumeInput), path.join(resumeOutput, "report.json"));
  const metadata = {
    title: "PuyoAI v12AC Evolution Resume",
    id: "ikakun624/puyoai-v12ac-evolution-resume",
    licenses: [{ name: "other" }],
  };
  await writeFile(
    path.join(resumeOutput, "dataset-metadata.json"),
    `${JSON.stringify(metadata, null, 2)}\n`,
  );
  const kernelMetadataPath = path.join(kernelOutput, "kernel-metadata.json");
  const kernelMetadata = JSON.parse(await readFile(kernelMetadataPath, "utf8"));
  kernelMetadata.dataset_sources.push(metadata.id);
  await writeFile(kernelMetadataPath, `${JSON.stringify(kernelMetadata, null, 2)}\n`);
}

console.log(JSON.stringify({ output, datasetOutput, kernelOutput, resumeOutput }));
