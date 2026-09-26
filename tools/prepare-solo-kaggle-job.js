#!/usr/bin/env node
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(new URL("..", import.meta.url).pathname);
const output = path.resolve(
  process.argv[2] ?? "training/solo_search/artifacts/kaggle-smoke",
);
const kernelOutput = path.join(output, "kernel");
const datasetOutput = path.join(output, "dataset");
const resumeDatasetOutput = path.join(output, "resume-dataset");
const trainingKernelOutput = path.join(output, "training-kernel");
const trainingDatasetOutput = path.join(output, "training-dataset");
await rm(output, { recursive: true, force: true });
await mkdir(kernelOutput, { recursive: true });
await mkdir(path.join(datasetOutput, "solo_search"), { recursive: true });
for (const filename of [
  "__init__.py",
  "jax_board.py",
  "jax_search.py",
  "jax_structure.py",
  "model.py",
  "train.py",
  "simulate.py",
  "verify_jax_board.py",
  "kaggle_runner.py",
]) {
  await cp(
    path.join(root, "training/solo_search", filename),
    path.join(datasetOutput, "solo_search", filename),
  );
}
if (process.env.SOLO_TRAINING_INPUT) {
  const trainingInput = path.resolve(process.env.SOLO_TRAINING_INPUT);
  await mkdir(trainingDatasetOutput, { recursive: true });
  await mkdir(path.join(trainingDatasetOutput, "solo_search"), { recursive: true });
  for (const filename of [
    "__init__.py",
    "jax_board.py",
    "jax_search.py",
    "jax_structure.py",
    "model.py",
    "train.py",
    "simulate.py",
  ]) {
    await cp(
      path.join(root, "training/solo_search", filename),
      path.join(trainingDatasetOutput, "solo_search", filename),
    );
  }
  await cp(
    trainingInput,
    path.join(trainingDatasetOutput, "initial-v13-trajectories.jsonl.gz"),
  );
  if (process.env.SOLO_SEED_INPUT_DIR) {
    const seedInput = path.resolve(process.env.SOLO_SEED_INPUT_DIR);
    for (const filename of ["train-pairs.json", "dev-pairs.json", "test-pairs.json", "latency-pairs.json"]) {
      await cp(path.join(seedInput, filename), path.join(trainingDatasetOutput, filename));
    }
  }
  await cp(
    path.join(root, "training/solo_search/kaggle/train-dataset-metadata.json"),
    path.join(trainingDatasetOutput, "dataset-metadata.json"),
  );
  await mkdir(trainingKernelOutput, { recursive: true });
  for (const filename of ["train-kernel.py", "simulate-kernel.py"]) {
    await cp(
      path.join(root, "training/solo_search/kaggle", filename),
      path.join(trainingKernelOutput, filename),
    );
  }
  await cp(
    path.join(root, "training/solo_search/kaggle/train-kernel-metadata.json"),
    path.join(trainingKernelOutput, "kernel-metadata.json"),
  );
}
for (const filename of ["kernel.py", "kernel-metadata.json"]) {
  await cp(
    path.join(root, "training/solo_search/kaggle", filename),
    path.join(kernelOutput, filename),
  );
}
await cp(
  path.join(root, "training/solo_search/kaggle/dataset-metadata.json"),
  path.join(datasetOutput, "dataset-metadata.json"),
);
await cp(
  path.join(root, "training/solo_search/artifacts/jax-fixtures.jsonl.gz"),
  path.join(datasetOutput, "jax-fixtures.jsonl.gz"),
);
const metadataPath = path.join(kernelOutput, "kernel-metadata.json");
const metadata = JSON.parse(await readFile(metadataPath, "utf8"));
if (process.env.SOLO_KAGGLE_SLUG) {
  metadata.id = `ikakun624/${process.env.SOLO_KAGGLE_SLUG}`;
  metadata.title = process.env.SOLO_KAGGLE_TITLE ?? metadata.title;
}
await writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
const datasetMetadata = JSON.parse(
  await readFile(path.join(datasetOutput, "dataset-metadata.json"), "utf8"),
);
let resumeDataset = null;
if (process.env.SOLO_RESUME_INPUT) {
  const resumeInput = path.resolve(process.env.SOLO_RESUME_INPUT);
  await mkdir(resumeDatasetOutput, { recursive: true });
  for (const filename of ["resume-state.json", "state.npz"]) {
    await cp(path.join(resumeInput, filename), path.join(resumeDatasetOutput, filename));
  }
  await cp(
    path.join(root, "training/solo_search/kaggle/resume-dataset-metadata.json"),
    path.join(resumeDatasetOutput, "dataset-metadata.json"),
  );
  resumeDataset = JSON.parse(
    await readFile(path.join(resumeDatasetOutput, "dataset-metadata.json"), "utf8"),
  ).id;
}
console.log(
  JSON.stringify({
    output,
    kernelOutput,
    datasetOutput,
    kernel: metadata.id,
    dataset: datasetMetadata.id,
    resumeDatasetOutput: resumeDataset ? resumeDatasetOutput : null,
    resumeDataset,
    trainingKernelOutput: process.env.SOLO_TRAINING_INPUT ? trainingKernelOutput : null,
    trainingDatasetOutput: process.env.SOLO_TRAINING_INPUT ? trainingDatasetOutput : null,
  }),
);
