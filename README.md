# PuyoAI

Browser-based Puyo Puyo simulator and replay viewer aimed at a PPT2-like solo ruleset.

## Current Features

- deterministic solo simulator core
- PPT2-like top rules: the third column's 12th row is the topout point, 13th-row puyos do not join clears, and 14th-row placements disappear
- chain replay viewer
- manual placement and random play
- search AI with configurable depth and beam width
- optional value-assisted search that mixes a streamed value model into leaf evaluation
- exportable search dataset for later learning experiments
- Python training toolkit for a separate learned policy baseline
- browser-usable learned policy mode on the normal viewer page

## Run

Start a local static server from the project root:

```bash
python3 -m http.server 4173
```

Then open:

`http://localhost:4173`

This app is fully client-side. No backend server is required.

For the headless parallel runner, open:

`http://localhost:4173/batch.html`

## Deploy To GitHub Pages

This repo already includes a GitHub Pages workflow:

- `.github/workflows/deploy-pages.yml`

To publish it:

1. Create a GitHub repository and push this project to the `main` branch.
2. On GitHub, open `Settings -> Pages`.
3. Set `Build and deployment -> Source` to `GitHub Actions`.
4. Push to `main`, or run the `Deploy GitHub Pages` workflow manually.

For a project repository, the site URL will usually be:

- `https://<your-user>.github.io/<repo>/`

Because this app uses relative asset paths, it should work from that subpath without additional base-path changes.

## Search AI

The right-side AI panel supports:

- `AI Analyze`: search only and show the best move plus top candidates
- `AI Move`: let the search AI play one move
- `AI x10`: let the search AI play ten moves in sequence
- `AI Run`: keep playing until stopped or topout
- `Stop`: stop continuous execution after the current search finishes
- `Export Dataset`: download the accumulated search records as JSON
- `Value Assist`: in Search mode, add the exported `value_mlp` prediction to the search leaf score

## Batch Runner

The batch runner page is a field-less parallel execution mode.

- choose `Parallel Count`, `Depth`, `Beam Width`, and `Seed Base`
- choose a `Bulk Search Profile` and `Bulk Value Assist`, or assign a different `Search Profile`, `Value Assist`, and `Value Weight` to each worker
- `Apply Bulk To All`: copy the bulk profile and value-assist settings into every worker
- `Start All`: launch all workers at once
- `Stop All`: request all workers to stop together
- `Export Slim`: download all-turn lightweight policy samples
- `Export 10+ Focus`: download full-detail samples from the 11 turns before and the trigger turn when a 10+ chain occurs
- `Export Value`: download value-learning samples with state, search summary, immediate result, and 12/24/48-turn future labels
- `Export Benchmark`: download a compact benchmark summary with per-profile totals, per-worker totals, and 7+ chain events
- each worker card shows current turn, score, max chains, worker total turns, and completed games
- the summary panel shows total turns across all workers and a compact 7+ chain histogram split by run label, so assisted and non-assisted workers can be compared directly
- benchmark exports include lightweight pre-fire features for 7+ chain events, plus bucket/profile aggregate averages for evaluation tuning
- value exports are intended for Phase 2 value-function learning: raw board state plus future max-chain, large-chain hit counts, score gain, and topout labels

## Search Profile Tuning

Phase 1 tuning keeps the search AI architecture and automatically tests small
mutations around `chain_builder_v9b`.

Run a quick local sweep:

```bash
npm run tune:v9b -- --candidates 12 --turns 1200 --games 2 --depth 3 --beam-width 16
```

For a more reliable sweep, raise `--turns`, `--candidates`, and usually use
`--depth 4 --beam-width 24` when you can wait longer. Reports are written to
`log/puyoai-tuning-report-*.json` and include the best temporary profile config.
`chain_builder_v11` is the promoted `tuned_v9b_008` candidate from the first
v9b tuning run.
`chain_builder_v12` is the promoted generation-8 evolution champion
`evolve_chain_builder_v11_g008_c007_e08280`.

To re-benchmark promising candidates from the same tuning seed, use `--only`.
For example, this reruns only `tuned_v9b_003` and `tuned_v9b_008` plus the v9b
baseline:

```bash
npm run tune:v9b -- --only 3,8 --turns 8000 --games 4 --depth 3 --beam-width 16 --seed v9b-tune
```

## Search Profile Evolution

For repeated automatic tuning from the current `chain_builder_v11` baseline, use
the staged evolution runner:

```bash
npm run evolve:v11 -- \
  --generations 20 \
  --population 36 \
  --stage1-turns 3000 \
  --stage2-turns 5000 \
  --stage3-turns 10000 \
  --parallel-profiles 4 \
  --depth 3 \
  --beam-width 24
```

Each generation creates 36 new candidate profiles, keeps `chain_builder_v11`,
the current champion, and recent hall-of-fame candidates as protected
baselines, then runs three elimination stages. Stage seed sets are randomized
for every generation and stage, but every candidate in the same comparison uses
the same seed set for fairness. Candidate configs are de-duplicated by weight
hash and distance so later generations do not keep testing the same v11
neighborhood when the champion has not changed.

The runner writes `log/puyoai-evolution-report-*.json` as it goes. Only promote
a result into `search-profiles.js` after a longer benchmark confirms it beats
v11 on separate seeds.

## Learned Policy Training

The search AI remains the main baseline. The `training/` package is a separate supervised-learning pipeline that imitates search outputs from exported datasets.

### 1. Prepare a Python environment

On Apple Silicon, the scripts will automatically prefer `mps` if your PyTorch build supports it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```

### 2. Export datasets from the app

- use `Export Slim` on the batch runner to save all-turn policy samples
- optionally use `Export 10+ Focus` to save detailed samples around 10-chain-or-larger triggers

### 3. Train a learned policy

Run training from the repository root with module syntax:

```bash
python3 -m training.train_policy \
  --slim /path/to/puyoai-search-slim.json \
  --focus /path/to/puyoai-search-chain-focus.json \
  --output models/policy_mlp.pt
```

Useful knobs:

- `--epochs 12`
- `--batch-size 256`
- `--focus-weight 2.0`
- `--distill-weight 0.2`
- `--device auto|mps|cpu|cuda`

The trainer prints one JSON line per epoch and saves the checkpoint with the best validation top-1 accuracy.

### 4. Evaluate a checkpoint

```bash
python3 -m training.evaluate_policy \
  --checkpoint models/policy_mlp.pt \
  --slim /path/to/puyoai-search-slim.json \
  --focus /path/to/puyoai-search-chain-focus.json
```

This reports combined, slim-only, and focus-only top-1/top-3 metrics.

### 5. Inspect one sample

```bash
python3 -m training.predict_policy \
  --checkpoint models/policy_mlp.pt \
  --input /path/to/puyoai-search-slim.json \
  --index 0
```

`predict_policy` also accepts a single JSON object shaped like either:

- an exported sample with `state` and `bestActionKey`
- a wrapper with `state`
- a raw state object with `boardRows`, `currentPair`, and `nextQueue`

The learned policy is intentionally separate from the search AI so both can coexist in this repository and be compared later.

### 6. Export for the web viewer

After training, convert the `.pt` checkpoint into the JSON asset used by the browser:

```bash
python3 -m training.export_web_policy \
  --checkpoint models/policy_mlp.pt \
  --output models/policy_mlp.web.json
```

Once `models/policy_mlp.web.json` exists, the normal viewer page can switch between:

- `Search`: the original beam-search AI
- `Learned`: the exported MLP policy

Open `http://localhost:4173`, change `AI Mode` to `Learned`, and then use `AI Move` / `AI Run` to watch the learned policy stack on the same field UI.

### 7. Train a whole model suite in one command

The repo also includes a suite runner and a default preset list in [training/model_suite.json](/Users/takuma/PuyoAI/training/model_suite.json:1).

```bash
python3 -m training.run_model_suite \
  --slim /path/to/puyoai-search-slim.json \
  --focus /path/to/puyoai-search-chain-focus.json
```

This sequentially:

- trains every configured model preset
- evaluates each checkpoint
- exports each checkpoint to `models/<id>.web.json`
- writes `models/manifest.json` for the web viewer

After that, the normal viewer can switch among multiple learned models with the `Learned Model` selector.

## Streaming Value Training

For larger value-function runs, avoid browser downloads. The CLI can generate
search-value samples and train from the stream without holding a giant JSON file
in memory.

```bash
python3 -m training.train_value_stream \
  --turns 300000 \
  --games 30 \
  --workers 2 \
  --depth 3 \
  --beam-width 16 \
  --search-profile chain_builder_v11 \
  --output models/value_mlp.pt
```

This launches `tools/generate-value-stream.js`, which writes one `search_value`
JSON object per line. Python reads those lines immediately, keeps only a bounded
replay buffer, and saves the trained value checkpoint. The default target uses
the 48-turn future label, while the stream also carries 12- and 24-turn labels.
During long runs, `models/value_mlp.pt` is saved at every `--save-every`
interval, while `models/value_mlp.best.pt` is updated only when validation loss
improves.
`--workers` starts parallel Node generators and splits `--turns` across them, so
`--turns 300000 --workers 2` means about 150,000 turns per generator and
300,000 turns total.
To continue training from an existing value checkpoint, pass `--resume` and use
a new seed to avoid replaying the same deterministic sequence:

```bash
python3 -m training.train_value_stream \
  --turns 300000 \
  --games 30 \
  --workers 2 \
  --depth 3 \
  --beam-width 16 \
  --search-profile chain_builder_v11 \
  --seed value-stream-2 \
  --resume models/value_mlp.pt \
  --output models/value_mlp.pt
```

If you only want to inspect or archive the stream manually:

```bash
npm run generate:value -- --turns 10000 --games 5 > log/value-sample.jsonl
```

After training, export the latest checkpoint for the browser:

```bash
npm run export:value
```

Then open the normal viewer, keep `AI Mode` as `Search`, set `Value Assist` to
`On`, and compare against the same search profile with `Value Assist` off. The
default `Value Weight` is intentionally conservative; raise it only after a
benchmark shows the value-assisted search is better.

## Test

```bash
node --test
```

For the Python-side scripts, a lightweight syntax check is:

```bash
python3 -m compileall training
```
