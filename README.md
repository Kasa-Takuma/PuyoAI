# PuyoAI

Browser-based Puyo Puyo simulator and replay viewer aimed at a PPT2-like solo ruleset.

## Current Features

- deterministic solo simulator core
- chain replay viewer
- manual placement and random play
- search AI with configurable depth and beam width
- exportable search dataset for later learning experiments

## Run

Start a local static server from the project root:

```bash
python3 -m http.server 4173
```

Then open:

`http://localhost:4173`

This app is fully client-side. No backend server is required.

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
- `Export Dataset`: download the accumulated search records as JSON

## Test

```bash
node --test
```
