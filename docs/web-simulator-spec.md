# PuyoAI Web Simulator Spec v0.1

## Goal

Build a solo-play Puyo Puyo simulator and viewer that behaves as closely as practical to the Puyo side of `Puyo Puyo Tetris 2`.

The first milestone is not the AI itself. The first milestone is a deterministic simulator core that can:

- accept a board, current pair, and placement
- resolve chains and scoring
- produce replayable events for viewing
- run locally on both Mac and iPhone

## Runtime And Deployment

- App type: Web app
- Language: TypeScript
- Build tool: Vite
- Packaging target: PWA first
- Future packaging option: Capacitor
- Server requirement: none for v1
- Storage: browser local storage or IndexedDB
- AI execution: Web Worker

This project should run fully client-side in the browser. A backend API is not required for simulation, replay, scoring, or the first AI versions.

## Product Scope

### Included In v1

- simulator core
- chain replay viewer
- score and chain count display
- seeded next queue generation
- AI-ready legal action enumeration

### Excluded In v1

- ojama
- offset
- versus logic
- online play
- falling-input simulation
- human-operation UI
- undo and redo

## Rule Target

The target is `Puyo Puyo Tetris 2 / Versus / Puyo vs Puyo`, but only the solo-relevant engine behavior is implemented in v1.

For parts that are not fully confirmed for PPT2 itself, the engine should follow well-established Tsu-style community references and remain replaceable where needed.

## Core Design

### Board

- internal size: `6 x 14`
- visible rows: `12`
- hidden rows: `2`
- coordinates: `x = 0..5`, `y = 0..13`
- `y = 0` is the bottom row
- chain detection and gravity operate over all 14 rows

### Colors

- `EMPTY`
- `RED`
- `GREEN`
- `BLUE`
- `YELLOW`

The initial implementation uses 4 playable colors plus empty space.

### Pair Representation

Each pair consists of:

- `axis`
- `child`

### Action Representation

Each move is represented as:

- `column`
- `orientation`

Orientations:

- `UP`
- `RIGHT`
- `DOWN`
- `LEFT`

Rules:

- `UP` and `DOWN` place both puyos in the same column
- `LEFT` and `RIGHT` place the pair horizontally across adjacent columns
- in horizontal placements, each puyo falls independently within its own column

### Legal Actions

The simulator should expose a function that enumerates legal placements for a given pair and board.

Expected counts:

- different-color pair: up to `22`
- same-color pair: deduplicate symmetric placements to `11`

## Loss Condition

Loss is determined immediately after placement and before chain resolution.

The game is lost if both placed puyos end up outside the visible field.

Concrete rule:

- if both final placement cells have `y >= 12`, mark the move as `topout = true`
- when `topout = true`, the turn ends without continuing into normal chain resolution

## Chain Resolution

The simulator resolves one turn in this order:

1. place the pair
2. check topout
3. find connected groups of 4 or more same-color puyos
4. erase matched groups
5. apply gravity
6. repeat until no more clears occur
7. finish the turn

The simulator does not need falling-input replay to show chain playback. Replay is built from state transitions after placement.

## Scoring

Use classic Tsu-style scoring for the solo core.

Base formula:

`score = 10 * erased_count * multiplier`

Where:

`multiplier = clamp(chain_bonus + color_bonus + group_bonus, 1, 999)`

### Chain Bonus

- chain 1: `0`
- chain 2: `8`
- chain 3: `16`
- chain 4: `32`
- chain 5: `64`
- chain 6 and later: increase by `32` each step

### Color Bonus

- 1 color: `0`
- 2 colors: `3`
- 3 colors: `6`
- 4 colors: `12`
- 5 colors: `24`

### Group Bonus

- size 4: `0`
- size 5: `2`
- size 6: `3`
- size 7: `4`
- size 8: `5`
- size 9: `6`
- size 10: `7`
- size 11 or more: `10`

### Explicitly Excluded From v1

- drop bonus
- all-clear attack bonus
- garbage conversion

## Replay Event Model

Each resolved turn returns a structured event list for viewing.

Event sequence:

- `place`
- `clear`
- `gravity`
- repeated `clear` and `gravity`
- `settle`

Each event should include a full board snapshot in v1. This is heavier than a diff-only format, but it keeps the viewer simple and reliable.

## Turn Result Model

The simulator should return a single turn result object containing:

- `finalBoard`
- `topout`
- `totalChains`
- `totalScore`
- `stepScores`
- `allClear`
- `events`

`allClear` can still be reported as state information even though no all-clear bonus is applied in v1.

## Randomizer

The simulator core itself should be deterministic and RNG-free.

Random generation belongs in a separate module.

Requirements:

- seedable
- swappable implementation
- explicit queue output
- viewer next count configurable
- AI visible next count configurable

The first randomizer can be `PPT2/Tsu-like`, but it should not be baked into the core because exact PPT2 behavior is not fully locked down yet.

## Module Boundaries

Planned project structure:

- `src/core`
- `src/randomizer`
- `src/worker`
- `src/app`

Suggested responsibilities:

- `src/core`: board state, placement, gravity, clear detection, scoring, replay events
- `src/randomizer`: seed-based next queue generation
- `src/worker`: AI search and heavy simulation batches
- `src/app`: viewer and app state

## UI Direction

The first UI should prioritize observation over play.

Included:

- board view
- next view
- chain count
- total score
- replay controls for resolved events

Excluded:

- manual left-right movement
- rotation during fall
- timing-sensitive input

## Testing Strategy

Before AI work begins, the simulator should be verified with fixed golden cases.

Minimum target:

- at least 10 board-and-placement cases
- expected final board
- expected chain count
- expected total score
- expected topout behavior

## Reference Policy

- `pp-sim2` may be referenced for simulator structure and board resolution ideas
- `puyoAI.js` from `pp-sim2` is out of scope and should not be used as a design source
- official PPT2 sources and community Tsu references are both acceptable inputs when documenting behavior

## Open Items

These remain intentionally flexible until implementation or verification demands stricter choices:

- exact randomizer behavior for PPT2
- exact viewer presentation details
- whether snapshots remain the event payload long-term or later become diffs
- whether to keep all-clear as metadata only or expose it in UI immediately

## Immediate Next Step

The next implementation-phase task should be to scaffold the TypeScript web app and create a pure simulator core with tests before adding the viewer.
