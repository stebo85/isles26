---
name: Repository provenance explorer prototype
status: active
tags: [design, provenance, tooling]
description: Architecture, trust boundary, and roadmap for the repository provenance explorer.
---

# Repository Provenance Explorer Prototype

## Purpose

The prototype in `tools/repo-provenance/` tests whether a repository can be
explained as evidence-backed provenance rather than as a flat summary or raw Git
graph. It is intentionally generic: ISLES26 is the worked example and validation
fixture, not a hard-coded data model.

The product must answer four questions:

1. What is the repository trying to achieve?
2. Which evidence caused important decisions?
3. Which branches were rejected, superseded, or later corrected?
4. What pipeline and artifact lineage exist at the analyzed commit?

## Current architecture

The prototype has two separable parts.

### Deterministic repository indexer

`scripts/analyze-repo.mjs` invokes Git with explicit arguments and reads a
single commit-pinned tree. It never evaluates tracked repository content. Hooks,
fsmonitor, pagers, credentials, external diffs, and global/system Git config are
disabled. It currently extracts:

- repository identity, branch, HEAD, commit count, contributor count, and files;
- recent commits with dates, subjects, and changed paths;
- candidate change episodes based on time, subject terms, and top-level paths;
- candidate decisions, corrections, results, and negative outcomes;
- files whose paths strongly suggest build, workflow, or analysis participation.

Pipeline candidates are left unconnected and marked speculative. Filename order
is not enough evidence to assert execution order. Adjacent story episodes use a
non-causal `precedes` relationship rather than inventing motivation.

All edge labels read literally as `source relation target`; forward dataflow uses
`feeds`, while `consumes` is reserved for an operation pointing to its input.

### Interactive evidence viewer

The web application imports schema 0.1 or 0.2 JSON and renders three overview projections. The
importer fully checks the schema, enums, unique IDs, endpoints, and bounded
sizes (5 MB / 1,000 nodes / 5,000 edges) before graph layout:

- **Story:** a chronological narrative including abandoned work when history is enabled.
- **Decisions:** decision and correction provenance with typed relationships.
- **Current pipeline:** training, runtime, output, and proof artifacts at a commit.

Selecting a node exposes its question, evidence, decision or operation,
consequence, confidence, and source trace. The current-truth control can remove
superseded, retracted, and negative nodes without erasing their history.

Schema 0.2 adds semantic zoom without turning the overview into an unreadable
graph. A node may own child nodes through `parentId` and may declare the child
graph's `drilldownLens`; otherwise the viewer infers the lens from internal
relationships and child membership. Opening it replaces the overview with that
local subgraph and adds a breadcrumb back to the originating lens. The evidence
panel uses four stable disclosure axes:

- **Why:** question, evidence, decision, consequence, and considered paths;
- **How:** structured inputs, outputs, parameters, entrypoints, and operations;
- **Proof:** commit-pinned sources, bounded excerpts, source kind, and whether
  the evidence was documented, statically inspected, runtime-observed, or
  human-reviewed;
- **History:** event time, assertion time, status changes, and corrections.

The curated ISLES26 graph currently proves this interaction with two vertical
slices: the official-metrics correction expands into the changing scorer
contract and evaluator role split; the component filter expands into its
probability input, threshold, connected-component semantics, size pruning,
sweep evidence, and binary output. The generic deterministic indexer remains on
schema 0.1 until it can populate the richer fields without inventing evidence.

Visible event dates use an explicit UTC display timezone. Any text rendered on
both the server and client must avoid process-local date formatting, otherwise a
timestamp near midnight can produce different HTML and fail React hydration.

## Why the storage model should not be a DAG

Repositories contain cyclic dependencies, parallel branches, conflicting
claims, and artifacts reused across phases. The durable model should therefore
be a versioned property graph. A DAG is a view obtained by:

- selecting a repository scope and commit SHA;
- selecting time-respecting relationship types;
- hiding or collapsing cycles and superseded subgraphs;
- laying out the result according to the chosen product lens.

Two timestamps are required:

- `eventAt`: when the experiment, change, or decision happened;
- `assertedAt`: when its interpretation was documented or corrected.

This difference is material in ISLES26: several experiments happened weeks
before the official metrics caused their interpretation to be retracted.

## Agentic analysis design

The next layer should be an evidence-citing analysis pipeline:

1. **Inventory:** deterministic collection of Git, documents, configuration,
   entrypoints, reports, PRs, issues, CI, and available artifacts.
2. **Episode proposal:** cluster related changes without claiming intent.
3. **Claim extraction:** emit typed questions, evidence, decisions, results,
   artifacts, and corrections with source spans.
4. **Adversarial verification:** independently try to falsify every claim,
   check chronological consistency, and find later contradictions.
5. **Human review:** require approval for low-confidence claims and allow a
   maintainer to merge or split episodes.
6. **Incremental update:** key evidence by repository identity and commit SHA,
   then revisit only episodes affected by new commits.

Tracked repository content is untrusted input. The local prototype assumes the
operator trusts the `.git` directory; Git itself warns against running commands
inside attacker-controlled Git metadata. A service that accepts arbitrary remote
repositories must instead create a sanitized clone in an unprivileged,
credential-free, network-disabled analysis sandbox. The analysis worker must
not interpret instructions found in source files as tool authorization and must
not execute repository code.

The CLI writes JSON to stdout by default. Explicit output files must be new,
their parent directory must already exist, and their canonical location must be
outside the inspected repository. This keeps scanning read-only and prevents
case aliases or repository-provided symlinks from bypassing containment checks.

## Proposed durable store

SQLite is sufficient initially:

- `analysis_runs` — repository, commit SHA, analyzer version, timestamps;
- `nodes` — typed claims and artifacts with status and confidence;
- `edges` — typed relationships with their own evidence;
- `sources` — commit, path, line range, PR, job, URL, or artifact digest;
- `node_sources` and `edge_sources` — field-level provenance;
- `reviews` — human verdicts and corrections;
- `content_index` — optional full-text search over permitted evidence.

A graph database should be considered only after real query patterns show that
SQLite recursive queries are insufficient.

## Next tracer-bullet slices

1. Make the ISLES26 curated dataset a checked schema fixture and validate all
   referenced paths and commits.
2. Add diff and Markdown extraction with bounded, cited source snippets.
3. Add a structured agent pass for question/evidence/decision/consequence.
4. Add an independent verifier and review queue.
5. Add GitHub PR, issue-event, review, and Actions enrichment.
6. Reconstruct runtime pipelines from explicit entrypoints and callers, labeling
   static inferences separately from runtime verification.
7. Cache indexes outside target repositories and update them incrementally.

## Acceptance criteria for the next version

- Every visible claim opens at least one exact source.
- No inferred pipeline edge is presented as runtime-confirmed.
- Retractions remain visible and identify the claim they supersede.
- The current pipeline is pinned to a displayed commit SHA.
- Loading malformed or referentially invalid JSON fails clearly.
- Scanning trusted Git metadata cannot execute repository hooks or project code;
  arbitrary remote Git metadata is isolated in a sanitized clone first.
- A new commit updates only affected episodes and preserves reviewed history.

Related fibers: [[models/status]], [[lessons/problems]].
