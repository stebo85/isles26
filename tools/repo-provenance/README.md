# RepoTrace prototype

RepoTrace turns repository evidence into three progressively disclosed views:

1. **Story** — the important turning points in time order.
2. **Decisions** — what was selected, rejected, superseded, or corrected.
3. **Current pipeline** — the artifacts and operations that produce today's result.

The application ships with a curated ISLES26 analysis as a worked example. It
also accepts schema-compatible JSON produced by the bundled deterministic
indexer, so the viewer itself is repository-independent.

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm ci
npm run dev
```

Open the local URL printed by the development server.

## Point it at another repository

The indexer reads a commit-pinned Git snapshot and tracked filenames. It does
not execute project code, package scripts, workflows, or build tools. The
operator must trust the repository's `.git` directory; tracked content is
treated as untrusted. Git is invoked with hooks, fsmonitor, pagers, credentials,
external diffs, and global/system configuration disabled as defense in depth.

```bash
npm run analyze -- \
  --repo /path/to/repository \
  --out /tmp/repotrace-analysis.json
```

`--out` must name a new file outside the inspected repository, and its parent
directory must already exist. The parent is resolved canonically before the
write. Omit `--out` to write JSON to stdout; existing files are never replaced.

Choose **Load analysis** in the application and select the generated JSON file.

The deterministic pass produces candidate change episodes and candidate
pipeline files. Those nodes are deliberately marked `strongly-inferred` or
`speculative`: it is an ingestion layer, not a substitute for an evidence-citing
agent review.

## Commands

- `npm run analyze -- --repo <path> --out <file>` — generate schema 0.1 JSON.
- `npm run dev` — run the interactive application locally.
- `npm run build` — compile the production application.
- `npm run test:unit` — test commit parsing, classification, and episode grouping.
- `npm test` — build and run all tests.

## Data contract

Each node records:

- its type, status, event time, and optional assertion time;
- the question, evidence, decision or operation, and consequence;
- whether the claim is explicit, strongly inferred, or speculative;
- source paths, commits, line ranges, or URLs;
- which progressive views should display it.

The underlying data is a versioned directed property graph. The UI renders an
acyclic projection appropriate to the selected view; it does not require the
stored repository model itself to be a DAG.

## Prototype boundary

Implemented:

- deterministic, read-only local Git ingestion;
- commit classification and episode grouping;
- conservative pipeline-candidate discovery;
- interactive graph layout, evidence drill-down, and history/current filtering;
- bounded generic JSON import with complete schema, enum, ID, and edge checks;
- curated ISLES26 evidence demonstrating corrections and retractions.

Not yet implemented:

- agentic reading of diffs, documents, PRs, issues, CI logs, and runtime artifacts;
- adversarial verification of generated claims;
- GitHub/GitLab connectors;
- incremental indexes keyed by commit SHA;
- a human review queue and shareable curated provenance files.

The current browser importer accepts at most 5 MB, 1,000 nodes, and 5,000 edges.
For a deployed build, set `SITE_URL` to the canonical `https://` origin used by
Open Graph metadata.

See [the repository-level design note](../../docs/repository_provenance_prototype.md)
for the architecture and next implementation slices.
