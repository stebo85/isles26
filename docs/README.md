---
name: ISLES'26 documentation
status: active
tags: [index, isles26]
description: Entry point and ownership map for the project documentation.
---

# ISLES'26 Documentation

Start with the three documents that define current truth:

1. [Challenge and data overview](challenge/overview.md) — task, dates, release,
   metadata, and local data state.
2. [Official metrics](evaluation/official-metrics.md) — the scoring semantics
   that decide model and operating-point selection.
3. [Model status](models/status.md) — current results, retractions, production
   choice, and remaining work.

## Structure

```text
docs/
├── challenge/
│   └── overview.md                 # task, dates, data release, metadata
├── data/
│   ├── atlas-r21-characterization.md # measured legacy-data properties
│   └── atlas-r30-preprocessing.md     # native vs MNI preprocessing
├── evaluation/
│   ├── official-metrics.md           # decision surface
│   └── framework.md                  # diagnostic surface
├── models/
│   ├── status.md                     # current project truth
│   └── synthstroke-lessons.md        # one campaign's detailed history
├── lessons/
│   └── problems.md                   # durable cross-cutting lessons
├── research/
│   ├── methods.md                    # canonical method survey
│   └── challenges/                   # predecessor evidence
│       ├── atlas-r2.md
│       └── isles22.md
└── design/
    └── repository-provenance.md       # tooling design note
```

## Ownership Rules

Each fact should have one canonical home:

- Challenge facts and metadata semantics belong in `challenge/overview.md`.
- Current model numbers and decisions belong in `models/status.md`.
- Ranked metric behavior belongs in `evaluation/official-metrics.md`; richer
  explanatory metrics belong in `evaluation/framework.md`.
- A model campaign keeps detailed chronology in its own note. Only reusable
  conclusions are promoted to `lessons/problems.md`.
- Literature claims belong in `research/`; project outcomes do not.
- Other files link to the owner instead of copying its tables or summaries.

This follows [FELT's](https://github.com/cailmdaley/felt) useful core: the
directory tree carries hierarchy, frontmatter stays small, and relationships
can be expressed with `[[wikilinks]]`. Files remain ordinary GitHub-readable
Markdown, so use relative Markdown links when the reader needs a clickable
path and wikilinks for lightweight semantic relationships.

Related fibers: [[challenge/overview]], [[evaluation/official-metrics]],
[[models/status]], [[lessons/problems]], [[research/methods]].
