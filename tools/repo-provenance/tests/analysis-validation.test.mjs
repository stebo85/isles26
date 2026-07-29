import assert from "node:assert/strict";
import test from "node:test";
import { repositoryAnalysisError } from "../app/lib/validate-analysis.mjs";

const node = {
  id: "one",
  title: "One",
  summary: "Summary",
  kind: "decision",
  status: "current",
  eventAt: "2026-07-28T00:00:00Z",
  lenses: ["story"],
  question: "Question?",
  evidence: "Evidence.",
  decision: "Decision.",
  consequence: "Consequence.",
  confidence: "explicit",
  sources: [{ label: "Commit", commit: "a".repeat(40) }],
};

const valid = {
  schemaVersion: "0.1",
  repository: {
    name: "fixture",
    branch: "main",
    head: "a".repeat(40),
    analyzedAt: "2026-07-28T00:00:00Z",
  },
  summary: "Fixture analysis.",
  stats: { commits: 1, contributors: 1, files: 1, decisions: 1, corrections: 0 },
  nodes: [node],
  edges: [],
};

test("accepts a complete schema 0.1 analysis", () => {
  assert.equal(repositoryAnalysisError(valid), null);
});

test("rejects missing fields and invalid enums", () => {
  assert.match(repositoryAnalysisError({ ...valid, stats: undefined }), /statistics/i);
  assert.match(
    repositoryAnalysisError({ ...valid, nodes: [{ ...node, kind: "html-script" }] }),
    /nodes/i,
  );
});

test("rejects duplicate IDs and dangling edges", () => {
  assert.match(repositoryAnalysisError({ ...valid, nodes: [node, { ...node }] }), /unique/i);
  assert.match(
    repositoryAnalysisError({
      ...valid,
      edges: [{ id: "edge", source: "one", target: "missing", relation: "precedes", lenses: ["story"] }],
    }),
    /missing node/i,
  );
});

test("requires a full snapshot SHA and commit-pinned paths", () => {
  assert.match(
    repositoryAnalysisError({
      ...valid,
      repository: { ...valid.repository, head: "abc123" },
    }),
    /metadata/i,
  );
  assert.match(
    repositoryAnalysisError({
      ...valid,
      nodes: [{ ...node, sources: [{ label: "Loose path", path: "scripts/run.sh" }] }],
    }),
    /nodes/i,
  );
});
