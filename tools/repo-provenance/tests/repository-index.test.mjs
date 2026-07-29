import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import {
  analyzeRepository,
  buildEpisodes,
  classifyCommit,
  parseChangedPaths,
  parseCommitMetadata,
} from "../scripts/repository-index.mjs";

test("classifies corrections before generic decisions", () => {
  assert.equal(classifyCommit("Fix threshold selection and ship defaults"), "correction");
  assert.equal(classifyCommit("Select the center-held-out model"), "decision");
  assert.equal(classifyCommit("Run K negative result"), "negative");
  assert.equal(classifyCommit("Record benchmark results"), "result");
  assert.equal(classifyCommit("Add project README"), "artifact");
});

test("parses NUL-delimited commit metadata without interpreting control characters", () => {
  const firstSha = "a".repeat(40);
  const secondSha = "b".repeat(40);
  const raw = [
    firstSha, "2026-07-28T00:00:00Z", "Fix geometry \u001e without injection", "",
    secondSha, "2026-07-27T00:00:00Z", "Add model report", "", "",
  ].join("\0");
  const commits = parseCommitMetadata(raw);
  assert.equal(commits.length, 2);
  assert.equal(commits[0].subject, "Fix geometry \u001e without injection");
  assert.deepEqual(commits[0].files, []);
  assert.equal(commits[1].subject, "Add model report");
});

test("rejects malformed commit metadata", () => {
  assert.throws(
    () => parseCommitMetadata(["not-a-sha", "2026-07-28T00:00:00Z", "subject", ""].join("\0")),
    /malformed commit metadata/,
  );
});

test("keeps only paths that resolve in the post-commit tree", () => {
  assert.deepEqual(
    parseChangedPaths(["D", "old/path.sh", "A", "new/path.sh", "M", "README.md", ""].join("\0")),
    ["new/path.sh", "README.md"],
  );
  assert.throws(() => parseChangedPaths(["M", "file", "D"].join("\0")), /malformed changed paths/);
});

test("groups nearby related commits but splits unrelated work", () => {
  const commitsNewestFirst = [
    { sha: "3", date: "2026-05-10T00:00:00Z", subject: "Add website", files: ["web/page.tsx"] },
    { sha: "2", date: "2026-05-02T00:00:00Z", subject: "Fix metric report", files: ["eval/report.py"] },
    { sha: "1", date: "2026-05-01T00:00:00Z", subject: "Add metric tests", files: ["eval/test.py"] },
  ];
  const episodes = buildEpisodes(commitsNewestFirst, 10);
  assert.equal(episodes.length, 2);
  assert.deepEqual(episodes[0].commits.map((commit) => commit.sha), ["1", "2"]);
  assert.deepEqual(episodes[1].commits.map((commit) => commit.sha), ["3"]);
});

test("pins every scan field to one commit and produces stable content", () => {
  const repository = new URL("../../..", import.meta.url).pathname;
  const first = analyzeRepository(repository, { limit: 1 });
  const second = analyzeRepository(repository, { limit: 1 });
  assert.deepEqual(first, second);
  assert.equal(first.repository.analyzedAt, first.repository.lastCommitAt);
  assert.notEqual(first.repository.firstCommitAt, first.repository.lastCommitAt);
  assert.equal("path" in first.repository, false);
  for (const node of first.nodes) {
    for (const source of node.sources) {
      if (!source.path) continue;
      assert.doesNotThrow(() => {
        execFileSync(
          "git",
          ["-C", repository, "cat-file", "-e", `${source.commit}:${source.path}`],
          { stdio: "ignore" },
        );
      }, `${source.commit}:${source.path} must resolve`);
    }
  }
});
