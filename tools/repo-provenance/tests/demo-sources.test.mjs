import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repositoryRoot = new URL("../../../", import.meta.url);

test("curated ISLES26 sources resolve to current files and real commits", async () => {
  const source = await readFile(
    new URL("../app/fixtures/isles26.ts", import.meta.url),
    "utf8",
  );
  const references = [
    ...source.matchAll(
      /src\(\s*"([^"]+)"\s*,\s*"([^"]+)"(?:\s*,\s*"([0-9a-f]+)")?/g,
    ),
  ].map((match) => ({ label: match[1], path: match[2], commit: match[3] }));

  const expectedEvidence = new Map([
    ["Baseline organization", /How to add a new model/i],
    ["Data status", /N=1453|1,453/],
    ["Living model status", /t=\+2\.47/],
    ["Retractions", /52 of 55 centers/],
    ["Official sweep", /thr0\.35_k20/],
    ["Equivalence test", /NOT the pass criterion/],
  ]);

  assert.ok(references.length >= 20, "expected the curated graph to cite its evidence");
  for (const reference of references) {
    assert.ok(reference.commit, `${reference.label} must pin a commit`);
    await access(new URL(reference.path, repositoryRoot));
    execFileSync(
      "git",
      ["-C", repositoryRoot.pathname, "cat-file", "-e", `${reference.commit}:${reference.path}`],
      { stdio: "ignore" },
    );
    const pattern = expectedEvidence.get(reference.label);
    if (pattern) {
      const snapshot = execFileSync(
        "git",
        ["-C", repositoryRoot.pathname, "show", `${reference.commit}:${reference.path}`],
        { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
      );
      assert.match(snapshot, pattern, `${reference.label} must contain its claimed evidence`);
    }
  }
});
