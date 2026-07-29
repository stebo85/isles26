import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const script = new URL("../scripts/analyze-repo.mjs", import.meta.url).pathname;
const repository = new URL("../../..", import.meta.url).pathname;

test("writes clean JSON to stdout by default", () => {
  const result = spawnSync(process.execPath, [script, "--repo", repository, "--limit", "1"], {
    encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
  const analysis = JSON.parse(result.stdout);
  assert.equal(analysis.schemaVersion, "0.1");
  assert.match(result.stderr, /Indexed isles26/);
});

test("refuses in-repository and existing output targets", () => {
  const inRepository = join(repository, "repotrace-must-not-write.json");
  const inside = spawnSync(
    process.execPath,
    [script, "--repo", repository, "--limit", "1", "--out", inRepository],
    { encoding: "utf8" },
  );
  assert.notEqual(inside.status, 0);
  assert.match(inside.stderr, /outside the repository/);

  const overwrite = spawnSync(
    process.execPath,
    [script, "--repo", repository, "--limit", "1", "--out", "/dev/null"],
    { encoding: "utf8" },
  );
  assert.notEqual(overwrite.status, 0);
  assert.match(overwrite.stderr, /exist|EEXIST/i);

  const caseAlias = repository.replace("/Users/", "/users/");
  if (caseAlias !== repository && existsSync(caseAlias)) {
    const aliasedInside = spawnSync(
      process.execPath,
      [script, "--repo", repository, "--limit", "1", "--out", join(caseAlias, "repotrace-alias.json")],
      { encoding: "utf8" },
    );
    assert.notEqual(aliasedInside.status, 0);
    assert.match(aliasedInside.stderr, /outside the repository/);
  }
});
