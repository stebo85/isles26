import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the RepoTrace application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  assert.equal(response.headers.get("x-content-type-options"), "nosniff");
  assert.equal(response.headers.get("x-frame-options"), "DENY");

  const html = await response.text();
  assert.match(html, /<title>RepoTrace · Repository provenance explorer<\/title>/i);
  assert.match(html, /Repository provenance/);
  assert.match(html, /RepoTrace/);
  assert.match(html, /Load analysis/);
  assert.match(html, /isles26/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("keeps the bounded generic importer and evidence contract in the client", async () => {
  const [explorer, schema, validator, packageJson] = await Promise.all([
    readFile(new URL("../app/components/ProvenanceExplorer.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/provenance.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/validate-analysis.mjs", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(explorer, /repositoryAnalysisError/);
  assert.match(explorer, /MAX_ANALYSIS_FILE_BYTES/);
  assert.match(explorer, /accept="application\/json,\.json"/);
  assert.match(explorer, /repository content is treated as untrusted evidence/);
  assert.match(schema, /schemaVersion:\s*"0\.1"/);
  assert.match(schema, /"superseded"/);
  assert.match(schema, /"retracted"/);
  assert.match(validator, /MAX_ANALYSIS_NODES = 1_000/);
  assert.match(packageJson, /"@xyflow\/react"/);
  assert.match(packageJson, /"elkjs"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
