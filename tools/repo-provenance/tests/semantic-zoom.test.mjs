import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";
import { isles26Analysis } from "../app/fixtures/isles26.ts";
import { projectGraph } from "../app/lib/provenance.ts";

const repositoryRoot = new URL("../../../", import.meta.url).pathname;

test("opens the decision drill-down with its relationships from the Story lens", () => {
  const projection = projectGraph(isles26Analysis, "story", "official-metrics", true);
  assert.equal(projection.lens, "decisions");
  assert.equal(projection.nodes.length, 6);
  assert.equal(projection.edges.length, 7);
});

test("current-truth drill-down excludes superseded children before selection", () => {
  const projection = projectGraph(isles26Analysis, "story", "official-metrics", false);
  assert.equal(projection.nodes.some((node) => node.id === "official-v1-contract"), false);
  assert.equal(projection.nodes.length, 5);
  assert.ok(projection.edges.every((edge) =>
    projection.nodes.some((node) => node.id === edge.source) &&
    projection.nodes.some((node) => node.id === edge.target)));
});

test("infers the scoped lens from a generic imported child graph", () => {
  const parent = {
    ...isles26Analysis.nodes.find((node) => node.id === "component-filter"),
    id: "multi-lens-parent",
    lenses: ["decisions", "pipeline"],
    drilldownLens: undefined,
  };
  const first = {
    ...isles26Analysis.nodes.find((node) => node.id === "filter-threshold"),
    id: "pipeline-child-a",
    parentId: parent.id,
  };
  const second = {
    ...isles26Analysis.nodes.find((node) => node.id === "filter-label-components"),
    id: "pipeline-child-b",
    parentId: parent.id,
  };
  const analysis = {
    ...isles26Analysis,
    nodes: [parent, first, second],
    edges: [{
      id: "pipeline-child-edge",
      source: first.id,
      target: second.id,
      relation: "precedes",
      lenses: ["pipeline"],
    }],
  };

  const projection = projectGraph(analysis, "decisions", parent.id, true);
  assert.equal(projection.lens, "pipeline");
  assert.deepEqual(projection.edges.map((edge) => edge.id), ["pipeline-child-edge"]);
});

test("every curated evidence excerpt occurs verbatim in its pinned source", () => {
  for (const node of isles26Analysis.nodes) {
    for (const source of node.sources) {
      if (!source.excerpt || !source.path || !source.commit) continue;
      const snapshot = execFileSync(
        "git",
        ["-C", repositoryRoot, "show", `${source.commit}:${source.path}`],
        { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
      );
      assert.ok(
        snapshot.includes(source.excerpt),
        `${node.id}: excerpt from ${source.path} must be verbatim`,
      );
    }
  }
});
