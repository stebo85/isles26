#!/usr/bin/env node

import { realpath, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve } from "node:path";
import { analyzeRepository, runGit } from "./repository-index.mjs";

function option(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} requires a value`);
  return value;
}

function usage() {
  console.log(`RepoTrace deterministic repository indexer

Usage:
  npm run analyze -- --repo /path/to/repository --out /path/to/analysis.json

Options:
  --repo   Local Git repository to inspect (default: current directory)
  --out    New JSON file to create outside the target repository (default: stdout)
  --limit  Maximum recent commits to inspect (default: 240)

The indexer is read-only with respect to the target repository. It never runs
repository code, hooks, package scripts, or workflows.`);
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  usage();
  process.exit(0);
}

try {
  const repository = resolve(option("--repo", process.cwd()));
  const outputValue = option("--out", null);
  const limit = Number(option("--limit", "240"));
  if (!Number.isInteger(limit) || limit < 1 || limit > 5000) {
    throw new Error("--limit must be an integer from 1 to 5000");
  }
  let output = null;
  if (outputValue) {
    const requestedOutput = resolve(outputValue);
    const targetRoot = await realpath(resolve(runGit(repository, ["rev-parse", "--show-toplevel"])));
    let outputParent;
    try {
      outputParent = await realpath(dirname(requestedOutput));
    } catch {
      throw new Error("--out parent directory must already exist");
    }
    output = resolve(outputParent, basename(requestedOutput));
    const fromTarget = relative(targetRoot, output);
    if (fromTarget === "" || (!fromTarget.startsWith("..") && !isAbsolute(fromTarget))) {
      throw new Error("--out must be outside the repository being inspected");
    }
  }

  const analysis = analyzeRepository(repository, { limit });
  const serialized = `${JSON.stringify(analysis, null, 2)}\n`;
  if (!output) {
    process.stdout.write(serialized);
    console.error(`Indexed ${analysis.repository.name} at ${analysis.repository.head.slice(0, 12)}`);
  } else {
    await writeFile(output, serialized, { encoding: "utf8", flag: "wx" });
    console.log(`Indexed ${analysis.repository.name} at ${analysis.repository.head.slice(0, 12)}`);
    console.log(`Wrote ${analysis.nodes.length} nodes and ${analysis.edges.length} edges to ${output}`);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
