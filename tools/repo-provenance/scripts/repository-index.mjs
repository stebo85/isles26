import { execFileSync } from "node:child_process";
import { basename, dirname, relative, resolve, sep } from "node:path";

const DECISION_PATTERN = /\b(adopt|choose|deprecat|drop|replace|retire|select|ship|stop|switch)\w*/i;
const CORRECTION_PATTERN = /\b(correct|fix|invalidate|retract|repair|restore|regression)\w*/i;
const NEGATIVE_PATTERN = /\b(abandon|fail|negative|no gain|regress|worse)\w*/i;
const RESULT_PATTERN = /\b(benchmark|complete|done|eval|result|score|summary|win)\w*/i;
const STOP_WORDS = new Set([
  "add", "after", "and", "before", "for", "from", "into", "latest", "more",
  "next", "run", "the", "this", "update", "with",
]);

const SAFE_GIT_OPTIONS = [
  "-c", "core.hooksPath=/dev/null",
  "-c", "core.fsmonitor=false",
  "-c", "core.pager=cat",
  "-c", "pager.log=false",
  "-c", "credential.helper=",
  "-c", "protocol.file.allow=never",
];

export function runGitRaw(repository, args) {
  return execFileSync("git", [...SAFE_GIT_OPTIONS, "-C", repository, ...args], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      GIT_CONFIG_GLOBAL: "/dev/null",
      GIT_CONFIG_SYSTEM: "/dev/null",
      GIT_EXTERNAL_DIFF: "",
      GIT_PAGER: "cat",
      GIT_TERMINAL_PROMPT: "0",
      PAGER: "cat",
    },
  });
}

export function runGit(repository, args) {
  return runGitRaw(repository, args).trim();
}

export function classifyCommit(subject) {
  if (CORRECTION_PATTERN.test(subject)) return "correction";
  if (NEGATIVE_PATTERN.test(subject)) return "negative";
  if (DECISION_PATTERN.test(subject)) return "decision";
  if (RESULT_PATTERN.test(subject)) return "result";
  return "artifact";
}

export function subjectTokens(subject) {
  return new Set(
    subject
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .split(/\s+/)
      .filter((token) => token.length > 2 && !STOP_WORDS.has(token)),
  );
}

function topLevelPaths(files) {
  return new Set(files.map((file) => file.split("/")[0]).filter(Boolean));
}

function intersects(left, right) {
  for (const value of left) if (right.has(value)) return true;
  return false;
}

export function parseCommitMetadata(raw) {
  const fields = raw.split("\0");
  const commits = [];
  for (let index = 0; index + 3 < fields.length; index += 4) {
    const [sha, date, subject, terminator] = fields.slice(index, index + 4);
    if (!sha && !date && !subject && !terminator) continue;
    if (!/^[0-9a-f]{40}$/i.test(sha) || Number.isNaN(Date.parse(date)) || terminator !== "") {
      throw new Error("Git returned malformed commit metadata.");
    }
    commits.push({ sha, date, subject, files: [] });
  }
  return commits;
}

export function parseChangedPaths(raw) {
  const fields = raw.split("\0");
  if (fields.at(-1) === "") fields.pop();
  if (fields.length % 2 !== 0) throw new Error("Git returned malformed changed paths.");

  const paths = [];
  for (let index = 0; index < fields.length; index += 2) {
    const status = fields[index];
    const path = fields[index + 1];
    if (!/^[A-Z][0-9]*$/.test(status) || !path) {
      throw new Error("Git returned malformed changed paths.");
    }
    // A source is resolved against the post-commit tree. Deleted paths exist
    // only in the parent and therefore cannot form an exact commit:path pair.
    if (!status.startsWith("D")) paths.push(path);
  }
  return paths;
}

function readCommits(root, head, limit) {
  const metadata = runGitRaw(root, [
    "log",
    "-z",
    `-n${limit}`,
    "--date=iso-strict",
    "--format=%H%x00%aI%x00%s%x00",
    head,
  ]);
  return parseCommitMetadata(metadata).map((commit) => ({
    ...commit,
    files: parseChangedPaths(runGitRaw(root, [
      "diff-tree", "--root", "--no-commit-id", "--name-status", "-z", "-r", "--no-renames", commit.sha,
    ])),
  }));
}

export function buildEpisodes(commits, maxEpisodes = 14) {
  const chronological = [...commits].reverse();
  const episodes = [];

  for (const commit of chronological) {
    const last = episodes.at(-1);
    if (!last) {
      episodes.push({ commits: [commit] });
      continue;
    }

    const previous = last.commits.at(-1);
    const gapMs = new Date(commit.date).getTime() - new Date(previous.date).getTime();
    const relatedPaths = intersects(
      topLevelPaths(commit.files),
      topLevelPaths(last.commits.flatMap((item) => item.files)),
    );
    const relatedWords = intersects(
      subjectTokens(commit.subject),
      new Set(last.commits.flatMap((item) => [...subjectTokens(item.subject)])),
    );
    const closeInTime = gapMs <= 3 * 24 * 60 * 60 * 1000;

    if (closeInTime && (relatedPaths || relatedWords || last.commits.length < 2)) {
      last.commits.push(commit);
    } else {
      episodes.push({ commits: [commit] });
    }
  }

  if (episodes.length <= maxEpisodes) return episodes;
  return episodes.slice(-maxEpisodes);
}

function mostSignificantCommit(commits) {
  const rank = { correction: 5, negative: 4, decision: 3, result: 2, artifact: 1 };
  return [...commits].sort(
    (a, b) => rank[classifyCommit(b.subject)] - rank[classifyCommit(a.subject)],
  )[0];
}

function eventQuestion(kind) {
  if (kind === "correction") return "Which earlier assumption or implementation needed correction?";
  if (kind === "negative") return "Which attempted direction failed to produce the intended result?";
  if (kind === "decision") return "Which implementation direction did the repository select?";
  if (kind === "result") return "What result changed the repository's understanding?";
  return "What repository capability or artifact changed?";
}

function episodeNode(episode, index) {
  const anchor = mostSignificantCommit(episode.commits);
  const kind = classifyCommit(anchor.subject);
  const paths = [...topLevelPaths(episode.commits.flatMap((commit) => commit.files))];
  const status = kind === "negative" ? "negative" : "current";
  const nodeKind = kind === "negative" ? "result" : kind;
  const subjects = episode.commits.map((commit) => commit.subject);
  const start = episode.commits[0];
  const end = episode.commits.at(-1);

  return {
    id: `episode-${index}-${anchor.sha.slice(0, 8)}`,
    title: anchor.subject,
    summary: `${episode.commits.length} commit${episode.commits.length === 1 ? "" : "s"} across ${paths.slice(0, 4).join(", ") || "repository root"}.`,
    kind: nodeKind,
    status,
    eventAt: end.date,
    assertedAt: kind === "correction" ? end.date : undefined,
    lenses: ["story", ...(kind === "artifact" ? [] : ["decisions"])],
    question: eventQuestion(kind),
    evidence: `Observed commits from ${start.sha.slice(0, 8)} through ${end.sha.slice(0, 8)}, changing ${paths.slice(0, 6).join(", ") || "root files"}.`,
    decision: subjects.slice(-3).join(" · "),
    consequence:
      "This is a deterministic candidate episode. Its meaning should be reviewed or enriched by an evidence-citing analysis agent.",
    confidence: "strongly-inferred",
    sources: episode.commits.slice(-3).map((commit) => ({
      label: commit.subject,
      path: commit.files[0],
      commit: commit.sha,
    })),
  };
}

function isPipelineCandidate(path) {
  const name = basename(path);
  return (
    /^(Dockerfile|Makefile|package\.json|pyproject\.toml|compose\.ya?ml)$/i.test(name) ||
    /(^|\/)(workflows?|pipelines?|scripts?)\//i.test(path) ||
    /(^|\/)analysis_[a-z0-9_-]+\.(sh|py)$/i.test(path) ||
    /\.(workflow|pipeline)\.ya?ml$/i.test(path)
  );
}

function pipelineNodes(files, commits, eventAt, head) {
  const candidates = files.filter(isPipelineCandidate).slice(0, 24);
  return candidates.map((path, index) => {
    const touchingCommit = commits.find((commit) => commit.files.includes(path));
    const siblingOrder = basename(path).match(/(?:analysis_)?(\d+)/)?.[1];
    return {
      id: `pipeline-${index}-${path.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`,
      title: basename(path),
      summary: `Detected pipeline candidate in ${dirname(path) === "." ? "the repository root" : dirname(path)}.`,
      kind: "pipeline-step",
      status: "uncertain",
      eventAt: touchingCommit?.date ?? eventAt,
      lenses: ["pipeline"],
      question: "Does this file participate in the current build, runtime, or analysis pipeline?",
      evidence: `Its path and filename match a high-signal pipeline pattern${siblingOrder ? ` with sequence ${siblingOrder}` : ""}.`,
      decision: "Treat as a candidate until its inputs, outputs, and callers are inspected.",
      consequence: "No execution relationship is asserted without further evidence.",
      confidence: "speculative",
      sources: [{
        label: "Pipeline candidate",
        path,
        commit: touchingCommit?.sha ?? head,
      }],
    };
  });
}

function chainEdges(nodes, lens, prefix) {
  const visible = nodes.filter((node) => node.lenses.includes(lens));
  return visible.slice(1).map((node, index) => ({
    id: `${prefix}-${index}`,
    source: visible[index].id,
    target: node.id,
    relation: "precedes",
    lenses: [lens],
  }));
}

export function analyzeRepository(repository, { limit = 240 } = {}) {
  const root = runGit(repository, ["rev-parse", "--show-toplevel"]);
  const head = runGit(root, ["rev-parse", "HEAD"]);
  const branch = runGit(root, ["branch", "--show-current"]) || "detached HEAD";
  const files = runGitRaw(root, ["ls-tree", "-r", "-z", "--name-only", head])
    .split("\0").filter((path) => path !== "");
  const commits = readCommits(root, head, limit);
  const commitCount = Number(runGit(root, ["rev-list", "--count", head]));
  const contributorText = runGit(root, ["shortlog", "-sne", head]);
  const firstCommitAt = runGit(root, [
    "log", "--max-parents=0", "--reverse", "-1", "--format=%aI", head,
  ]);
  const lastCommitAt = runGit(root, ["show", "-s", "--format=%aI", head]);
  const contributors = contributorText ? contributorText.split("\n").length : 0;
  const episodes = buildEpisodes(commits);
  const storyNodes = episodes.map(episodeNode);
  const pipeline = pipelineNodes(
    files,
    commits,
    commits[0]?.date ?? lastCommitAt,
    head,
  );
  const nodes = [...storyNodes, ...pipeline];
  const corrections = storyNodes.filter((node) => node.kind === "correction").length;
  const decisions = storyNodes.filter((node) => node.lenses.includes("decisions")).length;

  return {
    schemaVersion: "0.1",
    repository: {
      name: basename(root),
      branch,
      head,
      analyzedAt: lastCommitAt,
      firstCommitAt,
      lastCommitAt,
    },
    summary:
      `Deterministic first pass over ${commitCount.toLocaleString()} commits and ${files.length.toLocaleString()} tracked files. Candidate decisions require evidence review.`,
    stats: {
      commits: commitCount,
      contributors,
      files: files.length,
      decisions,
      corrections,
    },
    nodes,
    edges: [
      ...chainEdges(storyNodes, "story", "story"),
      ...chainEdges(storyNodes, "decisions", "decision"),
    ],
  };
}

export function displayPath(path, repository) {
  const resolved = resolve(path);
  const root = resolve(repository);
  return resolved.startsWith(`${root}${sep}`) ? relative(root, resolved) : resolved;
}
