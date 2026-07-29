export const MAX_ANALYSIS_FILE_BYTES = 5 * 1024 * 1024;
export const MAX_ANALYSIS_NODES = 1_000;
export const MAX_ANALYSIS_EDGES = 5_000;

const LENSES = new Set(["story", "decisions", "pipeline"]);
const STATUSES = new Set(["current", "superseded", "retracted", "negative", "uncertain"]);
const KINDS = new Set([
  "question", "evidence", "decision", "result", "correction", "artifact",
  "pipeline-step", "input", "output", "proof", "blocker",
]);
const CONFIDENCE = new Set(["explicit", "strongly-inferred", "speculative"]);
const RELATIONS = new Set([
  "motivated", "measured", "supports", "contradicts", "selects", "supersedes",
  "produces", "consumes", "feeds", "precedes", "guards", "blocks",
]);

const isRecord = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const isText = (value, max = 10_000) => typeof value === "string" && value.length > 0 && value.length <= max;
const isOptionalText = (value, max = 10_000) => value === undefined || isText(value, max);
const isDate = (value) => isText(value, 64) && !Number.isNaN(Date.parse(value));
const isOptionalDate = (value) => value === undefined || isDate(value);
const isCount = (value) => Number.isSafeInteger(value) && value >= 0 && value <= 1_000_000_000;

function validLensList(value) {
  return Array.isArray(value) && value.length > 0 && value.length <= LENSES.size &&
    new Set(value).size === value.length && value.every((lens) => LENSES.has(lens));
}

function validSource(source) {
  if (!isRecord(source) || !isText(source.label, 500)) return false;
  if (!isOptionalText(source.path, 4_096) || !isOptionalText(source.commit, 64) ||
      !isOptionalText(source.lines, 128) || !isOptionalText(source.url, 4_096)) return false;
  if (source.commit !== undefined && !/^[0-9a-f]{40}$/i.test(source.commit)) return false;
  if (source.path && !source.commit) return false;
  return Boolean(source.path || source.commit || source.url);
}

function validNode(node) {
  return isRecord(node) &&
    isText(node.id, 256) && isText(node.title, 1_000) && isText(node.summary) &&
    KINDS.has(node.kind) && STATUSES.has(node.status) && isDate(node.eventAt) &&
    isOptionalDate(node.assertedAt) && validLensList(node.lenses) &&
    isText(node.question) && isText(node.evidence) && isText(node.decision) &&
    isText(node.consequence) && CONFIDENCE.has(node.confidence) &&
    Array.isArray(node.sources) && node.sources.length <= 50 && node.sources.every(validSource);
}

function validEdge(edge) {
  return isRecord(edge) && isText(edge.id, 256) && isText(edge.source, 256) &&
    isText(edge.target, 256) && RELATIONS.has(edge.relation) && validLensList(edge.lenses);
}

/** Return a user-facing validation error, or null when schema 0.1 is valid. */
export function repositoryAnalysisError(value) {
  if (!isRecord(value) || value.schemaVersion !== "0.1") {
    return "The file does not match RepoTrace schema 0.1.";
  }
  const repository = value.repository;
  const stats = value.stats;
  if (!isRecord(repository) || !isText(repository.name, 500) ||
      !isText(repository.branch, 500) || !/^[0-9a-f]{40}$/i.test(repository.head) ||
      !isDate(repository.analyzedAt) || !isOptionalDate(repository.firstCommitAt) ||
      !isOptionalDate(repository.lastCommitAt) || !isOptionalText(repository.path, 4_096) ||
      !isText(value.summary)) {
    return "Repository metadata is incomplete or invalid.";
  }
  if (!isRecord(stats) || !["commits", "contributors", "files", "decisions", "corrections"]
    .every((key) => isCount(stats[key]))) {
    return "Repository statistics are incomplete or invalid.";
  }
  if (!Array.isArray(value.nodes) || !Array.isArray(value.edges)) {
    return "Nodes and edges must be arrays.";
  }
  if (value.nodes.length > MAX_ANALYSIS_NODES || value.edges.length > MAX_ANALYSIS_EDGES) {
    return `Analysis exceeds the ${MAX_ANALYSIS_NODES}-node or ${MAX_ANALYSIS_EDGES}-edge safety limit.`;
  }
  if (!value.nodes.every(validNode)) return "One or more provenance nodes are invalid.";
  if (!value.edges.every(validEdge)) return "One or more provenance edges are invalid.";

  const nodeIds = new Set(value.nodes.map((node) => node.id));
  const edgeIds = new Set(value.edges.map((edge) => edge.id));
  if (nodeIds.size !== value.nodes.length) return "Provenance node IDs must be unique.";
  if (edgeIds.size !== value.edges.length) return "Provenance edge IDs must be unique.";
  const broken = value.edges.find((edge) => !nodeIds.has(edge.source) || !nodeIds.has(edge.target));
  if (broken) return `Edge ${broken.id} refers to a missing node.`;
  return null;
}
