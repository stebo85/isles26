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
const SOURCE_KINDS = new Set(["code", "document", "report", "test", "commit", "url"]);
const VERIFICATION = new Set(["documented", "static-analysis", "runtime-observed", "human-reviewed"]);
const ALTERNATIVE_OUTCOMES = new Set(["selected", "retained", "rejected", "superseded", "pending"]);
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
      !isOptionalText(source.lines, 128) || !isOptionalText(source.url, 4_096) ||
      !isOptionalText(source.excerpt, 20_000)) return false;
  if (source.commit !== undefined && !/^[0-9a-f]{40}$/i.test(source.commit)) return false;
  if (source.kind !== undefined && !SOURCE_KINDS.has(source.kind)) return false;
  if (source.verification !== undefined && !VERIFICATION.has(source.verification)) return false;
  if (source.path && !source.commit) return false;
  return Boolean(source.path || source.commit || source.url);
}

function validImplementation(item) {
  return isRecord(item) && isText(item.label, 500) && isText(item.value) &&
    isOptionalText(item.code, 4_096);
}

function validAlternative(item) {
  return isRecord(item) && isText(item.label, 500) &&
    ALTERNATIVE_OUTCOMES.has(item.outcome) && isText(item.rationale);
}

function validHistoryEntry(item) {
  return isRecord(item) && isDate(item.at) && isText(item.label, 500) &&
    isText(item.summary) && (item.status === undefined || STATUSES.has(item.status));
}

function validNode(node) {
  return isRecord(node) &&
    isText(node.id, 256) && isText(node.title, 1_000) && isText(node.summary) &&
    KINDS.has(node.kind) && STATUSES.has(node.status) && isDate(node.eventAt) &&
    isOptionalDate(node.assertedAt) && validLensList(node.lenses) &&
    isText(node.question) && isText(node.evidence) && isText(node.decision) &&
    isText(node.consequence) && CONFIDENCE.has(node.confidence) &&
    Array.isArray(node.sources) && node.sources.length <= 50 && node.sources.every(validSource) &&
    isOptionalText(node.parentId, 256) && isOptionalText(node.drilldownLabel, 500) &&
    (node.drilldownLens === undefined || LENSES.has(node.drilldownLens)) &&
    (node.implementation === undefined || (Array.isArray(node.implementation) &&
      node.implementation.length <= 100 && node.implementation.every(validImplementation))) &&
    (node.alternatives === undefined || (Array.isArray(node.alternatives) &&
      node.alternatives.length <= 50 && node.alternatives.every(validAlternative))) &&
    (node.history === undefined || (Array.isArray(node.history) &&
      node.history.length <= 100 && node.history.every(validHistoryEntry)));
}

function validEdge(edge) {
  return isRecord(edge) && isText(edge.id, 256) && isText(edge.source, 256) &&
    isText(edge.target, 256) && RELATIONS.has(edge.relation) && validLensList(edge.lenses);
}

/** Return a user-facing validation error, or null when a supported schema is valid. */
export function repositoryAnalysisError(value) {
  if (!isRecord(value) || !["0.1", "0.2"].includes(value.schemaVersion)) {
    return "The file does not match a supported RepoTrace schema (0.1 or 0.2).";
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
  const missingParent = value.nodes.find((node) => node.parentId && !nodeIds.has(node.parentId));
  if (missingParent) return `Node ${missingParent.id} refers to a missing parent.`;
  const nodesById = new Map(value.nodes.map((node) => [node.id, node]));
  for (const node of value.nodes) {
    const ancestors = new Set([node.id]);
    let parentId = node.parentId;
    while (parentId) {
      if (ancestors.has(parentId)) return `Node ${node.id} has a cyclic parent chain.`;
      ancestors.add(parentId);
      parentId = nodesById.get(parentId)?.parentId;
    }
  }
  return null;
}
