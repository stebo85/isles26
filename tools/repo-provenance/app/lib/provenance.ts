import { repositoryAnalysisError } from "./validate-analysis.mjs";

export { MAX_ANALYSIS_FILE_BYTES, repositoryAnalysisError } from "./validate-analysis.mjs";

export type Lens = "story" | "decisions" | "pipeline";

export type ProvenanceStatus =
  | "current"
  | "superseded"
  | "retracted"
  | "negative"
  | "uncertain";

export type ProvenanceKind =
  | "question"
  | "evidence"
  | "decision"
  | "result"
  | "correction"
  | "artifact"
  | "pipeline-step"
  | "input"
  | "output"
  | "proof"
  | "blocker";

export type VerificationBasis =
  | "documented"
  | "static-analysis"
  | "runtime-observed"
  | "human-reviewed";

export type SourceKind =
  | "code"
  | "document"
  | "report"
  | "test"
  | "commit"
  | "url";

export interface SourceReference {
  label: string;
  path?: string;
  commit?: string;
  lines?: string;
  url?: string;
  excerpt?: string;
  kind?: SourceKind;
  verification?: VerificationBasis;
}

export interface ImplementationDetail {
  label: string;
  value: string;
  code?: string;
}

export interface DecisionAlternative {
  label: string;
  outcome: "selected" | "retained" | "rejected" | "superseded" | "pending";
  rationale: string;
}

export interface ProvenanceHistoryEntry {
  at: string;
  label: string;
  summary: string;
  status?: ProvenanceStatus;
}

export interface ProvenanceNode {
  id: string;
  title: string;
  summary: string;
  kind: ProvenanceKind;
  status: ProvenanceStatus;
  eventAt: string;
  assertedAt?: string;
  lenses: Lens[];
  question: string;
  evidence: string;
  decision: string;
  consequence: string;
  confidence: "explicit" | "strongly-inferred" | "speculative";
  sources: SourceReference[];
  parentId?: string;
  drilldownLabel?: string;
  drilldownLens?: Lens;
  implementation?: ImplementationDetail[];
  alternatives?: DecisionAlternative[];
  history?: ProvenanceHistoryEntry[];
}

export interface ProvenanceEdge {
  id: string;
  source: string;
  target: string;
  relation:
    | "motivated"
    | "measured"
    | "supports"
    | "contradicts"
    | "selects"
    | "supersedes"
    | "produces"
    | "consumes"
    | "feeds"
    | "precedes"
    | "guards"
    | "blocks";
  lenses: Lens[];
}

export interface RepositoryAnalysis {
  schemaVersion: "0.1" | "0.2";
  repository: {
    name: string;
    path?: string;
    branch: string;
    head: string;
    analyzedAt: string;
    firstCommitAt?: string;
    lastCommitAt?: string;
  };
  summary: string;
  stats: {
    commits: number;
    contributors: number;
    files: number;
    decisions: number;
    corrections: number;
  };
  nodes: ProvenanceNode[];
  edges: ProvenanceEdge[];
}

export const lensLabels: Record<Lens, string> = {
  story: "Story",
  decisions: "Decisions",
  pipeline: "Current pipeline",
};

export const relationLabels: Record<ProvenanceEdge["relation"], string> = {
  motivated: "motivated",
  measured: "measured",
  supports: "supports",
  contradicts: "contradicts",
  selects: "selects",
  supersedes: "supersedes",
  produces: "produces",
  consumes: "consumes",
  feeds: "feeds",
  precedes: "precedes",
  guards: "guards",
  blocks: "blocks",
};

export function isRepositoryAnalysis(value: unknown): value is RepositoryAnalysis {
  return repositoryAnalysisError(value) === null;
}

export function isVisibleAtHistorySetting(
  node: ProvenanceNode,
  includeHistory: boolean,
): boolean {
  return includeHistory || !["superseded", "retracted", "negative"].includes(node.status);
}

export function projectGraph(
  analysis: RepositoryAnalysis,
  lens: Lens,
  scopeId: string | undefined,
  includeHistory: boolean,
): { lens: Lens; nodes: ProvenanceNode[]; edges: ProvenanceEdge[] } {
  const scopeParent = scopeId
    ? analysis.nodes.find((node) => node.id === scopeId)
    : undefined;
  const nodes = analysis.nodes.filter((node) => {
    if (scopeId ? node.parentId !== scopeId : node.parentId) return false;
    if (!scopeId && !node.lenses.includes(lens)) return false;
    return isVisibleAtHistorySetting(node, includeHistory);
  });
  const nodeIds = new Set(nodes.map((node) => node.id));
  const internalEdges = analysis.edges.filter(
    (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target),
  );
  const projectionLens = scopeParent
    ? scopeParent.drilldownLens ?? (["story", "decisions", "pipeline"] as Lens[])
      .reduce((best, candidate) => {
        const score = internalEdges.filter((edge) => edge.lenses.includes(candidate)).length * 1_000 +
          nodes.filter((node) => node.lenses.includes(candidate)).length;
        const bestScore = internalEdges.filter((edge) => edge.lenses.includes(best)).length * 1_000 +
          nodes.filter((node) => node.lenses.includes(best)).length;
        return score > bestScore ? candidate : best;
      }, lens)
    : lens;
  const edges = analysis.edges.filter(
    (edge) => edge.lenses.includes(projectionLens) &&
      nodeIds.has(edge.source) && nodeIds.has(edge.target),
  );
  return { lens: projectionLens, nodes, edges };
}

export function nodeDate(node: ProvenanceNode): Date {
  const parsed = new Date(node.eventAt);
  return Number.isNaN(parsed.getTime()) ? new Date(0) : parsed;
}

export function formatEventDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
    year: "numeric",
  }).format(date);
}
