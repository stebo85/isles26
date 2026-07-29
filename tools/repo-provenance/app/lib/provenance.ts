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

export interface SourceReference {
  label: string;
  path?: string;
  commit?: string;
  lines?: string;
  url?: string;
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
  schemaVersion: "0.1";
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
