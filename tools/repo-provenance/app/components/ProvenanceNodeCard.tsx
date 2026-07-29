"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpFromLine,
  Box,
  CheckCircle2,
  CircleHelp,
  FileCheck2,
  GitBranch,
  Lightbulb,
  RefreshCcw,
  SearchCheck,
} from "lucide-react";
import type { ProvenanceNode } from "../lib/provenance";
import { formatEventDate } from "../lib/provenance";

export type FlowDirection = "DOWN" | "RIGHT";

export type ProvenanceFlowNode = Node<
  { item: ProvenanceNode; direction: FlowDirection },
  "provenance"
>;

const icons = {
  question: CircleHelp,
  evidence: SearchCheck,
  decision: Lightbulb,
  result: CheckCircle2,
  correction: RefreshCcw,
  artifact: Box,
  "pipeline-step": GitBranch,
  input: ArrowDownToLine,
  output: ArrowUpFromLine,
  proof: FileCheck2,
  blocker: AlertTriangle,
} as const;

export function ProvenanceNodeCard({ data, selected }: NodeProps<ProvenanceFlowNode>) {
  const { item, direction } = data;
  const Icon = icons[item.kind];
  const horizontal = direction === "RIGHT";

  return (
    <article
      className={`provenance-node kind-${item.kind} status-${item.status}${selected ? " is-selected" : ""}`}
      aria-label={`${item.kind}: ${item.title}`}
    >
      <Handle
        type="target"
        position={horizontal ? Position.Left : Position.Top}
        className="provenance-handle"
      />
      <div className="node-meta">
        <span className="node-kind">
          <Icon aria-hidden="true" size={14} strokeWidth={1.8} />
          {item.kind.replace("-", " ")}
        </span>
        <time dateTime={item.eventAt}>{formatEventDate(item.eventAt)}</time>
      </div>
      <h3>{item.title}</h3>
      <p>{item.summary}</p>
      <div className="node-status">
        <span>{item.status}</span>
        <span>{item.drilldownLabel ? "drill down" : item.confidence}</span>
      </div>
      <Handle
        type="source"
        position={horizontal ? Position.Right : Position.Bottom}
        className="provenance-handle"
      />
    </article>
  );
}
