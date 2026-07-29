import {
  MarkerType,
  Position,
  type Edge,
} from "@xyflow/react";
import type {
  Lens,
  ProvenanceEdge,
  ProvenanceNode,
} from "./provenance";
import { relationLabels } from "./provenance";
import type {
  FlowDirection,
  ProvenanceFlowNode,
} from "../components/ProvenanceNodeCard";

export interface LaidOutGraph {
  nodes: ProvenanceFlowNode[];
  edges: Edge[];
}

export async function layoutGraph(
  items: ProvenanceNode[],
  relations: ProvenanceEdge[],
  lens: Lens,
): Promise<LaidOutGraph> {
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  const elk = new ELK();
  const direction: FlowDirection = lens === "pipeline" ? "RIGHT" : "DOWN";
  const horizontal = direction === "RIGHT";
  const width = lens === "pipeline" ? 260 : 286;
  const height = lens === "pipeline" ? 142 : 154;

  const graph = await elk.layout({
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": direction,
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.spacing.nodeNodeBetweenLayers": lens === "pipeline" ? "76" : "86",
      "elk.spacing.nodeNode": "44",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    },
    children: items.map((item) => ({
      id: item.id,
      width,
      height,
    })),
    edges: relations.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  });

  const children = new Map((graph.children ?? []).map((child) => [child.id, child]));
  const titles = new Map(items.map((item) => [item.id, item.title]));
  const nodes: ProvenanceFlowNode[] = items.map((item) => {
    const position = children.get(item.id);
    return {
      id: item.id,
      type: "provenance",
      position: { x: position?.x ?? 0, y: position?.y ?? 0 },
      sourcePosition: horizontal ? Position.Right : Position.Bottom,
      targetPosition: horizontal ? Position.Left : Position.Top,
      data: { item, direction },
      ariaLabel: `${item.kind}: ${item.title}`,
      style: { width, height },
    };
  });

  const edges: Edge[] = relations.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: relationLabels[edge.relation],
    ariaLabel: `${titles.get(edge.source) ?? edge.source} ${relationLabels[edge.relation]} ${titles.get(edge.target) ?? edge.target}`,
    type: "smoothstep",
    markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15 },
    className: `relation-${edge.relation}`,
    labelBgPadding: [6, 4],
    labelBgBorderRadius: 4,
  }));

  return { nodes, edges };
}
