"use client";

import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from "@xyflow/react";
import {
  ArrowLeft,
  BookOpen,
  ChevronRight,
  CircleDot,
  FileJson,
  GitCommitHorizontal,
  History,
  ListTree,
  RotateCcw,
  Route,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isles26Analysis } from "../fixtures/isles26";
import { ProvenanceNodeCard, type ProvenanceFlowNode } from "./ProvenanceNodeCard";
import { layoutGraph } from "../lib/layoutGraph";
import {
  formatEventDate,
  isVisibleAtHistorySetting,
  lensLabels,
  MAX_ANALYSIS_FILE_BYTES,
  projectGraph,
  relationLabels,
  repositoryAnalysisError,
  type Lens,
  type ProvenanceNode,
  type RepositoryAnalysis,
} from "../lib/provenance";

const nodeTypes = { provenance: ProvenanceNodeCard };
const lenses: Lens[] = ["story", "decisions", "pipeline"];
const detailTabs = ["why", "how", "proof", "history"] as const;
type DetailTab = (typeof detailTabs)[number];

function WhyPanel({ item }: { item: ProvenanceNode }) {
  return (
    <div className="detail-tab-panel" id="detail-panel-why" role="tabpanel">
      <div className="evidence-sections">
        <section>
          <h3>Question</h3>
          <p>{item.question}</p>
        </section>
        <section>
          <h3>Evidence</h3>
          <p>{item.evidence}</p>
        </section>
        <section>
          <h3>Decision or operation</h3>
          <p>{item.decision}</p>
        </section>
        <section>
          <h3>Consequence</h3>
          <p>{item.consequence}</p>
        </section>
      </div>
      {item.alternatives && item.alternatives.length > 0 && (
        <section className="alternative-list" aria-labelledby="alternatives-heading">
          <h3 id="alternatives-heading">Paths and outcomes</h3>
          <ul>
            {item.alternatives.map((alternative) => (
              <li key={`${alternative.label}-${alternative.outcome}`}>
                <div>
                  <strong>{alternative.label}</strong>
                  <span className={`outcome outcome-${alternative.outcome}`}>
                    {alternative.outcome}
                  </span>
                </div>
                <p>{alternative.rationale}</p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function HowPanel({ item }: { item: ProvenanceNode }) {
  const details = item.implementation ?? [
    { label: "Operation or decision", value: item.decision },
    { label: "Resulting contract", value: item.consequence },
  ];
  return (
    <div className="detail-tab-panel" id="detail-panel-how" role="tabpanel">
      <dl className="implementation-list">
        {details.map((detail) => (
          <div key={`${detail.label}-${detail.value}`}>
            <dt>{detail.label}</dt>
            <dd>{detail.value}</dd>
            {detail.code && <code>{detail.code}</code>}
          </div>
        ))}
      </dl>
    </div>
  );
}

function ProofPanel({ item }: { item: ProvenanceNode }) {
  return (
    <div className="detail-tab-panel" id="detail-panel-proof" role="tabpanel">
      <section className="source-list source-list-tab" aria-labelledby="source-heading">
        <h3 id="source-heading">
          <ShieldCheck size={15} aria-hidden="true" />
          Evidence trace
        </h3>
        <ul>
          {item.sources.map((source, index) => (
            <li key={`${source.label}-${index}`}>
              <ChevronRight size={14} aria-hidden="true" />
              <div>
                <div className="source-title-row">
                  <strong>{source.label}</strong>
                  {source.kind && <span className="source-type">{source.kind}</span>}
                  {source.verification && (
                    <span className="source-verification">{source.verification}</span>
                  )}
                </div>
                {source.path && <code>{source.path}</code>}
                <div className="source-location">
                  {source.commit && (
                    <span title={source.commit}>commit {source.commit.slice(0, 12)}</span>
                  )}
                  {source.lines && <span>lines {source.lines}</span>}
                </div>
                {source.excerpt && (
                  <pre className="source-excerpt"><code>{source.excerpt}</code></pre>
                )}
                {source.url && /^https?:\/\//i.test(source.url) && (
                  <a href={source.url} target="_blank" rel="noreferrer">
                    Open source
                  </a>
                )}
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function HistoryPanel({ item }: { item: ProvenanceNode }) {
  const entries = item.history?.length
    ? item.history
    : [
        {
          at: item.eventAt,
          label: "Event recorded",
          summary: item.summary,
          status: item.status,
        },
        ...(item.assertedAt && item.assertedAt !== item.eventAt
          ? [{
              at: item.assertedAt,
              label: "Interpretation asserted",
              summary: item.decision,
              status: item.status,
            }]
          : []),
      ];
  return (
    <div className="detail-tab-panel" id="detail-panel-history" role="tabpanel">
      <ol className="history-list">
        {entries.map((entry, index) => (
          <li key={`${entry.at}-${entry.label}-${index}`}>
            <div className="history-marker" aria-hidden="true" />
            <div>
              <div className="history-entry-heading">
                <strong>{entry.label}</strong>
                <time dateTime={entry.at}>{formatEventDate(entry.at)}</time>
              </div>
              <p>{entry.summary}</p>
              {entry.status && <span className={`outcome outcome-${entry.status}`}>{entry.status}</span>}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Explorer() {
  const [analysis, setAnalysis] = useState<RepositoryAnalysis>(isles26Analysis);
  const [lens, setLens] = useState<Lens>("story");
  const [showHistory, setShowHistory] = useState(true);
  const [selectedId, setSelectedId] = useState("official-metrics");
  const [scopePath, setScopePath] = useState<string[]>([]);
  const [detailTab, setDetailTab] = useState<DetailTab>("why");
  const [nodes, setNodes, onNodesChange] = useNodesState<ProvenanceFlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [loadError, setLoadError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const { fitView } = useReactFlow();

  const nodesById = useMemo(
    () => new Map(analysis.nodes.map((node) => [node.id, node])),
    [analysis.nodes],
  );
  const scopeId = scopePath.at(-1);
  const scopeParent = scopeId ? nodesById.get(scopeId) : undefined;
  const projection = useMemo(
    () => projectGraph(analysis, lens, scopeId, showHistory),
    [analysis, lens, scopeId, showHistory],
  );
  const graphLens = projection.lens;
  const visibleItems = projection.nodes;

  const visibleIds = useMemo(
    () => new Set(visibleItems.map((item) => item.id)),
    [visibleItems],
  );

  const visibleRelations = projection.edges;

  useEffect(() => {
    let cancelled = false;
    layoutGraph(visibleItems, visibleRelations, graphLens)
      .then((graph) => {
        if (cancelled) return;
        setNodes(graph.nodes);
        setEdges(graph.edges);
        setSelectedId((current) =>
          visibleIds.has(current) ? current : (visibleItems.at(-1)?.id ?? ""),
        );
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(() => {
            void fitView({ padding: 0.22, maxZoom: 1, duration: 220 });
          });
        });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setLoadError(
          error instanceof Error
            ? `Could not lay out the graph: ${error.message}`
            : "Could not lay out the graph.",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [fitView, graphLens, setEdges, setNodes, visibleIds, visibleItems, visibleRelations]);

  const selected = useMemo(
    () => analysis.nodes.find((item) => item.id === selectedId) ?? visibleItems[0],
    [analysis, selectedId, visibleItems],
  );

  const onNodeClick = useCallback((_: unknown, node: { id: string }) => {
    setSelectedId(node.id);
    setDetailTab("why");
  }, []);

  const onSelectionChange = useCallback(
    ({ nodes: selectedNodes }: { nodes: Array<{ id: string }> }) => {
      if (selectedNodes[0]) {
        setSelectedId(selectedNodes[0].id);
        setDetailTab("why");
      }
    },
    [],
  );

  const loadAnalysis = useCallback(async (file: File) => {
    setLoadError("");
    try {
      if (file.size > MAX_ANALYSIS_FILE_BYTES) {
        throw new Error("Analysis files must be 5 MB or smaller.");
      }
      const parsed: unknown = JSON.parse(await file.text());
      const validationError = repositoryAnalysisError(parsed);
      if (validationError) throw new Error(validationError);
      const validAnalysis = parsed as RepositoryAnalysis;
      setAnalysis(validAnalysis);
      setLens("story");
      setShowHistory(true);
      setScopePath([]);
      setDetailTab("why");
      setSelectedId(validAnalysis.nodes.find((node) => node.lenses.includes("story"))?.id ?? "");
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Could not read that analysis file.");
    } finally {
      if (fileInput.current) fileInput.current.value = "";
    }
  }, []);

  const resetDemo = useCallback(() => {
    setAnalysis(isles26Analysis);
    setLens("story");
    setShowHistory(true);
    setScopePath([]);
    setDetailTab("why");
    setSelectedId("official-metrics");
    setLoadError("");
  }, []);

  const isDemo = analysis.repository.name === isles26Analysis.repository.name &&
    analysis.repository.head === isles26Analysis.repository.head;

  const childrenFor = useCallback((nodeId: string) => analysis.nodes.filter(
    (node) => node.parentId === nodeId && isVisibleAtHistorySetting(node, showHistory),
  ), [analysis.nodes, showHistory]);

  const selectedChildren = useMemo(
    () => selected ? childrenFor(selected.id) : [],
    [childrenFor, selected],
  );

  const openDrilldown = useCallback((nodeId: string) => {
    const children = childrenFor(nodeId);
    if (children.length === 0) return;
    setScopePath((current) => [...current, nodeId]);
    setSelectedId(children[0].id);
    setDetailTab("why");
  }, [childrenFor]);

  const closeDrilldown = useCallback(() => {
    setScopePath((current) => {
      const parentId = current.at(-1);
      const next = current.slice(0, -1);
      if (parentId) setSelectedId(parentId);
      return next;
    });
    setDetailTab("why");
  }, []);

  const selectLens = useCallback((nextLens: Lens) => {
    setLens(nextLens);
    setScopePath([]);
    setDetailTab("why");
  }, []);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">
            <Route size={19} />
          </span>
          <div>
            <span className="eyebrow">Repository provenance</span>
            <strong>RepoTrace</strong>
          </div>
        </div>
        <div className="topbar-actions">
          {!isDemo && (
            <button className="button subtle" type="button" onClick={resetDemo}>
              <RotateCcw size={15} aria-hidden="true" />
              Reset demo
            </button>
          )}
          <button
            className="button"
            type="button"
            onClick={() => fileInput.current?.click()}
          >
            <Upload size={15} aria-hidden="true" />
            Load analysis
          </button>
          <input
            ref={fileInput}
            className="sr-only"
            type="file"
            accept="application/json,.json"
            aria-label="Load a RepoTrace analysis JSON file"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              if (file) void loadAnalysis(file);
            }}
          />
        </div>
      </header>

      <section className="repository-header" aria-labelledby="repository-title">
        <div>
          <div className="repo-kicker">
            <GitCommitHorizontal size={15} aria-hidden="true" />
            <span>{analysis.repository.branch}</span>
            <span aria-hidden="true">·</span>
            <code>{analysis.repository.head.slice(0, 12)}</code>
          </div>
          <h1 id="repository-title">{analysis.repository.name}</h1>
          <p>{analysis.summary}</p>
        </div>
        <dl className="repository-stats">
          <div>
            <dt>Commits</dt>
            <dd>{analysis.stats.commits.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Decisions</dt>
            <dd>{analysis.stats.decisions}</dd>
          </div>
          <div>
            <dt>Corrections</dt>
            <dd>{analysis.stats.corrections}</dd>
          </div>
          <div>
            <dt>Snapshot</dt>
            <dd>{formatEventDate(analysis.repository.analyzedAt)}</dd>
          </div>
        </dl>
      </section>

      {loadError && (
        <div className="load-error" role="alert">
          <FileJson size={16} aria-hidden="true" />
          {loadError}
        </div>
      )}

      <nav className="lens-bar" aria-label="Progressive disclosure level">
        <div className="lens-tabs" role="group" aria-label="Repository lens">
          {lenses.map((item, index) => (
            <button
              key={item}
              type="button"
              className={lens === item ? "lens-tab active" : "lens-tab"}
              aria-pressed={lens === item}
              onClick={() => selectLens(item)}
            >
              <span>{index + 1}</span>
              {lensLabels[item]}
            </button>
          ))}
        </div>
        {lens !== "pipeline" && (
          <button
            className={showHistory ? "history-toggle active" : "history-toggle"}
            type="button"
            aria-pressed={showHistory}
            onClick={() => setShowHistory((value) => !value)}
          >
            <History size={15} aria-hidden="true" />
            {showHistory ? "Including superseded work" : "Current truth only"}
          </button>
        )}
      </nav>

      {scopeParent && (
        <nav className="scope-bar" aria-label="Drill-down breadcrumb">
          <button className="scope-back" type="button" onClick={closeDrilldown}>
            <ArrowLeft size={14} aria-hidden="true" />
            Back to {scopePath.length > 1 ? "parent" : lensLabels[lens]}
          </button>
          <ol className="scope-breadcrumb">
            <li>
              <button type="button" onClick={() => {
                setScopePath([]);
                setSelectedId(scopePath[0] ?? "");
                setDetailTab("why");
              }}>
                {lensLabels[lens]}
              </button>
            </li>
            {scopePath.map((id, index) => {
              const item = nodesById.get(id);
              const current = index === scopePath.length - 1;
              return (
                <li key={id} aria-current={current ? "page" : undefined}>
                  <ChevronRight size={12} aria-hidden="true" />
                  {current ? (
                    <span>{item?.title ?? id}</span>
                  ) : (
                    <button type="button" onClick={() => {
                      setScopePath(scopePath.slice(0, index + 1));
                      setSelectedId(childrenFor(id)[0]?.id ?? "");
                      setDetailTab("why");
                    }}>
                      {item?.title ?? id}
                    </button>
                  )}
                </li>
              );
            })}
          </ol>
          <span className="scope-summary">{scopeParent.summary}</span>
        </nav>
      )}

      <section className="workspace" aria-label={`${lensLabels[lens]} graph and evidence`}>
        <div className="graph-panel">
          <div className="graph-caption">
            <span>
              <CircleDot size={14} aria-hidden="true" />
              {visibleItems.length} nodes · {visibleRelations.length} relationships
            </span>
            <span>{scopeId ? "Inspect why, how, proof, and history" : "Select a node to inspect its evidence"}</span>
          </div>
          <div className="flow-canvas">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              onSelectionChange={onSelectionChange}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable
              minZoom={0.2}
              maxZoom={1.7}
              fitView
              fitViewOptions={{ padding: 0.22, maxZoom: 1 }}
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={24}
                size={1}
                color="var(--graph-dot)"
              />
              <Controls showInteractive={false} position="bottom-left" />
            </ReactFlow>
            <ul className="sr-only" aria-label="Visible graph relationships">
              {visibleRelations.map((edge) => {
                const source = analysis.nodes.find((node) => node.id === edge.source);
                const target = analysis.nodes.find((node) => node.id === edge.target);
                return (
                  <li key={`accessible-${edge.id}`}>
                    {source?.title ?? edge.source} {relationLabels[edge.relation]}{" "}
                    {target?.title ?? edge.target}
                  </li>
                );
              })}
            </ul>
          </div>
        </div>

        <aside className="evidence-panel" aria-live="polite">
          {selected ? (
            <>
              <div className="evidence-heading">
                <div>
                  <span className={`status-pill status-${selected.status}`}>
                    {selected.status}
                  </span>
                  <span className="confidence">{selected.confidence}</span>
                </div>
                <time dateTime={selected.eventAt}>{formatEventDate(selected.eventAt)}</time>
              </div>
              <h2>{selected.title}</h2>
              <p className="evidence-summary">{selected.summary}</p>

              {selectedChildren.length > 0 && (
                <button
                  className="drilldown-button"
                  type="button"
                  onClick={() => openDrilldown(selected.id)}
                >
                  <ListTree size={15} aria-hidden="true" />
                  <span>
                    <strong>{selected.drilldownLabel ?? "Open details"}</strong>
                    <small>{selectedChildren.length} evidence nodes</small>
                  </span>
                  <ChevronRight size={15} aria-hidden="true" />
                </button>
              )}

              <div className="detail-tabs" role="tablist" aria-label="Evidence detail">
                {detailTabs.map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    role="tab"
                    id={`detail-tab-${tab}`}
                    aria-selected={detailTab === tab}
                    aria-controls={`detail-panel-${tab}`}
                    className={detailTab === tab ? "active" : ""}
                    onClick={() => setDetailTab(tab)}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              {detailTab === "why" && <WhyPanel item={selected} />}
              {detailTab === "how" && <HowPanel item={selected} />}
              {detailTab === "proof" && <ProofPanel item={selected} />}
              {detailTab === "history" && <HistoryPanel item={selected} />}
            </>
          ) : (
            <div className="empty-evidence">
              <BookOpen size={22} aria-hidden="true" />
              <p>No evidence node is selected.</p>
            </div>
          )}
        </aside>
      </section>

      <footer>
        <span>Read-only prototype · repository content is treated as untrusted evidence</span>
        <span>Schema {analysis.schemaVersion}</span>
      </footer>
    </main>
  );
}

export function ProvenanceExplorer() {
  return (
    <ReactFlowProvider>
      <Explorer />
    </ReactFlowProvider>
  );
}
