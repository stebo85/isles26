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
  BookOpen,
  ChevronRight,
  CircleDot,
  FileJson,
  GitCommitHorizontal,
  History,
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
  lensLabels,
  MAX_ANALYSIS_FILE_BYTES,
  relationLabels,
  repositoryAnalysisError,
  type Lens,
  type RepositoryAnalysis,
} from "../lib/provenance";

const nodeTypes = { provenance: ProvenanceNodeCard };
const lenses: Lens[] = ["story", "decisions", "pipeline"];

function Explorer() {
  const [analysis, setAnalysis] = useState<RepositoryAnalysis>(isles26Analysis);
  const [lens, setLens] = useState<Lens>("story");
  const [showHistory, setShowHistory] = useState(true);
  const [selectedId, setSelectedId] = useState("official-metrics");
  const [nodes, setNodes, onNodesChange] = useNodesState<ProvenanceFlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [loadError, setLoadError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const { fitView } = useReactFlow();

  const visibleItems = useMemo(() => {
    return analysis.nodes.filter((node) => {
      if (!node.lenses.includes(lens)) return false;
      if (showHistory) return true;
      return !["superseded", "retracted", "negative"].includes(node.status);
    });
  }, [analysis, lens, showHistory]);

  const visibleIds = useMemo(
    () => new Set(visibleItems.map((item) => item.id)),
    [visibleItems],
  );

  const visibleRelations = useMemo(
    () =>
      analysis.edges.filter(
        (edge) =>
          edge.lenses.includes(lens) &&
          visibleIds.has(edge.source) &&
          visibleIds.has(edge.target),
      ),
    [analysis, lens, visibleIds],
  );

  useEffect(() => {
    let cancelled = false;
    layoutGraph(visibleItems, visibleRelations, lens)
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
  }, [fitView, lens, setEdges, setNodes, visibleIds, visibleItems, visibleRelations]);

  const selected = useMemo(
    () => analysis.nodes.find((item) => item.id === selectedId) ?? visibleItems[0],
    [analysis, selectedId, visibleItems],
  );

  const onNodeClick = useCallback((_: unknown, node: { id: string }) => {
    setSelectedId(node.id);
  }, []);

  const onSelectionChange = useCallback(
    ({ nodes: selectedNodes }: { nodes: Array<{ id: string }> }) => {
      if (selectedNodes[0]) setSelectedId(selectedNodes[0].id);
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
    setSelectedId("official-metrics");
    setLoadError("");
  }, []);

  const isDemo = analysis.repository.name === isles26Analysis.repository.name &&
    analysis.repository.head === isles26Analysis.repository.head;

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
              onClick={() => setLens(item)}
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

      <section className="workspace" aria-label={`${lensLabels[lens]} graph and evidence`}>
        <div className="graph-panel">
          <div className="graph-caption">
            <span>
              <CircleDot size={14} aria-hidden="true" />
              {visibleItems.length} nodes · {visibleRelations.length} relationships
            </span>
            <span>Select a node to inspect its evidence</span>
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

              <div className="evidence-sections">
                <section>
                  <h3>Question</h3>
                  <p>{selected.question}</p>
                </section>
                <section>
                  <h3>Evidence</h3>
                  <p>{selected.evidence}</p>
                </section>
                <section>
                  <h3>Decision or operation</h3>
                  <p>{selected.decision}</p>
                </section>
                <section>
                  <h3>Consequence</h3>
                  <p>{selected.consequence}</p>
                </section>
              </div>

              <section className="source-list" aria-labelledby="source-heading">
                <h3 id="source-heading">
                  <ShieldCheck size={15} aria-hidden="true" />
                  Evidence trace
                </h3>
                <ul>
                  {selected.sources.map((source, index) => (
                    <li key={`${source.label}-${index}`}>
                      <ChevronRight size={14} aria-hidden="true" />
                      <div>
                        <strong>{source.label}</strong>
                        {source.path && <code>{source.path}</code>}
                        {source.commit && (
                          <span title={source.commit}>commit {source.commit.slice(0, 12)}</span>
                        )}
                        {source.lines && <span>lines {source.lines}</span>}
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
        <span>Schema 0.1</span>
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
