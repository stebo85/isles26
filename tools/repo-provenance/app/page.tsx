import type { Metadata } from "next";
import { ProvenanceExplorer } from "./components/ProvenanceExplorer";

export const metadata: Metadata = {
  title: "RepoTrace · Repository provenance explorer",
  description:
    "Trace a repository from evidence through decisions to its current pipeline.",
};

export default function Home() {
  return <ProvenanceExplorer />;
}
