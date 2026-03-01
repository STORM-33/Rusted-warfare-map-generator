"use client";

import { useState } from "react";
import { usePyodide } from "@/hooks/usePyodide";
import { WizardApp } from "@/components/WizardApp";
import { QuickGeneratePage } from "@/components/QuickGeneratePage";
import { LoadingScreen } from "@/components/LoadingScreen";

type AppMode = "wizard" | "quick";

export default function Home() {
  const [mode, setMode] = useState<AppMode>("wizard");
  const pyodide = usePyodide();

  if (pyodide.loading && !pyodide.ready) {
    return (
      <LoadingScreen
        stage={pyodide.loadingStage}
        progress={pyodide.loadingProgress}
      />
    );
  }

  return (
    <div className="app-shell">
      <nav className="mode-toggle">
        <button
          type="button"
          className={`mode-btn ${mode === "wizard" ? "active" : ""}`}
          onClick={() => setMode("wizard")}
        >
          Wizard
        </button>
        <button
          type="button"
          className={`mode-btn ${mode === "quick" ? "active" : ""}`}
          onClick={() => setMode("quick")}
        >
          Quick Generate
        </button>
      </nav>
      {mode === "wizard" ? (
        <WizardApp pyodide={pyodide} />
      ) : (
        <QuickGeneratePage pyodide={pyodide} />
      )}
    </div>
  );
}
