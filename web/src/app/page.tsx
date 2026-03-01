"use client";

import { useState } from "react";
import { usePyodide } from "@/hooks/usePyodide";
import { WizardApp } from "@/components/WizardApp";
import { QuickGeneratePage } from "@/components/QuickGeneratePage";
import { LoadingScreen } from "@/components/LoadingScreen";

type AppMode = "wizard" | "quick";

export default function Home() {
  const [mode, setMode] = useState<AppMode>("wizard");
  const wizardPyodide = usePyodide();
  const quickPyodide = usePyodide();
  const activePyodide = mode === "wizard" ? wizardPyodide : quickPyodide;
  const showLoading = activePyodide.loading && !activePyodide.ready;

  return (
    <>
      {showLoading ? (
        <LoadingScreen
          stage={activePyodide.loadingStage}
          progress={activePyodide.loadingProgress}
        />
      ) : null}
      <div
        className="app-shell"
        style={{ display: showLoading ? "none" : "flex" }}
      >
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
          <div className="nav-links">
            <a
              href="https://github.com/STORM-33/Rusted-warfare-map-generator"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-btn github-btn"
            >
              <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              GitHub
            </a>
            <a
              href="https://ko-fi.com/storm33"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-btn kofi-btn"
            >
              <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
                <path d="M23.881 8.948c-.773-4.085-4.859-4.593-4.859-4.593H.723c-.604 0-.679.798-.679.798s-.082 7.324-.022 11.822c.164 2.424 2.586 2.672 2.586 2.672s8.267-.023 11.966-.049c2.438-.426 2.683-2.566 2.658-3.734 4.352.24 7.422-2.831 6.649-6.916zm-11.062 3.511c-1.246 1.453-4.011 3.976-4.011 3.976s-.121.119-.31.023c-.076-.057-.108-.09-.108-.09-.443-.441-3.368-3.049-4.034-3.954-.709-.965-1.041-2.7-.091-3.71.951-1.01 3.005-1.086 4.363.407 0 0 1.565-1.782 3.468-.963 1.904.82 1.832 3.011.723 4.311zm6.173.478c-.928.116-1.682.028-1.682.028V7.284h1.77s1.971.551 1.971 2.638c0 1.913-.985 2.667-2.059 3.015z" />
              </svg>
              Support
            </a>
          </div>
        </nav>
        <div style={{ display: mode === "wizard" ? "flex" : "none", flex: 1, minHeight: 0 }}>
          <WizardApp pyodide={wizardPyodide} />
        </div>
        <div style={{ display: mode === "quick" ? "flex" : "none", flex: 1, minHeight: 0 }}>
          <QuickGeneratePage pyodide={quickPyodide} />
        </div>
      </div>
    </>
  );
}
