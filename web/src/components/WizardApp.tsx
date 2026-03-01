"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePyodide } from "@/hooks/usePyodide";
import { useWizardState } from "@/hooks/useWizardState";
import type { RenderMode, WorkerAction, WorkerStepCompleteMessage } from "@/lib/types";
import type { ExtractedTilesets } from "@/lib/tilesetExtractor";
import { loadTilesetsFromBlueprint } from "@/lib/tilesetExtractor";
import { LoadingScreen } from "@/components/LoadingScreen";
import { StepBar } from "@/components/StepBar";
import { MapCanvas } from "@/components/MapCanvas";
import { CoastlineStep } from "@/components/steps/CoastlineStep";
import { HillDrawingStep } from "@/components/steps/HillDrawingStep";
import { HeightOceanStep } from "@/components/steps/HeightOceanStep";
import { CommandCenterStep } from "@/components/steps/CommandCenterStep";
import { ResourceStep } from "@/components/steps/ResourceStep";
import { FinalizeStep } from "@/components/steps/FinalizeStep";

type MirroringMode =
  | "none"
  | "horizontal"
  | "vertical"
  | "diagonal1"
  | "diagonal2"
  | "both";

type RenderPreference = RenderMode | "auto";

const DEFAULT_GRID = [
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0],
];

const clampValue = (value: number, min: number, max: number) =>
  Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : min;

const getMirroredCells = (
  row: number,
  col: number,
  size: number,
  mode: MirroringMode,
) => {
  const cells = new Set<string>();
  const addCell = (r: number, c: number) => {
    if (r >= 0 && c >= 0 && r < size && c < size) {
      cells.add(`${r}:${c}`);
    }
  };

  addCell(row, col);
  if (mode === "horizontal" || mode === "both") {
    addCell(size - 1 - row, col);
  }
  if (mode === "vertical" || mode === "both") {
    addCell(row, size - 1 - col);
  }
  if (mode === "diagonal1") {
    addCell(col, row);
  }
  if (mode === "diagonal2") {
    addCell(size - 1 - col, size - 1 - row);
  }
  if (mode === "both") {
    addCell(size - 1 - row, size - 1 - col);
  }

  return Array.from(cells).map((entry) => {
    const [r, c] = entry.split(":").map(Number);
    return [r, c] as [number, number];
  });
};

const downloadFile = (bytes: Uint8Array, fileName: string) => {
  const copied = Uint8Array.from(bytes);
  const blob = new Blob([copied.buffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
};

export function WizardApp() {
  const {
    loading,
    ready,
    error,
    loadingStage,
    loadingProgress,
    snapshot,
    callAction,
  } = usePyodide();
  const {
    steps,
    currentStep,
    completedStep,
    goToStep,
    nextStep,
    prevStep,
    markStepComplete,
    isStepAccessible,
  } = useWizardState();

  const [grid, setGrid] = useState<number[][]>(DEFAULT_GRID);
  const [height, setHeight] = useState(160);
  const [width, setWidth] = useState(160);
  const [mirroring, setMirroring] = useState<MirroringMode>("vertical");
  const [tileset, setTileset] = useState(1);
  const [heightLevels, setHeightLevels] = useState(7);
  const [oceanLevels, setOceanLevels] = useState(3);
  const [numPlayers, setNumPlayers] = useState(4);
  const [numResources, setNumResources] = useState(12);
  const [drawValue, setDrawValue] = useState<1 | 2>(1);
  const [brushSize, setBrushSize] = useState(3);
  const [ccManual, setCcManual] = useState(true);
  const [resourceManual, setResourceManual] = useState(true);
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState("Waiting for runtime...");
  const [tilesets, setTilesets] = useState<ExtractedTilesets>();
  const [renderPreference, setRenderPreference] =
    useState<RenderPreference>("auto");
  const [renderMode, setRenderMode] = useState<RenderMode>("sampled");

  const blueprintCacheRef = useRef<Record<number, string>>({});

  useEffect(() => {
    if (!ready) {
      return;
    }
    setStatusText("Runtime ready");
    void callAction("get_state_snapshot").catch((runtimeError) => {
      setStatusText(
        `Snapshot error: ${runtimeError instanceof Error ? runtimeError.message : "unknown"}`,
      );
    });
  }, [ready, callAction]);

  useEffect(() => {
    let cancelled = false;
    loadTilesetsFromBlueprint(`/blueprints/generator_blueprint${tileset}.tmx`)
      .then((result) => {
        if (!cancelled) {
          setTilesets(result);
        }
      })
      .catch((tilesetError) => {
        if (!cancelled) {
          setStatusText(
            `Tileset load error: ${tilesetError instanceof Error ? tilesetError.message : "unknown"}`,
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [tileset]);

  const runAction = useCallback(
    async (
      label: string,
      action: Exclude<WorkerAction, "init">,
      params?: Record<string, unknown>,
    ) => {
      setBusy(true);
      setStatusText(label);
      try {
        const response = await callAction(action, params);
        setStatusText(`${label} complete`);
        return response;
      } catch (actionError) {
        const message =
          actionError instanceof Error ? actionError.message : "unknown";
        setStatusText(`${label} failed: ${message}`);
        throw actionError;
      } finally {
        setBusy(false);
      }
    },
    [callAction],
  );

  const getBlueprintXml = useCallback(async (pattern: number) => {
    const cached = blueprintCacheRef.current[pattern];
    if (cached) {
      return cached;
    }
    const response = await fetch(`/blueprints/generator_blueprint${pattern}.tmx`);
    if (!response.ok) {
      throw new Error(`Blueprint ${pattern} not found`);
    }
    const xml = await response.text();
    blueprintCacheRef.current[pattern] = xml;
    return xml;
  }, []);

  const handleToggleGridCell = useCallback(
    (row: number, col: number) => {
      setGrid((previous) => {
        const size = previous.length;
        const next = previous.map((line) => [...line]);
        const newValue = next[row][col] ? 0 : 1;
        for (const [mirrorRow, mirrorCol] of getMirroredCells(
          row,
          col,
          size,
          mirroring,
        )) {
          next[mirrorRow][mirrorCol] = newValue;
        }
        return next;
      });
    },
    [mirroring],
  );

  const handleGenerateCoastline = useCallback(async () => {
    await runAction("Generating coastline", "run_coastline", {
      grid,
      height: clampValue(height, 40, 640),
      width: clampValue(width, 40, 640),
      mirroring,
      tileset,
      heightLevels,
      oceanLevels,
      numPlayers,
      numResources,
    });
    markStepComplete(0);
    goToStep(1);
  }, [
    goToStep,
    grid,
    height,
    heightLevels,
    markStepComplete,
    mirroring,
    numPlayers,
    numResources,
    oceanLevels,
    runAction,
    tileset,
    width,
  ]);

  const handleDrawWalls = useCallback(
    (points: [number, number][], value: 0 | 1 | 2) => {
      void callAction("draw_walls", { points, value, brush_size: brushSize }).catch((drawError) => {
        setStatusText(
          `Wall draw error: ${drawError instanceof Error ? drawError.message : "unknown"}`,
        );
      });
    },
    [brushSize, callAction],
  );

  const handleCanvasClick = useCallback(
    (row: number, col: number) => {
      if (currentStep === 3 && ccManual) {
        void callAction("place_cc_manual", { row, col })
          .then(() => {
            markStepComplete(3);
            setStatusText("Command center updated");
          })
          .catch((clickError) => {
            setStatusText(
              `CC placement failed: ${clickError instanceof Error ? clickError.message : "unknown"}`,
            );
          });
      } else if (currentStep === 4 && resourceManual) {
        void callAction("place_resource_manual", { row, col })
          .then(() => {
            markStepComplete(4);
            setStatusText("Resource placement updated");
          })
          .catch((clickError) => {
            setStatusText(
              `Resource placement failed: ${clickError instanceof Error ? clickError.message : "unknown"}`,
            );
          });
      }
    },
    [callAction, ccManual, currentStep, markStepComplete, resourceManual],
  );

  const handleFinalize = useCallback(async () => {
    const blueprintXml = await getBlueprintXml(tileset);
    const response = (await runAction("Finalizing map", "run_finalize", {
      blueprintXml,
    })) as WorkerStepCompleteMessage;
    if (response.tmxBytes) {
      downloadFile(response.tmxBytes, "generated_map.tmx");
      setStatusText("generated_map.tmx downloaded");
      markStepComplete(5);
    } else {
      setStatusText("Finalize completed without output bytes");
    }
  }, [getBlueprintXml, markStepComplete, runAction, tileset]);

  const mapView = useMemo(() => {
    if (currentStep <= 1) {
      return "coastline" as const;
    }
    if (currentStep === 5 && snapshot?.matrices.id_matrix) {
      return "final" as const;
    }
    return "height" as const;
  }, [currentStep, snapshot]);

  const interactionMode = useMemo(() => {
    if (currentStep === 1) {
      return "draw" as const;
    }
    if ((currentStep === 3 && ccManual) || (currentStep === 4 && resourceManual)) {
      return "click" as const;
    }
    return "none" as const;
  }, [ccManual, currentStep, resourceManual]);

  const canAdvanceFromStep0 = Boolean(snapshot?.matrices.coastline_height_map);

  const handleNext = () => {
    if (currentStep === 0 && !canAdvanceFromStep0) {
      return;
    }
    if (currentStep === 1) {
      markStepComplete(1);
    }
    if (currentStep === 3) {
      markStepComplete(3);
    }
    if (currentStep === 4) {
      markStepComplete(4);
    }
    nextStep();
  };

  const handleStepContent = () => {
    if (currentStep === 0) {
      return (
        <CoastlineStep
          grid={grid}
          height={height}
          width={width}
          mirroring={mirroring}
          tileset={tileset}
          disabled={busy || !ready}
          onToggleCell={handleToggleGridCell}
          onHeightChange={(value) => setHeight(clampValue(value, 40, 640))}
          onWidthChange={(value) => setWidth(clampValue(value, 40, 640))}
          onMirroringChange={(value) => setMirroring(value as MirroringMode)}
          onTilesetChange={(value) => setTileset(clampValue(value, 1, 5))}
          onGenerate={handleGenerateCoastline}
        />
      );
    }
    if (currentStep === 1) {
      return (
        <HillDrawingStep
          drawValue={drawValue}
          brushSize={brushSize}
          disabled={busy || !ready}
          onDrawValueChange={setDrawValue}
          onBrushSizeChange={setBrushSize}
          onClear={() => void runAction("Clearing walls", "clear_walls")}
        />
      );
    }
    if (currentStep === 2) {
      return (
        <HeightOceanStep
          heightLevels={heightLevels}
          oceanLevels={oceanLevels}
          disabled={busy || !ready}
          onHeightLevelsChange={(value) =>
            setHeightLevels(clampValue(value, 1, 20))
          }
          onOceanLevelsChange={(value) => setOceanLevels(clampValue(value, 1, 20))}
          onGenerate={() =>
            void runAction("Generating height/ocean", "run_height_ocean", {
              heightLevels,
              oceanLevels,
            }).then(() => markStepComplete(2))
          }
        />
      );
    }
    if (currentStep === 3) {
      return (
        <CommandCenterStep
          numPlayers={numPlayers}
          manualMode={ccManual}
          disabled={busy || !ready}
          onNumPlayersChange={(value) => setNumPlayers(clampValue(value, 2, 10))}
          onManualModeChange={setCcManual}
          onRandom={() =>
            void runAction("Placing random command centers", "place_cc_random", {
              numPlayers,
            }).then(() => markStepComplete(3))
          }
          onUndo={() => void runAction("Undo command center", "undo_cc")}
          onClear={() => void runAction("Clear command centers", "clear_cc")}
        />
      );
    }
    if (currentStep === 4) {
      return (
        <ResourceStep
          numResources={numResources}
          manualMode={resourceManual}
          disabled={busy || !ready}
          onNumResourcesChange={(value) =>
            setNumResources(clampValue(value, 0, 50))
          }
          onManualModeChange={setResourceManual}
          onRandom={() =>
            void runAction("Placing random resources", "place_resource_random", {
              numResources,
            }).then(() => markStepComplete(4))
          }
          onUndo={() => void runAction("Undo resource", "undo_resource")}
          onClear={() => void runAction("Clear resources", "clear_resource")}
        />
      );
    }
    return (
      <FinalizeStep
        disabled={busy || !ready}
        renderPreference={renderPreference}
        onRenderPreferenceChange={setRenderPreference}
        onExport={() => void handleFinalize()}
      />
    );
  };

  if (loading && !ready) {
    return <LoadingScreen stage={loadingStage} progress={loadingProgress} />;
  }

  return (
    <div className="wizard-shell">
      <header className="wizard-header">
        <h1>Rusted Warfare Map Generator (Web)</h1>
        <p>{statusText}</p>
        {error ? <p className="error-text">Worker error: {error}</p> : null}
        <p>
          Render mode: <strong>{renderMode}</strong>
        </p>
      </header>

      <StepBar
        steps={steps}
        currentStep={currentStep}
        completedStep={completedStep}
        onSelect={goToStep}
        isStepAccessible={isStepAccessible}
      />

      <div className="wizard-main">
        <aside className="wizard-panel">{handleStepContent()}</aside>
        <section className="wizard-preview">
          <MapCanvas
            snapshot={snapshot}
            tilesets={tilesets}
            view={mapView}
            requestedMode={renderPreference}
            interactionMode={interactionMode}
            drawValue={drawValue}
            brushSize={brushSize}
            onDraw={handleDrawWalls}
            onClickCell={handleCanvasClick}
            onRenderModeChange={setRenderMode}
          />
        </section>
      </div>

      <footer className="wizard-footer">
        <button type="button" className="ghost-btn" onClick={prevStep} disabled={currentStep === 0}>
          Back
        </button>
        <button
          type="button"
          className="primary-btn"
          onClick={handleNext}
          disabled={currentStep >= steps.length - 1 || (currentStep === 0 && !canAdvanceFromStep0)}
        >
          Next
        </button>
      </footer>
    </div>
  );
}
