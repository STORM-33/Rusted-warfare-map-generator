"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { UsePyodideResult } from "@/hooks/usePyodide";
import { useWizardState } from "@/hooks/useWizardState";
import type { CoastlineFrame, WorkerAction, WorkerStepCompleteMessage, WizardSnapshot } from "@/lib/types";
import type { ExtractedTilesets } from "@/lib/tilesetExtractor";
import { loadTilesetsFromBlueprint } from "@/lib/tilesetExtractor";
import { StepBar } from "@/components/StepBar";
import { MapCanvas } from "@/components/MapCanvas";
import { CoastlineStep } from "@/components/steps/CoastlineStep";
import { HillDrawingStep } from "@/components/steps/HillDrawingStep";
import { HeightOceanStep } from "@/components/steps/HeightOceanStep";
import { CommandCenterStep } from "@/components/steps/CommandCenterStep";
import { ResourceStep } from "@/components/steps/ResourceStep";
import { FinalizeStep } from "@/components/steps/FinalizeStep";
import JSZip from "jszip";

type MirroringMode =
  | "none"
  | "horizontal"
  | "vertical"
  | "diagonal1"
  | "diagonal2"
  | "both";



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

export function WizardApp({ pyodide }: { pyodide: UsePyodideResult }) {
  const {
    ready,
    error,
    snapshot,
    callAction,
  } = pyodide;
  const {
    steps,
    currentStep,
    completedStep,
    goToStep,
    nextStep,
    prevStep,
    markStepComplete,
    isStepAccessible,
    resetWizard,
  } = useWizardState();

  const [grid, setGrid] = useState<number[][]>(DEFAULT_GRID);
  const [height, setHeight] = useState(160);
  const [width, setWidth] = useState(160);
  const [mirroring, setMirroring] = useState<MirroringMode>("vertical");
  const [tileset, setTileset] = useState(5);
  const [heightLevels, setHeightLevels] = useState(7);
  const [oceanLevels, setOceanLevels] = useState(3);
  const [numPlayers, setNumPlayers] = useState(4);
  const [numResources, setNumResources] = useState(12);
  const [drawValue, setDrawValue] = useState<1 | 2>(1);
  const [brushSize, setBrushSize] = useState(3);
  const [ccManual, setCcManual] = useState(true);
  const [ccMirrored, setCcMirrored] = useState(true);
  const [resourceManual, setResourceManual] = useState(true);
  const [resourceMirrored, setResourceMirrored] = useState(true);
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState("Waiting for runtime...");
  const [tilesets, setTilesets] = useState<ExtractedTilesets>();

  const [animationFrames, setAnimationFrames] = useState<CoastlineFrame[]>([]);
  const [animationIndex, setAnimationIndex] = useState(-1);
  const animationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const animating = animationFrames.length > 0 && animationIndex >= 0;

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
    // Clear any previous animation
    if (animationTimerRef.current) {
      clearInterval(animationTimerRef.current);
      animationTimerRef.current = null;
    }
    setAnimationFrames([]);
    setAnimationIndex(-1);

    const response = (await runAction("Generating coastline", "run_coastline", {
      grid,
      height: clampValue(height, 40, 640),
      width: clampValue(width, 40, 640),
      mirroring,
      tileset,
      heightLevels,
      oceanLevels,
      numPlayers,
      numResources,
    })) as WorkerStepCompleteMessage;

    markStepComplete(0);

    // Start animation if frames are available
    if (response.frames && response.frames.length > 0) {
      setAnimationFrames(response.frames);
      setAnimationIndex(0);
    }
  }, [
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

  // Animation timer: advance through frames
  useEffect(() => {
    if (animationIndex < 0 || animationFrames.length === 0) {
      return;
    }
    if (animationIndex >= animationFrames.length) {
      // Animation complete
      setAnimationFrames([]);
      setAnimationIndex(-1);
      return;
    }
    const timer = setInterval(() => {
      setAnimationIndex((prev) => prev + 1);
    }, 150);
    animationTimerRef.current = timer;
    return () => {
      clearInterval(timer);
      animationTimerRef.current = null;
    };
  }, [animationIndex, animationFrames.length]);

  // Build snapshot with animation frame override
  const displaySnapshot = useMemo((): WizardSnapshot | null => {
    if (!animating || !snapshot) {
      return snapshot;
    }
    const frame = animationFrames[animationIndex];
    if (!frame) {
      return snapshot;
    }
    return {
      ...snapshot,
      meta: {
        ...snapshot.meta,
        height: frame.shape[0],
        width: frame.shape[1],
      },
      matrices: {
        ...snapshot.matrices,
        coastline_height_map: {
          shape: frame.shape,
          data: frame.data,
        },
      },
    };
  }, [animating, animationFrames, animationIndex, snapshot]);

  const animationProgress = animating
    ? `Step ${animationIndex + 1}/${animationFrames.length}`
    : undefined;

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
    (row: number, col: number, isRightClick?: boolean) => {
      if (currentStep === 3 && ccManual) {
        const actionName = isRightClick ? "remove_cc_manual" : "place_cc_manual";
        void callAction(actionName, { row, col, mirrored: ccMirrored })
          .then((response) => {
            const newCount = response.snapshot?.cc_positions.length ?? 0;
            if (isRightClick) {
              setStatusText("CC removed.");
            } else if (newCount > 0) {
              markStepComplete(3);
              setStatusText(`${newCount} CC positions placed.`);
            } else {
              setStatusText("Invalid position (water, wall, or out of bounds).");
            }
          })
          .catch((clickError) => {
            setStatusText(
              `CC action failed: ${clickError instanceof Error ? clickError.message : "unknown"}`,
            );
          });
      } else if (currentStep === 4 && resourceManual) {
        const actionName = isRightClick ? "remove_resource_manual" : "place_resource_manual";
        void callAction(actionName, { row, col, mirrored: resourceMirrored })
          .then((response) => {
            const newCount = response.snapshot?.resource_positions.length ?? 0;
            if (isRightClick) {
              setStatusText("Resource removed.");
            } else if (newCount > 0) {
              markStepComplete(4);
              setStatusText(`${newCount} resource positions placed.`);
            } else {
              setStatusText("Invalid position.");
            }
          })
          .catch((clickError) => {
            setStatusText(
              `Resource action failed: ${clickError instanceof Error ? clickError.message : "unknown"}`,
            );
          });
      }
    },
    [callAction, ccManual, currentStep, markStepComplete, resourceManual, ccMirrored, resourceMirrored],
  );

  const handleFinalize = useCallback(async () => {
    const blueprintXml = await getBlueprintXml(tileset);
    if (!blueprintXml) {
      setStatusText("Failed to build blueprint XML");
      return;
    }
    setStatusText("Finalizing... please wait");

    let modeStr = "0v0";
    if (snapshot?.cc_positions) {
      const numCcs = snapshot.cc_positions.length;
      const half1 = Math.floor(numCcs / 2);
      const half2 = Math.ceil(numCcs / 2);
      modeStr = half1 === 0 && half2 === 0 ? "0v0" : `${half1}v${half2}`;
    }

    const pad = (n: number) => n.toString().padStart(2, "0");
    const d = new Date();
    const timestampStr = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    const baseName = `generated_${modeStr}_${timestampStr}`;

    const response = (await runAction("Finalizing map", "run_finalize", {
      blueprintXml,
    })) as WorkerStepCompleteMessage;

    if (response.tmxBytes) {
      const zip = new JSZip();
      zip.file(`${baseName}.tmx`, response.tmxBytes);

      const baseCanvas = document.getElementById("wizard-map-canvas-base") as HTMLCanvasElement;
      const overlayCanvas = document.getElementById("wizard-map-canvas-overlay") as HTMLCanvasElement;

      if (baseCanvas && overlayCanvas) {
        const thumbCanvas = document.createElement("canvas");
        thumbCanvas.width = 200;
        thumbCanvas.height = 200;
        const ctx = thumbCanvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(baseCanvas, 0, 0, baseCanvas.width, baseCanvas.height, 0, 0, 200, 200);
          ctx.drawImage(overlayCanvas, 0, 0, overlayCanvas.width, overlayCanvas.height, 0, 0, 200, 200);

          await new Promise<void>((resolve) => {
            thumbCanvas.toBlob((blob) => {
              if (blob) {
                zip.file(`${baseName}_map.png`, blob);
              }
              resolve();
            }, "image/png");
          });
        }
      }

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(zipBlob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${baseName}.zip`;
      anchor.click();
      URL.revokeObjectURL(url);

      setStatusText(`${baseName}.zip downloaded`);
      markStepComplete(5);
    } else {
      setStatusText("Finalize completed without output bytes");
    }
  }, [getBlueprintXml, markStepComplete, runAction, snapshot?.cc_positions, tileset]);

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
  const canAdvanceFromStep2 = Boolean(snapshot?.matrices.height_map);

  const handleNext = () => {
    if (currentStep === 0 && !canAdvanceFromStep0) {
      return;
    }
    if (currentStep === 2 && !canAdvanceFromStep2) {
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

  const handleFinish = async () => {
    try {
      setBusy(true);
      await callAction("reset_state");
      setGrid(DEFAULT_GRID);
      setAnimationFrames([]);
      setAnimationIndex(-1);
      resetWizard();
      setStatusText("App state reset for a new map.");
    } catch (e) {
      setStatusText("Failed to reset state.");
    } finally {
      setBusy(false);
    }
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
          animating={animating}
          animationProgress={animationProgress}
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
          mirrored={ccMirrored}
          disabled={busy || !ready}
          onNumPlayersChange={(value) => setNumPlayers(clampValue(value, 2, 10))}
          onManualModeChange={setCcManual}
          onMirroredChange={setCcMirrored}
          onRandom={() =>
            void runAction("Placing random command centers", "place_cc_random", {
              numPlayers,
            }).then((response) => {
              markStepComplete(3);
              const count = (response as WorkerStepCompleteMessage).snapshot?.cc_positions.length ?? 0;
              setStatusText(`Placed ${count} command centers.`);
            })
          }
          onUndo={() =>
            void runAction("Undo command center", "undo_cc").then((response) => {
              const count = (response as WorkerStepCompleteMessage).snapshot?.cc_positions.length ?? 0;
              setStatusText(`${count} CC positions remain.`);
            })
          }
          onClear={() =>
            void runAction("Clear command centers", "clear_cc").then(() => {
              setStatusText("Cleared all command centers.");
            })
          }
        />
      );
    }
    if (currentStep === 4) {
      return (
        <ResourceStep
          numResources={numResources}
          manualMode={resourceManual}
          mirrored={resourceMirrored}
          disabled={busy || !ready}
          onNumResourcesChange={(value) =>
            setNumResources(clampValue(value, 0, 50))
          }
          onManualModeChange={setResourceManual}
          onMirroredChange={setResourceMirrored}
          onRandom={() =>
            void runAction("Placing random resources", "place_resource_random", {
              numResources,
            }).then((response) => {
              markStepComplete(4);
              const count = (response as WorkerStepCompleteMessage).snapshot?.resource_positions.length ?? 0;
              setStatusText(`Placed ${count} resource positions.`);
            })
          }
          onUndo={() =>
            void runAction("Undo resource", "undo_resource").then((response) => {
              const count = (response as WorkerStepCompleteMessage).snapshot?.resource_positions.length ?? 0;
              setStatusText(`${count} resource positions remain.`);
            })
          }
          onClear={() =>
            void runAction("Clear resources", "clear_resource").then(() => {
              setStatusText("Cleared all resources.");
            })
          }
        />
      );
    }
    return (
      <FinalizeStep
        disabled={busy || !ready}
        onExport={() => void handleFinalize()}
      />
    );
  };

  return (
    <div className="wizard-shell">
      <header className="wizard-header">
        <h1>Rusted Warfare Map Generator</h1>
        <p>{statusText}</p>
        {error ? <p className="error-text">Worker error: {error}</p> : null}
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
            snapshot={displaySnapshot}
            tilesets={tilesets}
            view={mapView}
            requestedMode="auto"
            interactionMode={interactionMode}
            drawValue={drawValue}
            canvasIdPrefix="wizard-map-canvas"
            onDraw={handleDrawWalls}
            onClickCell={handleCanvasClick}
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
          onClick={currentStep >= steps.length - 1 ? handleFinish : handleNext}
          disabled={
            (currentStep === 0 && !canAdvanceFromStep0) ||
            (currentStep === 2 && !canAdvanceFromStep2) ||
            animating
          }
        >
          {currentStep >= steps.length - 1 ? "Finish" : "Next"}
        </button>
      </footer>
    </div>
  );
}
