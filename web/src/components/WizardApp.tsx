"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { UseMapEngineResult } from "@/hooks/useMapEngine";
import { useWizardState } from "@/hooks/useWizardState";
import type { CoastlineFrame, HillDrawingMode, Polygon, WorkerAction, WorkerStepCompleteMessage, WizardSnapshot } from "@/lib/types";
import type { ExtractedTilesets } from "@/lib/tilesetExtractor";
import { loadTilesetsFromBlueprint } from "@/lib/tilesetExtractor";
import { generatePolygonId, findPolygonAtPoint, mergeIntersectingPolygons } from "@/lib/polygonUtils";
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

export function WizardApp({ mapEngine }: { mapEngine: UseMapEngineResult }) {
  const {
    ready,
    error,
    snapshot,
    callAction,
  } = mapEngine;
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
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState("Waiting for runtime...");
  const [tilesets, setTilesets] = useState<ExtractedTilesets>();

  const [animationFrames, setAnimationFrames] = useState<CoastlineFrame[]>([]);
  const [animationIndex, setAnimationIndex] = useState(-1);
  const animationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ========== HILL DRAWING MODE STATE ==========
  const [hillMode, setHillMode] = useState<HillDrawingMode>("brush");
  
  // ========== CC/RESOURCE MODE STATE ==========
  const [ccManual, setCcManual] = useState(true);
  const [ccMirrored, setCcMirrored] = useState(true);
  const [resourceManual, setResourceManual] = useState(true);
  const [resourceMirrored, setResourceMirrored] = useState(true);
  
  // ========== BRUSH MODE STATE ==========
  const [drawValue, setDrawValue] = useState<1 | 2>(1);
  const [eraseMode, setEraseMode] = useState(false);
  const [brushSize, setBrushSize] = useState(3);
  
  // ========== POLYGON MODE STATE ==========
  const [polygons, setPolygons] = useState<Polygon[]>([]);
  const [drawingPolygon, setDrawingPolygon] = useState<Polygon | null>(null);
  const [selectedPolygonId, setSelectedPolygonId] = useState<number | null>(null);
  const [selectedEdgeIndex, setSelectedEdgeIndex] = useState<number | null>(null);

  const animating = animationFrames.length > 0 && animationIndex >= 0;

  // Sync hill mode with backend when it changes
  const handleModeChange = useCallback(async (mode: HillDrawingMode) => {
    setHillMode(mode);
    try {
      await callAction("set_hill_drawing_mode", { mode });
    } catch (err) {
      setStatusText(`Mode switch error: ${err instanceof Error ? err.message : "unknown"}`);
    }
  }, [callAction]);

  // Initialize hill mode from snapshot
  useEffect(() => {
    if (snapshot?.meta.hill_drawing_mode) {
      setHillMode(snapshot.meta.hill_drawing_mode);
    }
  }, [snapshot?.meta.hill_drawing_mode]);

  // Re-apply mirroring to the grid when the mirror mode changes
  useEffect(() => {
    if (mirroring === "none") return;
    setGrid((prev) => {
      const size = prev.length;
      const next = prev.map((row) => [...row]);
      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          for (const [mr, mc] of getMirroredCells(r, c, size, mirroring)) {
            next[mr][mc] = next[r][c];
          }
        }
      }
      return next;
    });
  }, [mirroring]);

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

  // ========== BRUSH MODE HANDLERS ==========
  const handleDrawBrushWalls = useCallback(
    (points: [number, number][], value: 0 | 1 | 2) => {
      void callAction("draw_brush_walls", { points, value, brush_size: brushSize }).catch((drawError) => {
        setStatusText(
          `Brush draw error: ${drawError instanceof Error ? drawError.message : "unknown"}`,
        );
      });
    },
    [brushSize, callAction],
  );

  const handleUndoBrush = useCallback(async () => {
    try {
      await callAction("undo_brush");
      setStatusText("Undo brush stroke");
    } catch (err) {
      setStatusText(`Undo failed: ${err instanceof Error ? err.message : "unknown"}`);
    }
  }, [callAction]);

  const handleRedoBrush = useCallback(async () => {
    try {
      await callAction("redo_brush");
      setStatusText("Redo brush stroke");
    } catch (err) {
      setStatusText(`Redo failed: ${err instanceof Error ? err.message : "unknown"}`);
    }
  }, [callAction]);

  const handleClearBrushWalls = useCallback(() => {
    void runAction("Clearing brush walls", "clear_brush_walls");
  }, [runAction]);

  // ========== POLYGON MODE HANDLERS ==========
  const CLOSE_THRESHOLD = 3;

  const handleUpdatePolygons = useCallback(async (newPolygons: Polygon[]) => {
    setPolygons(newPolygons);
    try {
      await callAction("update_polygons", {
        polygons: newPolygons
          .filter((p) => p.closed && p.vertices.length >= 3)
          .map((p) => ({
            id: p.id,
            vertices: p.vertices,
            edgeGaps: p.edgeGaps,
          })),
      });
    } catch (syncError) {
      setStatusText(
        `Polygon sync error: ${syncError instanceof Error ? syncError.message : "unknown"}`,
      );
    }
  }, [callAction]);

  const handleUndoPolygons = useCallback(async () => {
    try {
      await callAction("undo_polygons");
      setStatusText("Undo polygon action");
    } catch (err) {
      setStatusText(`Undo failed: ${err instanceof Error ? err.message : "unknown"}`);
    }
  }, [callAction]);

  const handleRedoPolygons = useCallback(async () => {
    try {
      await callAction("redo_polygons");
      setStatusText("Redo polygon action");
    } catch (err) {
      setStatusText(`Redo failed: ${err instanceof Error ? err.message : "unknown"}`);
    }
  }, [callAction]);

  const handleClearAllPolygons = useCallback(() => {
    setPolygons([]);
    setDrawingPolygon(null);
    setSelectedPolygonId(null);
    setSelectedEdgeIndex(null);
    void runAction("Clearing all polygons", "clear_all_polygons");
  }, [runAction]);

  const handleToggleEdgeGap = useCallback(async (polygonId: number, edgeIndex: number) => {
    try {
      await callAction("toggle_edge_gap", { polygon_id: polygonId, edge_index: edgeIndex });
      // Update local state to reflect the change
      setPolygons((prev) =>
        prev.map((p) => {
          if (p.id === polygonId && edgeIndex < p.edgeGaps.length) {
            const newGaps = [...p.edgeGaps];
            newGaps[edgeIndex] = !newGaps[edgeIndex];
            return { ...p, edgeGaps: newGaps };
          }
          return p;
        })
      );
    } catch (err) {
      setStatusText(`Toggle gap failed: ${err instanceof Error ? err.message : "unknown"}`);
    }
  }, [callAction]);

  const handlePolygonClick = useCallback(
    (row: number, col: number) => {
      if (drawingPolygon) {
        const verts = drawingPolygon.vertices;
        // Check if clicking near first vertex to close
        if (verts.length >= 3) {
          const [fr, fc] = verts[0];
          const dist = Math.sqrt((row - fr) ** 2 + (col - fc) ** 2);
          if (dist <= CLOSE_THRESHOLD) {
            const closed: Polygon = {
              ...drawingPolygon,
              closed: true,
            };
            const mapRows = snapshot?.meta.height;
            const mapCols = snapshot?.meta.width;
            const newPolygons = mergeIntersectingPolygons(
              polygons, closed, mapRows, mapCols, mirroring,
            );
            setDrawingPolygon(null);
            // Select the last polygon in the result (the merged one or the new one)
            setSelectedPolygonId(newPolygons[newPolygons.length - 1]?.id ?? null);
            setSelectedEdgeIndex(null);
            void handleUpdatePolygons(newPolygons);
            return;
          }
        }
        // Add vertex (allow intersection - backend will handle union)
        setDrawingPolygon({
          ...drawingPolygon,
          vertices: [...verts, [row, col]],
          edgeGaps: [...drawingPolygon.edgeGaps, false],
        });
      } else {
        // Check if clicking inside an existing polygon (for selection)
        const hit = findPolygonAtPoint(polygons, row, col);
        if (hit) {
          setSelectedPolygonId(hit.id);
          // Check if click is near an edge to select it
          const nearestEdge = findNearestEdge(hit, row, col, 3);
          setSelectedEdgeIndex(nearestEdge);
        } else {
          // Start new polygon
          setSelectedPolygonId(null);
          setSelectedEdgeIndex(null);
          setDrawingPolygon({
            id: generatePolygonId(),
            vertices: [[row, col]],
            edgeGaps: [],
            closed: false,
          });
        }
      }
    },
    [drawingPolygon, polygons, handleUpdatePolygons, snapshot?.meta.height, snapshot?.meta.width, mirroring],
  );

  const handlePolygonRightClick = useCallback(() => {
    if (drawingPolygon) {
      setDrawingPolygon(null);
    } else {
      setSelectedPolygonId(null);
      setSelectedEdgeIndex(null);
    }
  }, [drawingPolygon]);

  const handleDeletePolygon = useCallback(() => {
    if (selectedPolygonId == null) return;
    const newPolygons = polygons.filter((p) => p.id !== selectedPolygonId);
    setSelectedPolygonId(null);
    setSelectedEdgeIndex(null);
    void handleUpdatePolygons(newPolygons);
  }, [polygons, selectedPolygonId, handleUpdatePolygons]);

  const handleCancelDrawing = useCallback(() => {
    setDrawingPolygon(null);
  }, []);

  // Helper to find nearest edge of a polygon to a point
  const findNearestEdge = (poly: Polygon, row: number, col: number, threshold: number): number | null => {
    let bestDist = threshold;
    let bestEdge: number | null = null;
    
    for (let i = 0; i < poly.vertices.length; i++) {
      const [r1, c1] = poly.vertices[i];
      const [r2, c2] = poly.vertices[(i + 1) % poly.vertices.length];
      const dist = pointToSegmentDistance(row, col, r1, c1, r2, c2);
      if (dist < bestDist) {
        bestDist = dist;
        bestEdge = i;
      }
    }
    
    return bestEdge;
  };

  const pointToSegmentDistance = (px: number, py: number, x1: number, y1: number, x2: number, y2: number): number => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (dx === 0 && dy === 0) {
      return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
    }
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    const projX = x1 + t * dx;
    const projY = y1 + t * dy;
    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
  };

  const handleCanvasClick = useCallback(
    (row: number, col: number, isRightClick?: boolean) => {
      if (currentStep === 3 && ccManual) {
        const hasExisting = snapshot?.cc_positions.some(([r, c]) => r === row && c === col);
        const shouldRemove = isRightClick || hasExisting;
        const actionName = shouldRemove ? "remove_cc_manual" : "place_cc_manual";
        void callAction(actionName, { row, col, mirrored: ccMirrored })
          .then((response) => {
            const newCount = response.snapshot?.cc_positions.length ?? 0;
            if (shouldRemove) {
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
        const hasExisting = snapshot?.resource_positions.some(([r, c]) => r === row && c === col);
        const shouldRemove = isRightClick || hasExisting;
        const actionName = shouldRemove ? "remove_resource_manual" : "place_resource_manual";
        void callAction(actionName, { row, col, mirrored: resourceMirrored })
          .then((response) => {
            const newCount = response.snapshot?.resource_positions.length ?? 0;
            if (shouldRemove) {
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
    [callAction, ccManual, currentStep, markStepComplete, resourceManual, ccMirrored, resourceMirrored, snapshot?.cc_positions, snapshot?.resource_positions],
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
      return hillMode === "polygon" ? ("polygon" as const) : ("draw" as const);
    }
    if ((currentStep === 3 && ccManual) || (currentStep === 4 && resourceManual)) {
      return "click" as const;
    }
    return "none" as const;
  }, [ccManual, currentStep, hillMode, resourceManual]);

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
      setPolygons([]);
      setDrawingPolygon(null);
      setSelectedPolygonId(null);
      setSelectedEdgeIndex(null);
      setHillMode("brush");
      resetWizard();
      setStatusText("App state reset for a new map.");
    } catch {
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
          onHeightChange={setHeight}
          onWidthChange={setWidth}
          onMirroringChange={(value) => setMirroring(value as MirroringMode)}
          onTilesetChange={setTileset}
          onGenerate={handleGenerateCoastline}
        />
      );
    }
    if (currentStep === 1) {
      return (
        <HillDrawingStep
          // Mode
          mode={hillMode}
          onModeChange={handleModeChange}
          // Brush props
          drawValue={drawValue}
          eraseMode={eraseMode}
          brushSize={brushSize}
          onDrawValueChange={(v) => { setDrawValue(v); setEraseMode(false); }}
          onEraseModeChange={setEraseMode}
          onBrushSizeChange={setBrushSize}
          onClearBrush={handleClearBrushWalls}
          onUndoBrush={handleUndoBrush}
          onRedoBrush={handleRedoBrush}
          // Polygon props
          polygons={polygons}
          drawingPolygon={drawingPolygon}
          selectedPolygonId={selectedPolygonId}
          selectedEdgeIndex={selectedEdgeIndex}
          onPolygonClick={handlePolygonClick}
          onToggleEdgeGap={handleToggleEdgeGap}
          onDeletePolygon={handleDeletePolygon}
          onClearPolygons={handleClearAllPolygons}
          onCancelDrawing={handleCancelDrawing}
          onUndoPolygons={handleUndoPolygons}
          onRedoPolygons={handleRedoPolygons}
          // Common
          disabled={busy || !ready}
        />
      );
    }
    if (currentStep === 2) {
      return (
        <HeightOceanStep
          heightLevels={heightLevels}
          oceanLevels={oceanLevels}
          disabled={busy || !ready}
          onHeightLevelsChange={setHeightLevels}
          onOceanLevelsChange={setOceanLevels}
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
          mirroringMode={mirroring}
          disabled={busy || !ready}
          onNumPlayersChange={setNumPlayers}
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
          onNumResourcesChange={setNumResources}
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

  // Determine which props to pass to MapCanvas based on mode
  const showPolygons = hillMode === "polygon" && currentStep === 1;
  const showBrushDraw = hillMode === "brush" && currentStep === 1;

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
            eraseMode={eraseMode}
            canvasIdPrefix="wizard-map-canvas"
            // Polygon mode: show polygons, hide brush
            polygons={showPolygons ? polygons : []}
            drawingPolygon={showPolygons ? drawingPolygon : null}
            selectedPolygonId={showPolygons ? selectedPolygonId : null}
            selectedEdgeIndex={showPolygons ? selectedEdgeIndex : null}
            // Brush mode: show brush interactions
            onDraw={showBrushDraw ? handleDrawBrushWalls : undefined}
            onClickCell={handleCanvasClick}
            onPolygonClick={showPolygons ? handlePolygonClick : undefined}
            onPolygonRightClick={showPolygons ? handlePolygonRightClick : undefined}
            // Mode for canvas rendering
            hillMode={currentStep === 1 ? hillMode : null}
            mirroring={showPolygons ? mirroring : undefined}
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
