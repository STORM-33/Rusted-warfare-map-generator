"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { UsePyodideResult } from "@/hooks/usePyodide";
import type {
    QuickGenerateFrame,
    WorkerStepCompleteMessage,
    WizardSnapshot,
} from "@/lib/types";
import type { ExtractedTilesets } from "@/lib/tilesetExtractor";
import { loadTilesetsFromBlueprint } from "@/lib/tilesetExtractor";
import { MapCanvas } from "@/components/MapCanvas";
import JSZip from "jszip";

type MirroringMode =
    | "none"
    | "horizontal"
    | "vertical"
    | "diagonal1"
    | "diagonal2"
    | "both";

const MIRROR_OPTIONS: MirroringMode[] = [
    "none",
    "horizontal",
    "vertical",
    "diagonal1",
    "diagonal2",
    "both",
];

const DEFAULT_GRID = [
    [1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
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
    if (mode === "horizontal" || mode === "both") addCell(size - 1 - row, col);
    if (mode === "vertical" || mode === "both") addCell(row, size - 1 - col);
    if (mode === "diagonal1") addCell(col, row);
    if (mode === "diagonal2") addCell(size - 1 - col, size - 1 - row);
    if (mode === "both") addCell(size - 1 - row, size - 1 - col);
    return Array.from(cells).map((e) => {
        const [r, c] = e.split(":").map(Number);
        return [r, c] as [number, number];
    });
};

export function QuickGeneratePage({ pyodide }: { pyodide: UsePyodideResult }) {
    const {
        ready,
        error,
        snapshot,
        callAction,
    } = pyodide;

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
    const [lastTmxBytes, setLastTmxBytes] = useState<Uint8Array | null>(null);
    const [lastBaseName, setLastBaseName] = useState<string>("");

    // Animation state
    const [quickFrames, setQuickFrames] = useState<QuickGenerateFrame[]>([]);
    const [frameIndex, setFrameIndex] = useState(-1);
    const animTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const animating = quickFrames.length > 0 && frameIndex >= 0;

    const blueprintCacheRef = useRef<Record<number, string>>({});

    useEffect(() => {
        if (ready) {
            setStatusText("Runtime ready — set your parameters and click Generate");
            void callAction("get_state_snapshot").catch(() => { });
        }
    }, [ready, callAction]);

    useEffect(() => {
        let cancelled = false;
        loadTilesetsFromBlueprint(`/blueprints/generator_blueprint${tileset}.tmx`)
            .then((result) => {
                if (!cancelled) setTilesets(result);
            })
            .catch(() => { });
        return () => {
            cancelled = true;
        };
    }, [tileset]);

    const getBlueprintXml = useCallback(
        async (pattern: number) => {
            const cached = blueprintCacheRef.current[pattern];
            if (cached) return cached;
            const response = await fetch(
                `/blueprints/generator_blueprint${pattern}.tmx`,
            );
            if (!response.ok) throw new Error(`Blueprint ${pattern} not found`);
            const xml = await response.text();
            blueprintCacheRef.current[pattern] = xml;
            return xml;
        },
        [],
    );

    const handleToggleGridCell = useCallback(
        (row: number, col: number) => {
            setGrid((prev) => {
                const size = prev.length;
                const next = prev.map((line) => [...line]);
                const newValue = next[row][col] ? 0 : 1;
                for (const [mr, mc] of getMirroredCells(row, col, size, mirroring)) {
                    next[mr][mc] = newValue;
                }
                return next;
            });
        },
        [mirroring],
    );

    // Animation timer
    useEffect(() => {
        if (frameIndex < 0 || quickFrames.length === 0) return;
        if (frameIndex >= quickFrames.length) {
            // Animation done — keep showing final state for a bit
            setQuickFrames([]);
            setFrameIndex(-1);
            setStatusText("Generation complete — download your map below");
            return;
        }
        const timer = setInterval(() => {
            setFrameIndex((prev) => prev + 1);
        }, 150);
        animTimerRef.current = timer;
        return () => {
            clearInterval(timer);
            animTimerRef.current = null;
        };
    }, [frameIndex, quickFrames.length]);

    // Build a display snapshot from the current animation frame
    const displaySnapshot = useMemo((): WizardSnapshot | null => {
        if (!animating || !snapshot) return snapshot;
        const frame = quickFrames[frameIndex];
        if (!frame) return snapshot;

        const matrices: WizardSnapshot["matrices"] = {
            ...snapshot.matrices,
            height_map: frame.height_map,
        };
        if (frame.id_matrix) matrices.id_matrix = frame.id_matrix;
        if (frame.items_matrix) matrices.items_matrix = frame.items_matrix;
        if (frame.units_matrix) matrices.units_matrix = frame.units_matrix;

        return {
            ...snapshot,
            meta: {
                ...snapshot.meta,
                height: frame.height_map.shape[0],
                width: frame.height_map.shape[1],
            },
            matrices,
        };
    }, [animating, quickFrames, frameIndex, snapshot]);

    const mapView = useMemo(() => {
        if (animating) {
            const frame = quickFrames[frameIndex];
            if (frame?.id_matrix) return "final" as const;
            return "height" as const;
        }
        if (snapshot?.matrices.id_matrix) return "final" as const;
        if (snapshot?.matrices.height_map) return "height" as const;
        return "coastline" as const;
    }, [animating, quickFrames, frameIndex, snapshot]);

    const handleGenerate = useCallback(async () => {
        try {
            // Clear previous animation
            if (animTimerRef.current) {
                clearInterval(animTimerRef.current);
                animTimerRef.current = null;
            }
            setQuickFrames([]);
            setFrameIndex(-1);
            setLastTmxBytes(null);
            setLastBaseName("");

            setBusy(true);
            setStatusText("Generating... please wait");

            const blueprintXml = await getBlueprintXml(tileset);

            const response = (await callAction("quick_generate", {
                grid,
                height: clampValue(height, 40, 640),
                width: clampValue(width, 40, 640),
                mirroring,
                tileset,
                heightLevels,
                oceanLevels,
                numPlayers,
                numResources,
                blueprintXml,
            })) as WorkerStepCompleteMessage;

            // Build filename
            let modeStr = "0v0";
            if (response.snapshot?.cc_positions) {
                const numCcs = response.snapshot.cc_positions.length;
                const half1 = Math.floor(numCcs / 2);
                const half2 = Math.ceil(numCcs / 2);
                modeStr =
                    half1 === 0 && half2 === 0 ? "0v0" : `${half1}v${half2}`;
            }
            const pad = (n: number) => n.toString().padStart(2, "0");
            const d = new Date();
            const timestampStr = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
            const baseName = `generated_${modeStr}_${timestampStr}`;

            if (response.tmxBytes) {
                setLastTmxBytes(Uint8Array.from(response.tmxBytes));
                setLastBaseName(baseName);
            }

            // Start animation if frames available
            if (response.quickFrames && response.quickFrames.length > 0) {
                setQuickFrames(response.quickFrames);
                setFrameIndex(0);
                setStatusText("Playing generation preview...");
            } else {
                setStatusText("Generation complete");
            }
        } catch (e) {
            const message = e instanceof Error ? e.message : "unknown";
            setStatusText(`Generation failed: ${message}`);
        } finally {
            setBusy(false);
        }
    }, [
        callAction,
        getBlueprintXml,
        grid,
        height,
        heightLevels,
        mirroring,
        numPlayers,
        numResources,
        oceanLevels,
        tileset,
        width,
    ]);

    const handleDownload = useCallback(async () => {
        if (!lastTmxBytes || !lastBaseName) return;
        const zip = new JSZip();
        zip.file(`${lastBaseName}.tmx`, lastTmxBytes);

        // Thumbnail from canvas
        const baseCanvas = document.getElementById(
            "map-canvas-base",
        ) as HTMLCanvasElement;
        const overlayCanvas = document.getElementById(
            "map-canvas-overlay",
        ) as HTMLCanvasElement;
        if (baseCanvas && overlayCanvas) {
            const thumbCanvas = document.createElement("canvas");
            thumbCanvas.width = 200;
            thumbCanvas.height = 200;
            const ctx = thumbCanvas.getContext("2d");
            if (ctx) {
                ctx.drawImage(
                    baseCanvas,
                    0, 0, baseCanvas.width, baseCanvas.height,
                    0, 0, 200, 200,
                );
                ctx.drawImage(
                    overlayCanvas,
                    0, 0, overlayCanvas.width, overlayCanvas.height,
                    0, 0, 200, 200,
                );
                await new Promise<void>((resolve) => {
                    thumbCanvas.toBlob((blob) => {
                        if (blob) zip.file(`${lastBaseName}_map.png`, blob);
                        resolve();
                    }, "image/png");
                });
            }
        }

        const zipBlob = await zip.generateAsync({ type: "blob" });
        const url = URL.createObjectURL(zipBlob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `${lastBaseName}.zip`;
        anchor.click();
        URL.revokeObjectURL(url);
        setStatusText(`${lastBaseName}.zip downloaded`);
    }, [lastTmxBytes, lastBaseName]);

    const isDisabled = busy || !ready || animating;
    const animProgress = animating
        ? `${frameIndex + 1}/${quickFrames.length}`
        : undefined;

    return (
        <div className="wizard-shell">
            <header className="wizard-header">
                <h1>Quick Generate</h1>
                <p>{statusText}</p>
                {error ? <p className="error-text">Worker error: {error}</p> : null}
            </header>

            <div className="wizard-main">
                <aside className="wizard-panel">
                    <section className="panel-section">
                        <h2>Settings</h2>
                        <p>Configure all parameters, then hit Generate.</p>

                        <div className="coast-grid">
                            {grid.map((row, rowIndex) =>
                                row.map((value, colIndex) => (
                                    <button
                                        key={`${rowIndex}-${colIndex}`}
                                        type="button"
                                        className={`grid-cell ${value ? "on" : "off"}`}
                                        onClick={() => handleToggleGridCell(rowIndex, colIndex)}
                                        disabled={isDisabled}
                                    />
                                )),
                            )}
                        </div>

                        <div className="control-grid">
                            <label>
                                Height
                                <input
                                    type="number"
                                    min={40}
                                    max={640}
                                    step={20}
                                    value={height}
                                    onChange={(e) => setHeight(clampValue(Number(e.target.value), 40, 640))}
                                    disabled={isDisabled}
                                />
                            </label>
                            <label>
                                Width
                                <input
                                    type="number"
                                    min={40}
                                    max={640}
                                    step={20}
                                    value={width}
                                    onChange={(e) => setWidth(clampValue(Number(e.target.value), 40, 640))}
                                    disabled={isDisabled}
                                />
                            </label>
                            <label>
                                Mirroring
                                <select
                                    value={mirroring}
                                    onChange={(e) => setMirroring(e.target.value as MirroringMode)}
                                    disabled={isDisabled}
                                >
                                    {MIRROR_OPTIONS.map((opt) => (
                                        <option key={opt} value={opt}>
                                            {opt}
                                        </option>
                                    ))}
                                </select>
                            </label>
                            <label>
                                Tileset
                                <select
                                    value={tileset}
                                    onChange={(e) => setTileset(Number(e.target.value))}
                                    disabled={isDisabled}
                                >
                                    {[
                                        { id: 5, name: "Jungle" },
                                        { id: 2, name: "Winter" },
                                        { id: 3, name: "Volcanic" },
                                        { id: 4, name: "Desert" },
                                        { id: 1, name: "Forest" },
                                    ].map(({ id, name }) => (
                                        <option key={name} value={id}>
                                            {name}
                                        </option>
                                    ))}
                                </select>
                            </label>
                            <label>
                                Height Levels
                                <input
                                    type="number"
                                    min={1}
                                    max={7}
                                    value={heightLevels}
                                    onChange={(e) =>
                                        setHeightLevels(clampValue(Number(e.target.value), 1, 7))
                                    }
                                    disabled={isDisabled}
                                />
                            </label>
                            <label>
                                Ocean Levels
                                <input
                                    type="number"
                                    min={1}
                                    max={3}
                                    value={oceanLevels}
                                    onChange={(e) =>
                                        setOceanLevels(clampValue(Number(e.target.value), 1, 3))
                                    }
                                    disabled={isDisabled}
                                />
                            </label>
                            <label>
                                Players
                                <input
                                    type="number"
                                    min={0}
                                    max={10}
                                    step={2}
                                    value={numPlayers}
                                    onChange={(e) =>
                                        setNumPlayers(clampValue(Number(e.target.value), 0, 10))
                                    }
                                    disabled={isDisabled}
                                />
                            </label>
                            <label>
                                Resources
                                <input
                                    type="number"
                                    min={0}
                                    max={50}
                                    value={numResources}
                                    onChange={(e) =>
                                        setNumResources(clampValue(Number(e.target.value), 0, 50))
                                    }
                                    disabled={isDisabled}
                                />
                            </label>
                        </div>

                        <div className="button-row">
                            <button
                                type="button"
                                className="quick-btn"
                                onClick={() => void handleGenerate()}
                                disabled={isDisabled}
                            >
                                {animating
                                    ? `Generating... ${animProgress}`
                                    : "Generate Map"}
                            </button>
                            {lastTmxBytes && !animating && (
                                <button
                                    type="button"
                                    className="primary-btn"
                                    onClick={() => void handleDownload()}
                                >
                                    Download .zip
                                </button>
                            )}
                        </div>
                    </section>
                </aside>

                <section className="wizard-preview">
                    <MapCanvas
                        snapshot={displaySnapshot}
                        tilesets={tilesets}
                        view={mapView}
                        requestedMode="auto"
                        interactionMode="none"
                        drawValue={1}
                        onDraw={() => { }}
                        onClickCell={() => { }}
                    />
                </section>
            </div>
        </div>
    );
}
