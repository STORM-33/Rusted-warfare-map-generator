"use client";

type CoastlineStepProps = {
  grid: number[][];
  height: number;
  width: number;
  mirroring: string;
  tileset: number;
  disabled?: boolean;
  animating?: boolean;
  animationProgress?: string;
  onToggleCell: (row: number, col: number) => void;
  onHeightChange: (value: number) => void;
  onWidthChange: (value: number) => void;
  onMirroringChange: (value: string) => void;
  onTilesetChange: (value: number) => void;
  onGenerate: () => void;
};

const MIRROR_OPTIONS = [
  "none",
  "horizontal",
  "vertical",
  "diagonal1",
  "diagonal2",
  "both",
];

export function CoastlineStep({
  grid,
  height,
  width,
  mirroring,
  tileset,
  disabled,
  animating,
  animationProgress,
  onToggleCell,
  onHeightChange,
  onWidthChange,
  onMirroringChange,
  onTilesetChange,
  onGenerate,
}: CoastlineStepProps) {
  const isDisabled = disabled || animating;
  return (
    <section className="panel-section">
      <h2>1. Coastline</h2>
      <p>Toggle the 5x5 seed grid, then generate mirrored coastline terrain.</p>
      <div className="coast-grid">
        {grid.map((row, rowIndex) =>
          row.map((value, colIndex) => (
            <button
              key={`${rowIndex}-${colIndex}`}
              type="button"
              className={`grid-cell ${value ? "on" : "off"}`}
              onClick={() => onToggleCell(rowIndex, colIndex)}
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
            onChange={(event) => onHeightChange(Number(event.target.value))}
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
            onChange={(event) => onWidthChange(Number(event.target.value))}
            disabled={isDisabled}
          />
        </label>
        <label>
          Mirroring
          <select
            value={mirroring}
            onChange={(event) => onMirroringChange(event.target.value)}
            disabled={isDisabled}
          >
            {MIRROR_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label>
          Tileset
          <select
            value={tileset}
            onChange={(event) => onTilesetChange(Number(event.target.value))}
            disabled={isDisabled}
          >
            {[
              { id: 5, name: "Jungle" },
              { id: 2, name: "Winter" },
              { id: 3, name: "Volcanic" },
              { id: 4, name: "Desert" },
              { id: 1, name: "Forest" },
            ].map(
              ({ id, name }) => (
                <option key={name} value={id}>
                  {name}
                </option>
              ),
            )}
          </select>
        </label>
      </div>
      <button type="button" className="primary-btn" onClick={onGenerate} disabled={isDisabled}>
        {animating ? `Generating... ${animationProgress ?? ""}` : "Generate Coastline"}
      </button>
    </section>
  );
}
