"use client";

type HillDrawingStepProps = {
  drawValue: 1 | 2;
  eraseMode: boolean;
  brushSize: number;
  disabled?: boolean;
  onDrawValueChange: (value: 1 | 2) => void;
  onEraseModeChange: (erasing: boolean) => void;
  onBrushSizeChange: (size: number) => void;
  onClear: () => void;
};

export function HillDrawingStep({
  drawValue,
  eraseMode,
  brushSize,
  disabled,
  onDrawValueChange,
  onEraseModeChange,
  onBrushSizeChange,
  onClear,
}: HillDrawingStepProps) {
  return (
    <section className="panel-section">
      <h2>2. Hill Drawing</h2>
      <p>Draw on the map canvas. Use eraser or right-click to clear cells.</p>
      <div className="button-row">
        <button
          type="button"
          className={drawValue === 1 && !eraseMode ? "active-btn" : "ghost-btn"}
          onClick={() => onDrawValueChange(1)}
          disabled={disabled}
        >
          Wall Brush
        </button>
        <button
          type="button"
          className={drawValue === 2 && !eraseMode ? "active-btn" : "ghost-btn"}
          onClick={() => onDrawValueChange(2)}
          disabled={disabled}
        >
          Gap Brush
        </button>
        <button
          type="button"
          className={eraseMode ? "active-btn" : "ghost-btn"}
          onClick={() => onEraseModeChange(!eraseMode)}
          disabled={disabled}
        >
          Eraser
        </button>
        <button type="button" className="ghost-btn" onClick={onClear} disabled={disabled}>
          Clear Walls
        </button>
      </div>
      <label>
        Brush size: {brushSize}
        <input
          type="range"
          min={1}
          max={9}
          step={2}
          value={brushSize}
          onChange={(e) => onBrushSizeChange(Number(e.target.value))}
          disabled={disabled}
        />
      </label>
    </section>
  );
}
