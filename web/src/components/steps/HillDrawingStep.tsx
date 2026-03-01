"use client";

type HillDrawingStepProps = {
  drawValue: 1 | 2;
  brushSize: number;
  disabled?: boolean;
  onDrawValueChange: (value: 1 | 2) => void;
  onBrushSizeChange: (size: number) => void;
  onClear: () => void;
};

export function HillDrawingStep({
  drawValue,
  brushSize,
  disabled,
  onDrawValueChange,
  onBrushSizeChange,
  onClear,
}: HillDrawingStepProps) {
  return (
    <section className="panel-section">
      <h2>2. Hill Drawing</h2>
      <p>Draw on the map canvas. Left click uses selected brush, right click clears.</p>
      <div className="button-row">
        <button
          type="button"
          className={drawValue === 1 ? "active-btn" : "ghost-btn"}
          onClick={() => onDrawValueChange(1)}
          disabled={disabled}
        >
          Wall Brush
        </button>
        <button
          type="button"
          className={drawValue === 2 ? "active-btn" : "ghost-btn"}
          onClick={() => onDrawValueChange(2)}
          disabled={disabled}
        >
          Gap Brush
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
