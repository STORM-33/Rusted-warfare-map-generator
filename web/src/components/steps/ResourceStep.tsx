"use client";

type ResourceStepProps = {
  numResources: number;
  manualMode: boolean;
  mirrored: boolean;
  disabled?: boolean;
  onNumResourcesChange: (value: number) => void;
  onManualModeChange: (manual: boolean) => void;
  onMirroredChange: (mirrored: boolean) => void;
  onRandom: () => void;
  onUndo: () => void;
  onClear: () => void;
};

export function ResourceStep({
  numResources,
  manualMode,
  mirrored,
  disabled,
  onNumResourcesChange,
  onManualModeChange,
  onMirroredChange,
  onRandom,
  onUndo,
  onClear,
}: ResourceStepProps) {
  return (
    <section className="panel-section">
      <h2>5. Resources</h2>
      <p>Place mirrored resource pools manually or generate them automatically.</p>
      <div className="control-grid">
        <label>
          Num Resource Pulls
          <input
            type="number"
            min={0}
            max={50}
            value={numResources}
            onChange={(event) => onNumResourcesChange(Number(event.target.value))}
            disabled={disabled}
          />
        </label>
        {manualMode && (
          <label className="checkbox-label" style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "1.5rem" }}>
            <input
              type="checkbox"
              checked={mirrored}
              onChange={(e) => onMirroredChange(e.target.checked)}
              disabled={disabled}
            />
            Mirrored Placement
          </label>
        )}
      </div>
      <div className="button-row">
        <button
          type="button"
          className={manualMode ? "active-btn" : "ghost-btn"}
          onClick={() => onManualModeChange(true)}
          disabled={disabled}
        >
          Manual
        </button>
        <button
          type="button"
          className={!manualMode ? "active-btn" : "ghost-btn"}
          onClick={() => onManualModeChange(false)}
          disabled={disabled}
        >
          Random
        </button>
      </div>
      <div className="button-row">
        <button type="button" className="primary-btn" onClick={onRandom} disabled={disabled || manualMode}>
          Place Random
        </button>
        <button type="button" className="ghost-btn" onClick={onUndo} disabled={disabled}>
          Undo
        </button>
        <button type="button" className="ghost-btn" onClick={onClear} disabled={disabled}>
          Clear
        </button>
      </div>
    </section>
  );
}
