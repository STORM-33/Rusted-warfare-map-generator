"use client";

type CommandCenterStepProps = {
  numPlayers: number;
  manualMode: boolean;
  mirrored: boolean;
  disabled?: boolean;
  onNumPlayersChange: (value: number) => void;
  onManualModeChange: (manual: boolean) => void;
  onMirroredChange: (mirrored: boolean) => void;
  onRandom: () => void;
  onUndo: () => void;
  onClear: () => void;
};

export function CommandCenterStep({
  numPlayers,
  manualMode,
  mirrored,
  disabled,
  onNumPlayersChange,
  onManualModeChange,
  onMirroredChange,
  onRandom,
  onUndo,
  onClear,
}: CommandCenterStepProps) {
  return (
    <section className="panel-section">
      <h2>4. Command Centers</h2>
      <p>Use manual mode to click on the map, or random placement for mirrored players.</p>
      <div className="control-grid">
        <label>
          Num Players
          <input
            type="number"
            min={2}
            max={10}
            value={numPlayers}
            onChange={(event) => onNumPlayersChange(Number(event.target.value))}
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
