"use client";

import type { RenderMode } from "@/lib/types";

type RenderPreference = RenderMode | "auto";

type FinalizeStepProps = {
  disabled?: boolean;
  renderPreference: RenderPreference;
  onRenderPreferenceChange: (mode: RenderPreference) => void;
  onExport: () => void;
};

export function FinalizeStep({
  disabled,
  renderPreference,
  onRenderPreferenceChange,
  onExport,
}: FinalizeStepProps) {
  return (
    <section className="panel-section">
      <h2>6. Finalize</h2>
      <p>Generate final terrain tiles and download generated_map.tmx.</p>
      <label>
        Preview Mode
        <select
          value={renderPreference}
          onChange={(event) => onRenderPreferenceChange(event.target.value as RenderPreference)}
          disabled={disabled}
        >
          <option value="auto">Auto</option>
          <option value="sampled">Sampled</option>
          <option value="full">Full</option>
        </select>
      </label>
      <button type="button" className="primary-btn" onClick={onExport} disabled={disabled}>
        Finalize & Download TMX
      </button>
    </section>
  );
}
