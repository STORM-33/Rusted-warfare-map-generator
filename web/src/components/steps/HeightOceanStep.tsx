"use client";

import { NumberInput } from "@/components/NumberInput";

type HeightOceanStepProps = {
  heightLevels: number;
  oceanLevels: number;
  disabled?: boolean;
  onHeightLevelsChange: (value: number) => void;
  onOceanLevelsChange: (value: number) => void;
  onGenerate: () => void;
};

export function HeightOceanStep({
  heightLevels,
  oceanLevels,
  disabled,
  onHeightLevelsChange,
  onOceanLevelsChange,
  onGenerate,
}: HeightOceanStepProps) {
  return (
    <section className="panel-section">
      <h2>3. Height / Ocean</h2>
      <p>Generate terrain levels from Perlin noise on top of coastline + hill walls.</p>
      <div className="control-grid">
        <label>
          Height Levels
          <NumberInput value={heightLevels} min={1} max={20} onChange={onHeightLevelsChange} disabled={disabled} />
        </label>
        <label>
          Ocean Levels
          <NumberInput value={oceanLevels} min={1} max={20} onChange={onOceanLevelsChange} disabled={disabled} />
        </label>
      </div>
      <button type="button" className="primary-btn" onClick={onGenerate} disabled={disabled}>
        Generate Height & Ocean
      </button>
    </section>
  );
}
