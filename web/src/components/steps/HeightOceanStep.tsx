"use client";

import { NumberInput } from "@/components/NumberInput";

type HeightOceanStepProps = {
  heightLevels: number;
  oceanLevels: number;
  wallMagnetism: number;
  disabled?: boolean;
  onHeightLevelsChange: (value: number) => void;
  onOceanLevelsChange: (value: number) => void;
  onWallMagnetismChange: (value: number) => void;
  onGenerate: () => void;
};

export function HeightOceanStep({
  heightLevels,
  oceanLevels,
  wallMagnetism,
  disabled,
  onHeightLevelsChange,
  onOceanLevelsChange,
  onWallMagnetismChange,
  onGenerate,
}: HeightOceanStepProps) {
  return (
    <section className="panel-section">
      <h2>3. Height / Ocean</h2>
      <p>Generate terrain levels from Perlin noise on top of coastline + hill walls.</p>
      <div className="control-grid">
        <label>
          Height Levels
          <NumberInput value={heightLevels} min={1} max={7} onChange={onHeightLevelsChange} disabled={disabled} />
        </label>
        <label>
          Ocean Levels
          <NumberInput value={oceanLevels} min={1} max={3} onChange={onOceanLevelsChange} disabled={disabled} />
        </label>
      </div>
      <label className="slider-field">
        <span className="slider-label">
          Wall Magnetism
          <span className="slider-value">{wallMagnetism}</span>
        </span>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={wallMagnetism}
          onChange={(event) => onWallMagnetismChange(Number(event.target.value))}
          disabled={disabled}
        />
        <span className="slider-hint">
          0 = terrain ignores walls; higher pulls level changes onto wall lines.
        </span>
      </label>
      <button type="button" className="primary-btn" onClick={onGenerate} disabled={disabled}>
        Generate Height & Ocean
      </button>
    </section>
  );
}
