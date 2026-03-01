"use client";

type FinalizeStepProps = {
  disabled?: boolean;
  onExport: () => void;
};

export function FinalizeStep({
  disabled,
  onExport,
}: FinalizeStepProps) {
  return (
    <section className="panel-section">
      <h2>6. Finalize</h2>
      <p>Generate final terrain tiles and download a zip with the map (.tmx) and thumbnail (.png).</p>
      <button type="button" className="primary-btn" onClick={onExport} disabled={disabled}>
        Finalize & Download ZIP
      </button>
    </section>
  );
}
