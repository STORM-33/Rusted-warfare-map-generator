"use client";

type StepBarProps = {
  steps: readonly string[];
  currentStep: number;
  completedStep: number;
  onSelect: (step: number) => void;
  isStepAccessible: (step: number) => boolean;
};

export function StepBar({
  steps,
  currentStep,
  completedStep,
  onSelect,
  isStepAccessible,
}: StepBarProps) {
  return (
    <nav className="step-bar">
      {steps.map((label, index) => {
        const active = index === currentStep;
        const complete = index <= completedStep;
        const disabled = !isStepAccessible(index);
        return (
          <button
            key={label}
            type="button"
            className={`step-btn ${active ? "active" : ""} ${complete ? "complete" : ""}`}
            disabled={disabled}
            onClick={() => onSelect(index)}
          >
            <span className="step-index">{index + 1}</span>
            <span>{label}</span>
          </button>
        );
      })}
    </nav>
  );
}
