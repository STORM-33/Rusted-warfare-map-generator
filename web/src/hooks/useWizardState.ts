"use client";

import { useCallback, useMemo, useState } from "react";

export const WIZARD_STEP_LABELS = [
  "Coastline",
  "Hills",
  "Height/Ocean",
  "Command Centers",
  "Resources",
  "Finalize",
] as const;

export function useWizardState(totalSteps = WIZARD_STEP_LABELS.length) {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedStep, setCompletedStep] = useState(-1);

  const maxReachableStep = Math.min(totalSteps - 1, completedStep + 1);

  const goToStep = useCallback(
    (step: number) => {
      if (step < 0 || step >= totalSteps) {
        return;
      }
      if (step <= maxReachableStep) {
        setCurrentStep(step);
      }
    },
    [maxReachableStep, totalSteps],
  );

  const nextStep = useCallback(() => {
    setCurrentStep((prev) => Math.min(totalSteps - 1, prev + 1));
  }, [totalSteps]);

  const prevStep = useCallback(() => {
    setCurrentStep((prev) => Math.max(0, prev - 1));
  }, []);

  const markStepComplete = useCallback((step?: number) => {
    setCompletedStep((prev) => Math.max(prev, step ?? currentStep));
  }, [currentStep]);

  const isStepAccessible = useCallback(
    (step: number) => step <= maxReachableStep,
    [maxReachableStep],
  );

  const steps = useMemo(() => [...WIZARD_STEP_LABELS], []);

  return {
    steps,
    currentStep,
    completedStep,
    maxReachableStep,
    goToStep,
    nextStep,
    prevStep,
    markStepComplete,
    isStepAccessible,
  };
}
